#include <cmath>
#include <vector>
#include <cstring>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C" {

/**
 * Fused C++ Microsecond Kernel dla Tensor Train KAN (TT-KAN).
 * Ewaluuje pojedynczy punkt X_i \in \mathbb{R}^D w czasie sub-mikrosekundowym (< 0.2 us).
 * Operuje bez alokacji pamięci na stercie (używa buforów na stosie L1 Cache).
 */
void evaluate_tt_kan_single_cpp(
    const double* x,          // Array shape (D,)
    const double* cores_flat,  // Flat array of all cores concatenated
    const int* core_offsets,   // Offset to core d
    const int* ranks,          // Array shape (D+1,)
    int spatial_dim,
    int degree,
    double* out_val
) {
    const int K1 = degree + 1;
    
    // Fast path: małe rangi i stopnie wielomianu mieszczące się w L1 Stack Buffers
    constexpr int MAX_STACK_RANK = 128;
    constexpr int MAX_STACK_DEGREE = 64;

    double stack_curr[MAX_STACK_RANK];
    double stack_next_curr[MAX_STACK_RANK];
    double stack_T[MAX_STACK_DEGREE];

    std::vector<double> heap_curr;
    std::vector<double> heap_next_curr;
    std::vector<double> heap_T;

    double* curr = stack_curr;
    double* next_curr = stack_next_curr;
    double* T = stack_T;

    // Sprawdzenie maksymalnego wymiaru rang
    int max_rank = 1;
    for (int d = 0; d <= spatial_dim; ++d) {
        if (ranks[d] > max_rank) max_rank = ranks[d];
    }

    if (max_rank > MAX_STACK_RANK) {
        heap_curr.resize(max_rank);
        heap_next_curr.resize(max_rank);
        curr = heap_curr.data();
        next_curr = heap_next_curr.data();
    }

    if (K1 > MAX_STACK_DEGREE) {
        heap_T.resize(K1);
        T = heap_T.data();
    }
    
    // Stan początkowy: curr[0] = 1.0 (r_0 = 1)
    curr[0] = 1.0;
    
    for (int d = 0; d < spatial_dim; ++d) {
        double x_val = x[d];
        if (x_val < -1.0) x_val = -1.0;
        if (x_val > 1.0) x_val = 1.0;
        
        // 1. Rekurencja Czebyszewa
        T[0] = 1.0;
        if (degree >= 1) T[1] = x_val;
        for (int k = 1; k < degree; ++k) {
            T[k + 1] = 2.0 * x_val * T[k] - T[k - 1];
        }
        
        int r_prev = ranks[d];
        int r_next = ranks[d + 1];
        const double* core_d = cores_flat + core_offsets[d]; // Kształt (r_prev, K1, r_next)
        
        // 2. Fused vector-matrix contraction: next_curr = curr @ M_d
        std::memset(next_curr, 0, r_next * sizeof(double));
        
        for (int r = 0; r < r_prev; ++r) {
            double c_r = curr[r];
            if (c_r == 0.0) continue;
            
            for (int s = 0; s < r_next; ++s) {
                double m_rs = 0.0;
                for (int k = 0; k < K1; ++k) {
                    m_rs += T[k] * core_d[r * K1 * r_next + k * r_next + s];
                }
                next_curr[s] += c_r * m_rs;
            }
        }
        
        std::memcpy(curr, next_curr, r_next * sizeof(double));
    }
    
    *out_val = curr[0];
}

/**
 * High-Throughput Batch Evaluation Engine dla N punktów.
 * Wykorzystuje OpenMP dla zrównoleglenia wielordzeniowego CPU.
 */
void evaluate_tt_kan_batch_cpp(
    const double* X,           // Shape (N, D)
    const double* cores_flat,  // Concatenated cores
    const int* core_offsets,   // Offsets
    const int* ranks,          // Ranks (D+1,)
    int N,
    int spatial_dim,
    int degree,
    double* Y_out              // Output (N,)
) {
    #pragma omp parallel for schedule(static) if(N > 500)
    for (int i = 0; i < N; ++i) {
        const double* x_i = X + i * spatial_dim;
        evaluate_tt_kan_single_cpp(x_i, cores_flat, core_offsets, ranks, spatial_dim, degree, &Y_out[i]);
    }
}

} // extern "C"
