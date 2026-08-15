#pragma once

#include <cstddef>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace hs_kan {

// Oblicza wielomiany Czebyszewa T_0 ... T_K dla podanego x w zakresie [-1.0, 1.0]
inline void compute_chebyshev_scalar(double x, int degree, double* __restrict T) {
    if (x < -1.0) x = -1.0;
    if (x > 1.0) x = 1.0;

    T[0] = 1.0;
    if (degree >= 1) T[1] = x;
    for (int k = 1; k < degree; ++k) {
        T[k + 1] = 2.0 * x * T[k] - T[k - 1];
    }
}

// Oblicza wielomiany Czebyszewa T_k oraz ich analityczne pochodne dT_k/dx
inline void compute_chebyshev_and_deriv_scalar(
    double x, int degree, double* __restrict T, double* __restrict dT
) {
    if (x < -1.0) x = -1.0;
    if (x > 1.0) x = 1.0;

    T[0] = 1.0;
    dT[0] = 0.0;
    if (degree >= 1) {
        T[1] = x;
        dT[1] = 1.0;
    }
    for (int k = 1; k < degree; ++k) {
        T[k + 1] = 2.0 * x * T[k] - T[k - 1];
        dT[k + 1] = 2.0 * T[k] + 2.0 * x * dT[k] - dT[k - 1];
    }
}

// 1. TT-KAN Forward (Pojedynczy punkt)
void evaluate_tt_kan_single(
    const double* __restrict x,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int spatial_dim,
    int degree,
    double* __restrict out_val
);

// 2. TT-KAN Forward (Batch z OpenMP)
void evaluate_tt_kan_batch(
    const double* __restrict X,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int N,
    int spatial_dim,
    int degree,
    double* __restrict Y_out
);

// 3. TT-KAN Gradient (Pojedynczy punkt)
void evaluate_tt_kan_gradient_single(
    const double* __restrict x,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int spatial_dim,
    int degree,
    double* __restrict grad_out
);

// 4. TT-KAN Gradient (Batch z OpenMP)
void evaluate_tt_kan_gradient_batch(
    const double* __restrict X,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int N,
    int spatial_dim,
    int degree,
    double* __restrict grad_out
);

// 5. CP-KAN (TDFF-Net) Forward (Batch z OpenMP)
void evaluate_cp_kan_batch(
    const double* __restrict X,
    const double* __restrict factors_flat,
    const double* __restrict lambdas,
    int N,
    int spatial_dim,
    int rank,
    int degree,
    double* __restrict Y_out
);

// 6. CP-KAN (TDFF-Net) Gradient (Batch z OpenMP)
void evaluate_cp_kan_gradient_batch(
    const double* __restrict X,
    const double* __restrict factors_flat,
    const double* __restrict lambdas,
    int N,
    int spatial_dim,
    int rank,
    int degree,
    double* __restrict grad_out
);

// 7. Szybka transformacja modalna Czebyszewa (Nodal -> Modal) dla rdzenia TT
void project_chebyshev_modal_batch(
    const double* __restrict nodal_core,
    const double* __restrict V_inv,
    int r_prev,
    int K1,
    int r_next,
    double* __restrict modal_core_out
);

// 8. 2-Site DMRG Normal Equations Accumulator (Phi^T Phi i Phi^T Y) bez alokacji macierzy Phi
void build_dmrg_normal_equations_batch(
    const double* __restrict L_prev,
    const double* __restrict T_d,
    const double* __restrict T_d1,
    const double* __restrict R_next,
    const double* __restrict Y,
    int N,
    int r_prev,
    int K1,
    int r_next,
    double alpha,
    double* __restrict A_out,
    double* __restrict B_out
);

} // namespace hs_kan
