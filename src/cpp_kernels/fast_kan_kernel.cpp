#include "fast_kan_kernel.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

namespace hs_kan {

// Constants for stack buffers (L1 Cache resident)
constexpr int MAX_STACK_RANK = 64;
constexpr int MAX_STACK_DEGREE = 32;
constexpr int MAX_STACK_DIM = 32;

void evaluate_tt_kan_single(
    const double* __restrict x,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int spatial_dim,
    int degree,
    double* __restrict out_val
) {
    const int K1 = degree + 1;

    double stack_curr[MAX_STACK_RANK];
    double stack_next_curr[MAX_STACK_RANK];
    double stack_T[MAX_STACK_DEGREE];

    std::vector<double> heap_curr;
    std::vector<double> heap_next_curr;
    std::vector<double> heap_T;

    double* curr = stack_curr;
    double* next_curr = stack_next_curr;
    double* T = stack_T;

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

    curr[0] = 1.0;

    for (int d = 0; d < spatial_dim; ++d) {
        compute_chebyshev_scalar(x[d], degree, T);

        const int r_prev = ranks[d];
        const int r_next = ranks[d + 1];
        const double* __restrict core_d = cores_flat + core_offsets[d];

        std::memset(next_curr, 0, r_next * sizeof(double));

        for (int r = 0; r < r_prev; ++r) {
            const double c_r = curr[r];
            if (c_r == 0.0) continue;

            const double* __restrict core_r = core_d + (r * K1) * r_next;

            for (int s = 0; s < r_next; ++s) {
                double m_rs = 0.0;
                #pragma loop(ivdep)
                for (int k = 0; k < K1; ++k) {
                    m_rs += T[k] * core_r[k * r_next + s];
                }
                next_curr[s] += c_r * m_rs;
            }
        }

        std::memcpy(curr, next_curr, r_next * sizeof(double));
    }

    *out_val = curr[0];
}

void evaluate_tt_kan_batch(
    const double* __restrict X,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int N,
    int spatial_dim,
    int degree,
    double* __restrict Y_out
) {
    #pragma omp parallel for schedule(static) if(N > 100)
    for (int i = 0; i < N; ++i) {
        const double* x_i = X + i * spatial_dim;
        evaluate_tt_kan_single(x_i, cores_flat, core_offsets, ranks, spatial_dim, degree, &Y_out[i]);
    }
}

void evaluate_tt_kan_gradient_single(
    const double* __restrict x,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int spatial_dim,
    int degree,
    double* __restrict grad_out
) {
    const int K1 = degree + 1;
    const int D = spatial_dim;

    int max_rank = 1;
    int total_M_elements = 0;
    for (int d = 0; d < D; ++d) {
        if (ranks[d] > max_rank) max_rank = ranks[d];
        total_M_elements += ranks[d] * ranks[d + 1];
    }
    if (ranks[D] > max_rank) max_rank = ranks[D];

    bool use_stack = (D <= MAX_STACK_DIM) && (max_rank <= MAX_STACK_RANK) && 
                     (K1 <= MAX_STACK_DEGREE) && (total_M_elements <= 2048);

    // Stack memory
    double stack_T[MAX_STACK_DEGREE];
    double stack_dT[MAX_STACK_DEGREE];
    double stack_M[2048];
    double stack_dM[2048];
    double stack_L[MAX_STACK_DIM * MAX_STACK_RANK];
    double stack_R[MAX_STACK_DIM * MAX_STACK_RANK];
    double stack_temp[MAX_STACK_RANK];

    // Heap memory fallback
    std::vector<double> heap_T, heap_dT, heap_M, heap_dM, heap_L, heap_R, heap_temp;

    double* T = stack_T;
    double* dT = stack_dT;
    double* M_buf = stack_M;
    double* dM_buf = stack_dM;
    double* L_buf = stack_L;
    double* R_buf = stack_R;
    double* temp = stack_temp;

    if (!use_stack) {
        heap_T.resize(K1);
        heap_dT.resize(K1);
        heap_M.resize(total_M_elements);
        heap_dM.resize(total_M_elements);
        heap_L.resize(D * max_rank);
        heap_R.resize(D * max_rank);
        heap_temp.resize(max_rank);

        T = heap_T.data();
        dT = heap_dT.data();
        M_buf = heap_M.data();
        dM_buf = heap_dM.data();
        L_buf = heap_L.data();
        R_buf = heap_R.data();
        temp = heap_temp.data();
    }

    // 1. Oblicz M_d i dM_d dla każdego wymiaru
    int m_offset = 0;
    std::vector<int> m_offsets(D);
    for (int d = 0; d < D; ++d) {
        m_offsets[d] = m_offset;
        compute_chebyshev_and_deriv_scalar(x[d], degree, T, dT);

        const int r_prev = ranks[d];
        const int r_next = ranks[d + 1];
        const double* __restrict core_d = cores_flat + core_offsets[d];
        double* __restrict M_d = M_buf + m_offset;
        double* __restrict dM_d = dM_buf + m_offset;

        for (int r = 0; r < r_prev; ++r) {
            const double* __restrict core_r = core_d + (r * K1) * r_next;
            for (int s = 0; s < r_next; ++s) {
                double m_val = 0.0;
                double dm_val = 0.0;
                #pragma loop(ivdep)
                for (int k = 0; k < K1; ++k) {
                    const double w = core_r[k * r_next + s];
                    m_val += T[k] * w;
                    dm_val += dT[k] * w;
                }
                M_d[r * r_next + s] = m_val;
                dM_d[r * r_next + s] = dm_val;
            }
        }
        m_offset += r_prev * r_next;
    }

    // 2. Prefiksy lewe L^(d) (shape: r_{d+1})
    // L[d] znajduje się w L_buf + d * max_rank
    double* L_prev_ptr = nullptr;
    for (int d = 0; d < D; ++d) {
        const int r_prev = ranks[d];
        const int r_next = ranks[d + 1];
        const double* __restrict M_d = M_buf + m_offsets[d];
        double* __restrict L_curr = L_buf + d * max_rank;

        std::memset(L_curr, 0, r_next * sizeof(double));

        if (d == 0) {
            // L^{(-1)} = [1.0] -> L^(0) to pierwszy wiersz M_0 (r=0)
            for (int s = 0; s < r_next; ++s) {
                L_curr[s] = M_d[s];
            }
        } else {
            // L^(d) = L^(d-1) @ M_d
            for (int r = 0; r < r_prev; ++r) {
                const double l_r = L_prev_ptr[r];
                if (l_r == 0.0) continue;
                for (int s = 0; s < r_next; ++s) {
                    L_curr[s] += l_r * M_d[r * r_next + s];
                }
            }
        }
        L_prev_ptr = L_curr;
    }

    // 3. Sufiksy prawe R^(d) (shape: r_d)
    // R[d] znajduje się w R_buf + d * max_rank
    double* R_next_ptr = nullptr;
    for (int d = D - 1; d >= 0; --d) {
        const int r_curr = ranks[d];
        const int r_next = ranks[d + 1];
        const double* __restrict M_d = M_buf + m_offsets[d];
        double* __restrict R_d = R_buf + d * max_rank;

        std::memset(R_d, 0, r_curr * sizeof(double));

        if (d == D - 1) {
            // R^(D) = [1.0] -> R^(D-1) to kolumna s=0 z M_{D-1}
            for (int r = 0; r < r_curr; ++r) {
                R_d[r] = M_d[r * r_next + 0];
            }
        } else {
            // R^(d) = M_d @ R^(d+1)
            for (int r = 0; r < r_curr; ++r) {
                double val = 0.0;
                for (int s = 0; s < r_next; ++s) {
                    val += M_d[r * r_next + s] * R_next_ptr[s];
                }
                R_d[r] = val;
            }
        }
        R_next_ptr = R_d;
    }

    // 4. Złożenie analitycznego gradientu df/dx_m = L_{m-1} @ dM_m @ R_{m+1}
    for (int m = 0; m < D; ++m) {
        const int r_prev = ranks[m];
        const int r_next = ranks[m + 1];
        const double* __restrict dM_m = dM_buf + m_offsets[m];

        const double* __restrict L_prev = (m == 0) ? nullptr : (L_buf + (m - 1) * max_rank);
        const double* __restrict R_next = (m == D - 1) ? nullptr : (R_buf + (m + 1) * max_rank);

        // temp = L_{m-1} @ dM_m (shape r_next)
        if (m == 0) {
            // L_{-1} = [1.0]
            for (int s = 0; s < r_next; ++s) {
                temp[s] = dM_m[0 * r_next + s];
            }
        } else {
            std::memset(temp, 0, r_next * sizeof(double));
            for (int r = 0; r < r_prev; ++r) {
                const double l_r = L_prev[r];
                if (l_r == 0.0) continue;
                for (int s = 0; s < r_next; ++s) {
                    temp[s] += l_r * dM_m[r * r_next + s];
                }
            }
        }

        // df_dxm = temp @ R_{m+1}
        double df_dxm = 0.0;
        if (m == D - 1) {
            // R_D = [1.0]
            df_dxm = temp[0];
        } else {
            for (int s = 0; s < r_next; ++s) {
                df_dxm += temp[s] * R_next[s];
            }
        }

        grad_out[m] = df_dxm;
    }
}

void evaluate_tt_kan_gradient_batch(
    const double* __restrict X,
    const double* __restrict cores_flat,
    const int* __restrict core_offsets,
    const int* __restrict ranks,
    int N,
    int spatial_dim,
    int degree,
    double* __restrict grad_out
) {
    #pragma omp parallel for schedule(static) if(N > 100)
    for (int i = 0; i < N; ++i) {
        const double* x_i = X + i * spatial_dim;
        double* grad_i = grad_out + i * spatial_dim;
        evaluate_tt_kan_gradient_single(x_i, cores_flat, core_offsets, ranks, spatial_dim, degree, grad_i);
    }
}

void evaluate_cp_kan_batch(
    const double* __restrict X,
    const double* __restrict factors_flat,
    const double* __restrict lambdas,
    int N,
    int spatial_dim,
    int rank,
    int degree,
    double* __restrict Y_out
) {
    const int K1 = degree + 1;
    const int D = spatial_dim;
    const int R = rank;

    #pragma omp parallel for schedule(static) if(N > 100)
    for (int i = 0; i < N; ++i) {
        const double* __restrict x_i = X + i * D;

        double stack_accum[MAX_STACK_RANK];
        double stack_T[MAX_STACK_DEGREE];
        std::vector<double> heap_accum, heap_T;

        double* accum = stack_accum;
        double* T = stack_T;

        if (R > MAX_STACK_RANK) {
            heap_accum.resize(R);
            accum = heap_accum.data();
        }
        if (K1 > MAX_STACK_DEGREE) {
            heap_T.resize(K1);
            T = heap_T.data();
        }

        for (int r = 0; r < R; ++r) {
            accum[r] = 1.0;
        }

        for (int d = 0; d < D; ++d) {
            compute_chebyshev_scalar(x_i[d], degree, T);
            const double* __restrict W_d = factors_flat + (d * R * K1);

            for (int r = 0; r < R; ++r) {
                const double* __restrict W_dr = W_d + (r * K1);
                double phi = 0.0;
                #pragma loop(ivdep)
                for (int k = 0; k < K1; ++k) {
                    phi += W_dr[k] * T[k];
                }
                accum[r] *= phi;
            }
        }

        double f_val = 0.0;
        for (int r = 0; r < R; ++r) {
            f_val += accum[r] * lambdas[r];
        }
        Y_out[i] = f_val;
    }
}

void evaluate_cp_kan_gradient_batch(
    const double* __restrict X,
    const double* __restrict factors_flat,
    const double* __restrict lambdas,
    int N,
    int spatial_dim,
    int rank,
    int degree,
    double* __restrict grad_out
) {
    const int K1 = degree + 1;
    const int D = spatial_dim;
    const int R = rank;

    #pragma omp parallel for schedule(static) if(N > 100)
    for (int i = 0; i < N; ++i) {
        const double* __restrict x_i = X + i * D;
        double* __restrict grad_i = grad_out + i * D;

        bool use_stack = (D <= MAX_STACK_DIM) && (R <= MAX_STACK_RANK) && (K1 <= MAX_STACK_DEGREE);

        double stack_T[MAX_STACK_DEGREE];
        double stack_dT[MAX_STACK_DEGREE];
        double stack_phi[MAX_STACK_DIM * MAX_STACK_RANK];
        double stack_dphi[MAX_STACK_DIM * MAX_STACK_RANK];
        double stack_pref[MAX_STACK_DIM * MAX_STACK_RANK];
        double stack_suff[MAX_STACK_DIM * MAX_STACK_RANK];

        std::vector<double> heap_T, heap_dT, heap_phi, heap_dphi, heap_pref, heap_suff;

        double* T = stack_T;
        double* dT = stack_dT;
        double* phi = stack_phi;
        double* dphi = stack_dphi;
        double* pref = stack_pref;
        double* suff = stack_suff;

        if (!use_stack) {
            heap_T.resize(K1);
            heap_dT.resize(K1);
            heap_phi.resize(D * R);
            heap_dphi.resize(D * R);
            heap_pref.resize(D * R);
            heap_suff.resize(D * R);

            T = heap_T.data();
            dT = heap_dT.data();
            phi = heap_phi.data();
            dphi = heap_dphi.data();
            pref = heap_pref.data();
            suff = heap_suff.data();
        }

        // 1. Oblicz phi i dphi dla wszystkich (d, r)
        for (int d = 0; d < D; ++d) {
            compute_chebyshev_and_deriv_scalar(x_i[d], degree, T, dT);
            const double* __restrict W_d = factors_flat + (d * R * K1);
            double* __restrict phi_d = phi + (d * R);
            double* __restrict dphi_d = dphi + (d * R);

            for (int r = 0; r < R; ++r) {
                const double* __restrict W_dr = W_d + (r * K1);
                double p_val = 0.0;
                double dp_val = 0.0;
                #pragma loop(ivdep)
                for (int k = 0; k < K1; ++k) {
                    const double w = W_dr[k];
                    p_val += w * T[k];
                    dp_val += w * dT[k];
                }
                phi_d[r] = p_val;
                dphi_d[r] = dp_val;
            }
        }

        // 2. Prefiksy wzdłuż d: pref[d, r] = prod_{j=0}^{d-1} phi[j, r]
        for (int r = 0; r < R; ++r) {
            pref[0 * R + r] = 1.0;
        }
        for (int d = 1; d < D; ++d) {
            const double* __restrict prev_pref = pref + ((d - 1) * R);
            const double* __restrict prev_phi = phi + ((d - 1) * R);
            double* __restrict curr_pref = pref + (d * R);
            for (int r = 0; r < R; ++r) {
                curr_pref[r] = prev_pref[r] * prev_phi[r];
            }
        }

        // 3. Sufiksy wzdłuż d: suff[d, r] = prod_{j=d+1}^{D-1} phi[j, r]
        for (int r = 0; r < R; ++r) {
            suff[(D - 1) * R + r] = 1.0;
        }
        for (int d = D - 2; d >= 0; --d) {
            const double* __restrict next_suff = suff + ((d + 1) * R);
            const double* __restrict next_phi = phi + ((d + 1) * R);
            double* __restrict curr_suff = suff + (d * R);
            for (int r = 0; r < R; ++r) {
                curr_suff[r] = next_suff[r] * next_phi[r];
            }
        }

        // 4. Gradient po wymiarze m: sum_r lambda_r * dphi[m, r] * pref[m, r] * suff[m, r]
        for (int m = 0; m < D; ++m) {
            const double* __restrict dphi_m = dphi + (m * R);
            const double* __restrict pref_m = pref + (m * R);
            const double* __restrict suff_m = suff + (m * R);

            double g_m = 0.0;
            for (int r = 0; r < R; ++r) {
                g_m += lambdas[r] * dphi_m[r] * pref_m[r] * suff_m[r];
            }
            grad_i[m] = g_m;
        }
    }
}

} // namespace hs_kan

// C ABI compatibility wrappers for backwards compatibility
extern "C" {

void evaluate_tt_kan_single_cpp(
    const double* x,
    const double* cores_flat,
    const int* core_offsets,
    const int* ranks,
    int spatial_dim,
    int degree,
    double* out_val
) {
    hs_kan::evaluate_tt_kan_single(x, cores_flat, core_offsets, ranks, spatial_dim, degree, out_val);
}

void evaluate_tt_kan_batch_cpp(
    const double* X,
    const double* cores_flat,
    const int* core_offsets,
    const int* ranks,
    int N,
    int spatial_dim,
    int degree,
    double* Y_out
) {
    hs_kan::evaluate_tt_kan_batch(X, cores_flat, core_offsets, ranks, N, spatial_dim, degree, Y_out);
}

void evaluate_tt_kan_gradient_batch_cpp(
    const double* X,
    const double* cores_flat,
    const int* core_offsets,
    const int* ranks,
    int N,
    int spatial_dim,
    int degree,
    double* grad_out
) {
    hs_kan::evaluate_tt_kan_gradient_batch(X, cores_flat, core_offsets, ranks, N, spatial_dim, degree, grad_out);
}

void evaluate_cp_kan_batch_cpp(
    const double* X,
    const double* factors_flat,
    const double* lambdas,
    int N,
    int spatial_dim,
    int rank,
    int degree,
    double* Y_out
) {
    hs_kan::evaluate_cp_kan_batch(X, factors_flat, lambdas, N, spatial_dim, rank, degree, Y_out);
}

void evaluate_cp_kan_gradient_batch_cpp(
    const double* X,
    const double* factors_flat,
    const double* lambdas,
    int N,
    int spatial_dim,
    int rank,
    int degree,
    double* grad_out
) {
    hs_kan::evaluate_cp_kan_gradient_batch(X, factors_flat, lambdas, N, spatial_dim, rank, degree, grad_out);
}

} // extern "C"
