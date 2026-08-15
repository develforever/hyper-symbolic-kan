#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include "fast_kan_kernel.hpp"

namespace nb = nanobind;

void py_evaluate_tt_kan_batch(
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> X,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> cores_flat,
    nb::ndarray<const int, nb::c_contig, nb::device::cpu> core_offsets,
    nb::ndarray<const int, nb::c_contig, nb::device::cpu> ranks,
    int degree,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> out_Y
) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2D array of shape (N, D)");
    }
    const int N = static_cast<int>(X.shape(0));
    const int D = static_cast<int>(X.shape(1));

    if (static_cast<int>(ranks.size()) != D + 1) {
        throw std::invalid_argument("ranks must have size D + 1");
    }
    if (static_cast<int>(out_Y.size()) < N) {
        throw std::invalid_argument("out_Y must have size >= N");
    }

    const double* x_ptr = X.data();
    const double* cores_ptr = cores_flat.data();
    const int* offsets_ptr = core_offsets.data();
    const int* ranks_ptr = ranks.data();
    double* out_ptr = out_Y.data();

    // Zwolnij GIL na czas obliczeń wielowątkowych C++
    nb::gil_scoped_release release;
    hs_kan::evaluate_tt_kan_batch(
        x_ptr, cores_ptr, offsets_ptr, ranks_ptr, N, D, degree, out_ptr
    );
}

void py_evaluate_tt_kan_gradient_batch(
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> X,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> cores_flat,
    nb::ndarray<const int, nb::c_contig, nb::device::cpu> core_offsets,
    nb::ndarray<const int, nb::c_contig, nb::device::cpu> ranks,
    int degree,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> out_grad
) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2D array of shape (N, D)");
    }
    const int N = static_cast<int>(X.shape(0));
    const int D = static_cast<int>(X.shape(1));

    if (static_cast<int>(ranks.size()) != D + 1) {
        throw std::invalid_argument("ranks must have size D + 1");
    }
    if (static_cast<int>(out_grad.size()) < N * D) {
        throw std::invalid_argument("out_grad must have size >= N * D");
    }

    const double* x_ptr = X.data();
    const double* cores_ptr = cores_flat.data();
    const int* offsets_ptr = core_offsets.data();
    const int* ranks_ptr = ranks.data();
    double* grad_ptr = out_grad.data();

    nb::gil_scoped_release release;
    hs_kan::evaluate_tt_kan_gradient_batch(
        x_ptr, cores_ptr, offsets_ptr, ranks_ptr, N, D, degree, grad_ptr
    );
}

void py_evaluate_cp_kan_batch(
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> X,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> factors_flat,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> lambdas,
    int rank,
    int degree,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> out_Y
) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2D array of shape (N, D)");
    }
    const int N = static_cast<int>(X.shape(0));
    const int D = static_cast<int>(X.shape(1));

    if (static_cast<int>(lambdas.size()) != rank) {
        throw std::invalid_argument("lambdas must have size == rank");
    }
    if (static_cast<int>(factors_flat.size()) < D * rank * (degree + 1)) {
        throw std::invalid_argument("factors_flat size does not match D * rank * (degree + 1)");
    }
    if (static_cast<int>(out_Y.size()) < N) {
        throw std::invalid_argument("out_Y must have size >= N");
    }

    const double* x_ptr = X.data();
    const double* factors_ptr = factors_flat.data();
    const double* lambdas_ptr = lambdas.data();
    double* out_ptr = out_Y.data();

    nb::gil_scoped_release release;
    hs_kan::evaluate_cp_kan_batch(
        x_ptr, factors_ptr, lambdas_ptr, N, D, rank, degree, out_ptr
    );
}

void py_evaluate_cp_kan_gradient_batch(
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> X,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> factors_flat,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> lambdas,
    int rank,
    int degree,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> out_grad
) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2D array of shape (N, D)");
    }
    const int N = static_cast<int>(X.shape(0));
    const int D = static_cast<int>(X.shape(1));

    if (static_cast<int>(lambdas.size()) != rank) {
        throw std::invalid_argument("lambdas must have size == rank");
    }
    if (static_cast<int>(out_grad.size()) < N * D) {
        throw std::invalid_argument("out_grad must have size >= N * D");
    }

    const double* x_ptr = X.data();
    const double* factors_ptr = factors_flat.data();
    const double* lambdas_ptr = lambdas.data();
    double* grad_ptr = out_grad.data();

    nb::gil_scoped_release release;
    hs_kan::evaluate_cp_kan_gradient_batch(
        x_ptr, factors_ptr, lambdas_ptr, N, D, rank, degree, grad_ptr
    );
}

void py_project_chebyshev_modal_batch(
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> nodal_core,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> V_inv,
    int r_prev,
    int K1,
    int r_next,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> modal_core_out
) {
    if (static_cast<int>(nodal_core.size()) < r_prev * K1 * r_next) {
        throw std::invalid_argument("nodal_core size does not match r_prev * K1 * r_next");
    }
    if (static_cast<int>(V_inv.size()) < K1 * K1) {
        throw std::invalid_argument("V_inv size does not match K1 * K1");
    }
    if (static_cast<int>(modal_core_out.size()) < r_prev * K1 * r_next) {
        throw std::invalid_argument("modal_core_out size does not match r_prev * K1 * r_next");
    }

    const double* n_ptr = nodal_core.data();
    const double* v_ptr = V_inv.data();
    double* out_ptr = modal_core_out.data();

    nb::gil_scoped_release release;
    hs_kan::project_chebyshev_modal_batch(n_ptr, v_ptr, r_prev, K1, r_next, out_ptr);
}

void py_build_dmrg_normal_equations_batch(
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> L_prev,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> T_d,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> T_d1,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> R_next,
    nb::ndarray<const double, nb::c_contig, nb::device::cpu> Y,
    int r_prev,
    int K1,
    int r_next,
    double alpha,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> A_out,
    nb::ndarray<double, nb::c_contig, nb::device::cpu> B_out
) {
    const int N = static_cast<int>(Y.size());
    const int P = r_prev * K1 * K1 * r_next;

    if (static_cast<int>(A_out.size()) < P * P) {
        throw std::invalid_argument("A_out size must be at least P * P");
    }
    if (static_cast<int>(B_out.size()) < P) {
        throw std::invalid_argument("B_out size must be at least P");
    }

    const double* l_ptr = L_prev.data();
    const double* t1_ptr = T_d.data();
    const double* t2_ptr = T_d1.data();
    const double* r_ptr = R_next.data();
    const double* y_ptr = Y.data();
    double* a_ptr = A_out.data();
    double* b_ptr = B_out.data();

    nb::gil_scoped_release release;
    hs_kan::build_dmrg_normal_equations_batch(
        l_ptr, t1_ptr, t2_ptr, r_ptr, y_ptr, N, r_prev, K1, r_next, alpha, a_ptr, b_ptr
    );
}

NB_MODULE(_cpp_kernels, m) {
    m.doc() = "Hyper-Symbolic KAN Ultra-Fast C++ / SIMD Kernels (nanobind + OpenMP)";

    m.def(
        "evaluate_tt_kan_batch",
        &py_evaluate_tt_kan_batch,
        nb::arg("X"),
        nb::arg("cores_flat"),
        nb::arg("core_offsets"),
        nb::arg("ranks"),
        nb::arg("degree"),
        nb::arg("out_Y"),
        "High-throughput batch evaluation of TT-KAN fields in C++ with OpenMP and SIMD."
    );

    m.def(
        "evaluate_tt_kan_gradient_batch",
        &py_evaluate_tt_kan_gradient_batch,
        nb::arg("X"),
        nb::arg("cores_flat"),
        nb::arg("core_offsets"),
        nb::arg("ranks"),
        nb::arg("degree"),
        nb::arg("out_grad"),
        "High-throughput analytical gradient evaluation for TT-KAN fields in C++."
    );

    m.def(
        "evaluate_cp_kan_batch",
        &py_evaluate_cp_kan_batch,
        nb::arg("X"),
        nb::arg("factors_flat"),
        nb::arg("lambdas"),
        nb::arg("rank"),
        nb::arg("degree"),
        nb::arg("out_Y"),
        "High-throughput batch evaluation of CP-KAN (TDFF-Net) fields in C++."
    );

    m.def(
        "evaluate_cp_kan_gradient_batch",
        &py_evaluate_cp_kan_gradient_batch,
        nb::arg("X"),
        nb::arg("factors_flat"),
        nb::arg("lambdas"),
        nb::arg("rank"),
        nb::arg("degree"),
        nb::arg("out_grad"),
        "High-throughput analytical gradient evaluation of CP-KAN fields in C++."
    );

    m.def(
        "project_chebyshev_modal_batch",
        &py_project_chebyshev_modal_batch,
        nb::arg("nodal_core"),
        nb::arg("V_inv"),
        nb::arg("r_prev"),
        nb::arg("K1"),
        nb::arg("r_next"),
        nb::arg("modal_core_out"),
        "SIMD-accelerated modal Chebyshev basis projection (V_inv @ Nodal)."
    );

    m.def(
        "build_dmrg_normal_equations_batch",
        &py_build_dmrg_normal_equations_batch,
        nb::arg("L_prev"),
        nb::arg("T_d"),
        nb::arg("T_d1"),
        nb::arg("R_next"),
        nb::arg("Y"),
        nb::arg("r_prev"),
        nb::arg("K1"),
        nb::arg("r_next"),
        nb::arg("alpha"),
        nb::arg("A_out"),
        nb::arg("B_out"),
        "Direct multi-threaded 2-Site DMRG normal equations accumulator in C++."
    );
}
