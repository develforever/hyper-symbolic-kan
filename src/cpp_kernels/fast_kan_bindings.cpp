#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "fast_kan_kernel.hpp"

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Audit C2: every size product used to be evaluated in `int`, so a legal user
// configuration (r_prev = 32, degree = 7 -> P = 65536) made `P * P` overflow
// INT32_MAX. The validation `static_cast<int>(A_out.size()) < P * P` overflowed
// in exactly the same way and therefore always passed, after which the negative
// product was converted to a huge size_t inside memset -> heap corruption with
// the GIL released.
//
// The helpers below evaluate the same products in std::size_t and reject
// non-positive dimensions (a negative int would wrap to a huge size_t) before
// any kernel is entered.
// ---------------------------------------------------------------------------
namespace {

void require_positive(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be > 0, got " + std::to_string(value));
    }
}

// Overflow-safe a * b for std::size_t.
std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > static_cast<std::size_t>(-1) / a) {
        throw std::invalid_argument(std::string(what) + ": size product overflows std::size_t");
    }
    return a * b;
}

void require_size_at_least(std::size_t actual, std::size_t required, const char* name) {
    if (actual < required) {
        throw std::invalid_argument(
            std::string(name) + " must have size >= " + std::to_string(required) +
            ", got " + std::to_string(actual));
    }
}

}  // namespace

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

    if (static_cast<std::size_t>(ranks.size()) != static_cast<std::size_t>(D) + 1u) {
        throw std::invalid_argument("ranks must have size D + 1");
    }
    require_size_at_least(out_Y.size(), static_cast<std::size_t>(N), "out_Y");

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

    if (static_cast<std::size_t>(ranks.size()) != static_cast<std::size_t>(D) + 1u) {
        throw std::invalid_argument("ranks must have size D + 1");
    }
    require_size_at_least(
        out_grad.size(),
        checked_mul(static_cast<std::size_t>(N), static_cast<std::size_t>(D), "N * D"),
        "out_grad");

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

    require_positive(rank, "rank");
    require_positive(degree + 1, "degree + 1");
    if (static_cast<std::size_t>(lambdas.size()) != static_cast<std::size_t>(rank)) {
        throw std::invalid_argument("lambdas must have size == rank");
    }
    const std::size_t factors_required = checked_mul(
        checked_mul(static_cast<std::size_t>(D), static_cast<std::size_t>(rank), "D * rank"),
        static_cast<std::size_t>(degree + 1), "D * rank * (degree + 1)");
    require_size_at_least(factors_flat.size(), factors_required, "factors_flat");
    require_size_at_least(out_Y.size(), static_cast<std::size_t>(N), "out_Y");

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

    require_positive(rank, "rank");
    require_positive(degree + 1, "degree + 1");
    if (static_cast<std::size_t>(lambdas.size()) != static_cast<std::size_t>(rank)) {
        throw std::invalid_argument("lambdas must have size == rank");
    }
    require_size_at_least(
        factors_flat.size(),
        checked_mul(
            checked_mul(static_cast<std::size_t>(D), static_cast<std::size_t>(rank), "D * rank"),
            static_cast<std::size_t>(degree + 1), "D * rank * (degree + 1)"),
        "factors_flat");
    require_size_at_least(
        out_grad.size(),
        checked_mul(static_cast<std::size_t>(N), static_cast<std::size_t>(D), "N * D"),
        "out_grad");

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
    require_positive(r_prev, "r_prev");
    require_positive(K1, "K1");
    require_positive(r_next, "r_next");

    const std::size_t core_required = checked_mul(
        checked_mul(static_cast<std::size_t>(r_prev), static_cast<std::size_t>(K1), "r_prev * K1"),
        static_cast<std::size_t>(r_next), "r_prev * K1 * r_next");
    require_size_at_least(nodal_core.size(), core_required, "nodal_core");
    require_size_at_least(
        V_inv.size(),
        checked_mul(static_cast<std::size_t>(K1), static_cast<std::size_t>(K1), "K1 * K1"),
        "V_inv");
    require_size_at_least(modal_core_out.size(), core_required, "modal_core_out");

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

    require_positive(r_prev, "r_prev");
    require_positive(K1, "K1");
    require_positive(r_next, "r_next");

    const std::size_t P = checked_mul(
        checked_mul(
            checked_mul(static_cast<std::size_t>(r_prev), static_cast<std::size_t>(K1), "r_prev * K1"),
            static_cast<std::size_t>(K1), "r_prev * K1 * K1"),
        static_cast<std::size_t>(r_next), "r_prev * K1 * K1 * r_next");

    // P * P is checked by division, so the check itself cannot overflow.
    if (P > A_out.size() / P) {
        throw std::invalid_argument(
            "A_out must have size >= P * P, with P = r_prev * K1 * K1 * r_next = " +
            std::to_string(P) + ", got " + std::to_string(A_out.size()));
    }
    require_size_at_least(B_out.size(), P, "B_out");

    require_size_at_least(L_prev.size(),
        checked_mul(static_cast<std::size_t>(N), static_cast<std::size_t>(r_prev), "N * r_prev"), "L_prev");
    require_size_at_least(T_d.size(),
        checked_mul(static_cast<std::size_t>(N), static_cast<std::size_t>(K1), "N * K1"), "T_d");
    require_size_at_least(T_d1.size(),
        checked_mul(static_cast<std::size_t>(N), static_cast<std::size_t>(K1), "N * K1"), "T_d1");
    require_size_at_least(R_next.size(),
        checked_mul(static_cast<std::size_t>(N), static_cast<std::size_t>(r_next), "N * r_next"), "R_next");

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
