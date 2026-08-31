r"""Regression tests for audit C2: signed-integer overflow in size arithmetic.

Before the fix, ``build_dmrg_normal_equations_batch`` computed

    const int P = r_prev * K1 * K1 * r_next;   // 32 * 8 * 8 * 32 = 65536
    ...
    std::memset(A_out, 0, P * P * sizeof(double));

``P * P`` = 4 294 967 296 > INT32_MAX, i.e. signed overflow (undefined behaviour)
for a legal user configuration (TT rank 32, Chebyshev degree 7). The binding's
own bounds check, ``static_cast<int>(A_out.size()) < P * P``, overflowed
identically and therefore always passed; the negative product was then converted
to a huge ``size_t`` in ``memset`` -- with the GIL released, so nothing on the
Python side could catch it.

These tests pin the fixed behaviour: a size that cannot be honoured is rejected
with ``ValueError`` before any kernel runs. They are the pytest counterpart of
the ASan/UBSan CI job (``.github/workflows/ci.yml``, job ``sanitizers``), which
catches the same defect at the instruction level.
"""

import numpy as np
import pytest

from tests._native import requires_native

pytestmark = requires_native


def _native():
    from src.cpp_kernels import _cpp_kernels

    return _cpp_kernels


def test_dmrg_normal_equations_rejects_int_overflowing_P():
    """P = 65536 -> P*P overflows int32. Must raise, not corrupt the heap."""
    native = _native()

    r_prev, K1, r_next = 32, 8, 32
    P = r_prev * K1 * K1 * r_next
    assert P * P > 2**31 - 1, "test configuration no longer reproduces the overflow"

    N = 4
    L_prev = np.zeros((N, r_prev), dtype=np.float64)
    T_d = np.zeros((N, K1), dtype=np.float64)
    T_d1 = np.zeros((N, K1), dtype=np.float64)
    R_next = np.zeros((N, r_next), dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)

    # Deliberately far too small for P * P; the old check accepted it.
    A_out = np.zeros(1024, dtype=np.float64)
    B_out = np.zeros(P, dtype=np.float64)

    with pytest.raises(ValueError):
        native.build_dmrg_normal_equations_batch(
            L_prev, T_d, T_d1, R_next, Y, r_prev, K1, r_next, 1e-6, A_out, B_out
        )


def test_dmrg_normal_equations_rejects_undersized_outputs():
    """The same guard for a P that does not overflow -- ordinary bounds check."""
    native = _native()

    r_prev, K1, r_next = 2, 3, 2
    P = r_prev * K1 * K1 * r_next
    N = 5

    L_prev = np.zeros((N, r_prev), dtype=np.float64)
    T_d = np.zeros((N, K1), dtype=np.float64)
    T_d1 = np.zeros((N, K1), dtype=np.float64)
    R_next = np.zeros((N, r_next), dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)

    with pytest.raises(ValueError):
        native.build_dmrg_normal_equations_batch(
            L_prev, T_d, T_d1, R_next, Y, r_prev, K1, r_next, 1e-6,
            np.zeros(P * P - 1, dtype=np.float64), np.zeros(P, dtype=np.float64),
        )

    with pytest.raises(ValueError):
        native.build_dmrg_normal_equations_batch(
            L_prev, T_d, T_d1, R_next, Y, r_prev, K1, r_next, 1e-6,
            np.zeros(P * P, dtype=np.float64), np.zeros(P - 1, dtype=np.float64),
        )


@pytest.mark.parametrize("r_prev,K1,r_next", [(0, 3, 2), (-1, 3, 2), (2, 0, 2), (2, 3, -4)])
def test_dmrg_normal_equations_rejects_non_positive_dims(r_prev, K1, r_next):
    """A negative int would wrap to a huge size_t once cast; reject it up front."""
    native = _native()

    N = 3
    empty = np.zeros(64, dtype=np.float64)
    with pytest.raises(ValueError):
        native.build_dmrg_normal_equations_batch(
            empty, empty, empty, empty, np.zeros(N, dtype=np.float64),
            r_prev, K1, r_next, 1e-6, empty, empty,
        )


def test_dmrg_normal_equations_accepts_a_valid_small_problem():
    """The guards must not reject a correctly sized call."""
    native = _native()

    rng = np.random.default_rng(0)
    r_prev, K1, r_next = 2, 3, 2
    P = r_prev * K1 * K1 * r_next
    N = 16

    L_prev = np.ascontiguousarray(rng.normal(size=(N, r_prev)))
    T_d = np.ascontiguousarray(rng.normal(size=(N, K1)))
    T_d1 = np.ascontiguousarray(rng.normal(size=(N, K1)))
    R_next = np.ascontiguousarray(rng.normal(size=(N, r_next)))
    Y = np.ascontiguousarray(rng.normal(size=N))

    A_out = np.zeros(P * P, dtype=np.float64)
    B_out = np.zeros(P, dtype=np.float64)
    native.build_dmrg_normal_equations_batch(
        L_prev, T_d, T_d1, R_next, Y, r_prev, K1, r_next, 1e-6, A_out, B_out
    )

    # Reference: Phi = khatri-rao(L_prev, T_d, T_d1, R_next) row-wise.
    Phi = np.einsum("np,nk,nl,nq->npklq", L_prev, T_d, T_d1, R_next).reshape(N, P)
    A_ref = Phi.T @ Phi + 1e-6 * np.eye(P)
    B_ref = Phi.T @ Y

    assert np.max(np.abs(A_out.reshape(P, P) - A_ref)) < 1e-10
    assert np.max(np.abs(B_out - B_ref)) < 1e-10


def test_project_chebyshev_rejects_undersized_output():
    native = _native()

    r_prev, K1, r_next = 3, 4, 3
    nodal = np.zeros(r_prev * K1 * r_next, dtype=np.float64)
    V_inv = np.zeros(K1 * K1, dtype=np.float64)

    with pytest.raises(ValueError):
        native.project_chebyshev_modal_batch(
            nodal, V_inv, r_prev, K1, r_next, np.zeros(r_prev * K1 * r_next - 1, dtype=np.float64)
        )


def test_tt_gradient_rejects_undersized_output():
    native = _native()

    N, D, degree = 8, 3, 2
    ranks = np.array([1, 2, 2, 1], dtype=np.int32)
    K1 = degree + 1
    sizes = [ranks[d] * K1 * ranks[d + 1] for d in range(D)]
    offsets = np.array([int(sum(sizes[:d])) for d in range(D)], dtype=np.int32)
    cores = np.zeros(int(sum(sizes)), dtype=np.float64)
    X = np.zeros((N, D), dtype=np.float64)

    with pytest.raises(ValueError):
        native.evaluate_tt_kan_gradient_batch(
            X, cores, offsets, ranks, degree, np.zeros(N * D - 1, dtype=np.float64)
        )
