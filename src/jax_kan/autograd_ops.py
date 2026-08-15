r"""
JAX Backend & Custom VJP Autograd Operators for Hyper-Symbolic KAN.

Provides:
- Exact Chebyshev polynomial and derivative evaluation in JAX (`jax.numpy`).
- Analytical VJP rules for CP-KAN (Continuous Polyadic KAN / TDFF-Net) via `jax.custom_vjp`.
- Analytical VJP rules for TT-KAN (Tensor Train KAN) via `jax.custom_vjp`.
- Zero graph expansion memory overhead during backpropagation.
"""

from typing import Tuple, List, Sequence
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None
    jnp = None
    custom_vjp = None


def check_jax_available():
    if not _HAS_JAX:
        raise ImportError(
            "JAX is not installed. To use the JAX backend, install it via `pip install jax jaxlib`."
        )


def compute_chebyshev_jax(x_d: "jnp.ndarray", degree: int) -> "jnp.ndarray":
    """
    Computes Chebyshev polynomials T_0(x) ... T_K(x) for a 1D tensor in JAX.
    x_d: (N,)
    returns: (N, degree + 1)
    """
    x_clamped = jnp.clip(x_d, -1.0, 1.0)
    T = [jnp.ones_like(x_clamped)]
    if degree >= 1:
        T.append(x_clamped)
    for k in range(1, degree):
        T.append(2.0 * x_clamped * T[k] - T[k - 1])
    return jnp.stack(T, axis=-1)


def compute_chebyshev_and_deriv_jax(
    x_d: "jnp.ndarray", degree: int
) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
    """
    Computes Chebyshev polynomials T_k(x) and analytical derivatives dT_k/dx in JAX.
    x_d: (N,)
    returns: (T, dT) each of shape (N, degree + 1)
    """
    x_clamped = jnp.clip(x_d, -1.0, 1.0)
    T = [jnp.ones_like(x_clamped)]
    dT = [jnp.zeros_like(x_clamped)]

    if degree >= 1:
        T.append(x_clamped)
        dT.append(jnp.ones_like(x_clamped))

    for k in range(1, degree):
        T.append(2.0 * x_clamped * T[k] - T[k - 1])
        dT.append(2.0 * T[k] + 2.0 * x_clamped * dT[k] - dT[k - 1])

    return jnp.stack(T, axis=-1), jnp.stack(dT, axis=-1)


if _HAS_JAX:
    # -------------------------------------------------------------------------
    # CP-KAN Custom VJP (TDFF-Net)
    # -------------------------------------------------------------------------
    @custom_vjp
    def cp_kan_forward(
        X: jnp.ndarray,
        lambdas: jnp.ndarray,
        factors: jnp.ndarray
    ) -> jnp.ndarray:
        r"""
        CP-KAN forward evaluation in JAX.
        X: (N, D)
        lambdas: (R,)
        factors: (D, R, K + 1)
        returns: (N,)
        """
        N, D = X.shape
        R = lambdas.shape[0]
        K1 = factors.shape[2]
        degree = K1 - 1

        cp_prod = jnp.ones((N, R), dtype=X.dtype)
        for d in range(D):
            T_d = compute_chebyshev_jax(X[:, d], degree)  # (N, K1)
            phi_d = T_d @ factors[d].T  # (N, R)
            cp_prod = cp_prod * phi_d

        return cp_prod @ lambdas

    def _cp_kan_fwd(
        X: jnp.ndarray,
        lambdas: jnp.ndarray,
        factors: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple]:
        N, D = X.shape
        R = lambdas.shape[0]
        K1 = factors.shape[2]
        degree = K1 - 1

        T_list = []
        dT_list = []
        phi_list = []

        for d in range(D):
            T_d, dT_d = compute_chebyshev_and_deriv_jax(X[:, d], degree)
            phi_d = T_d @ factors[d].T
            T_list.append(T_d)
            dT_list.append(dT_d)
            phi_list.append(phi_d)

        # Prefix and suffix products
        pref = []
        curr_pref = jnp.ones((N, R), dtype=X.dtype)
        for d in range(D):
            pref.append(curr_pref)
            curr_pref = curr_pref * phi_list[d]

        suff = [None] * D
        curr_suff = jnp.ones((N, R), dtype=X.dtype)
        for d in range(D - 1, -1, -1):
            suff[d] = curr_suff
            curr_suff = curr_suff * phi_list[d]

        total_P = curr_pref  # (N, R)
        Y = total_P @ lambdas  # (N,)

        ctx = (X, lambdas, factors, T_list, dT_list, phi_list, pref, suff, total_P)
        return Y, ctx

    def _cp_kan_bwd(ctx, g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        X, lambdas, factors, T_list, dT_list, phi_list, pref, suff, total_P = ctx
        N, D = X.shape
        R = lambdas.shape[0]

        # 1. Gradient w.r.t. X: (N, D)
        grad_X_cols = []
        for d in range(D):
            Q_d = pref[d] * suff[d]  # (N, R)
            dphi_d = dT_list[d] @ factors[d].T  # (N, R)
            df_d = (dphi_d * Q_d) @ lambdas  # (N,)
            grad_X_cols.append(g * df_d)
        grad_X = jnp.stack(grad_X_cols, axis=-1)

        # 2. Gradient w.r.t. lambdas: (R,)
        grad_lambdas = total_P.T @ g

        # 3. Gradient w.r.t. factors: (D, R, K1)
        grad_factors_list = []
        for d in range(D):
            Q_d = pref[d] * suff[d]  # (N, R)
            # H_d = g[:, None] * lambdas[None, :] * Q_d -> (N, R)
            H_d = g[:, None] * lambdas[None, :] * Q_d
            # (R, N) @ (N, K1) -> (R, K1)
            g_fact_d = H_d.T @ T_list[d]
            grad_factors_list.append(g_fact_d)
        grad_factors = jnp.stack(grad_factors_list, axis=0)

        return grad_X, grad_lambdas, grad_factors

    cp_kan_forward.defvjp(_cp_kan_fwd, _cp_kan_bwd)


    # -------------------------------------------------------------------------
    # TT-KAN Custom VJP (Tensor Train KAN)
    # -------------------------------------------------------------------------
    @custom_vjp
    def tt_kan_forward(
        X: jnp.ndarray,
        *cores: jnp.ndarray
    ) -> jnp.ndarray:
        r"""
        TT-KAN forward evaluation in JAX.
        X: (N, D)
        cores: tuple of D tensors G^(d) with shape (r_{d-1}, K + 1, r_d)
        returns: (N,)
        """
        N, D = X.shape
        K1 = cores[0].shape[1]
        degree = K1 - 1

        curr = jnp.ones((N, 1), dtype=X.dtype)
        for d in range(D):
            T_d = compute_chebyshev_jax(X[:, d], degree)  # (N, K1)
            # M_d(n, r_prev, r_next) = sum_k T_d(n, k) * G^(d)(r_prev, k, r_next)
            # einsum: nk, rks -> nrs
            M_d = jnp.einsum('nk, rks -> nrs', T_d, cores[d])
            # curr: (N, 1, r_prev) @ (N, r_prev, r_next) -> (N, r_next)
            curr = jnp.squeeze(jnp.matmul(jnp.expand_dims(curr, 1), M_d), axis=1)

        return jnp.squeeze(curr, axis=-1)

    def _tt_kan_fwd(
        X: jnp.ndarray,
        *cores: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple]:
        N, D = X.shape
        K1 = cores[0].shape[1]
        degree = K1 - 1

        T_list = []
        dT_list = []
        M_list = []
        dM_list = []

        for d in range(D):
            T_d, dT_d = compute_chebyshev_and_deriv_jax(X[:, d], degree)
            M_d = jnp.einsum('nk, rks -> nrs', T_d, cores[d])
            dM_d = jnp.einsum('nk, rks -> nrs', dT_d, cores[d])
            T_list.append(T_d)
            dT_list.append(dT_d)
            M_list.append(M_d)
            dM_list.append(dM_d)

        # Left prefixes L[d]: shape (N, ranks[d])
        L = [jnp.ones((N, 1), dtype=X.dtype)]
        for d in range(D):
            L_next = jnp.squeeze(jnp.matmul(jnp.expand_dims(L[d], 1), M_list[d]), axis=1)
            L.append(L_next)

        # Right suffixes R[d]: shape (N, ranks[d])
        R = [None] * (D + 1)
        R[D] = jnp.ones((N, 1), dtype=X.dtype)
        for d in range(D - 1, -1, -1):
            R_prev = jnp.squeeze(jnp.matmul(M_list[d], jnp.expand_dims(R[d + 1], -1)), axis=-1)
            R[d] = R_prev

        Y = jnp.squeeze(L[D], axis=-1)
        ctx = (X, cores, T_list, dM_list, L, R)
        return Y, ctx

    def _tt_kan_bwd(ctx, g: jnp.ndarray) -> Tuple:
        X, cores, T_list, dM_list, L, R = ctx
        N, D = X.shape

        # 1. Gradient w.r.t. X: (N, D)
        grad_X_cols = []
        for d in range(D):
            # mid: (N, 1, r_prev) @ (N, r_prev, r_next) -> (N, r_next)
            mid = jnp.squeeze(jnp.matmul(jnp.expand_dims(L[d], 1), dM_list[d]), axis=1)
            # sum_s mid(n, s) * R[d+1](n, s)
            df_d = jnp.sum(mid * R[d + 1], axis=1)
            grad_X_cols.append(g * df_d)
        grad_X = jnp.stack(grad_X_cols, axis=-1)

        # 2. Gradient w.r.t. cores: tuple of D (r_{d-1}, K1, r_d)
        grad_cores = []
        for d in range(D):
            L_weighted = g[:, None] * L[d]  # (N, r_prev)
            # einsum: nr, nk, ns -> rks
            grad_c = jnp.einsum('nr, nk, ns -> rks', L_weighted, T_list[d], R[d + 1])
            grad_cores.append(grad_c)

        return (grad_X, *grad_cores)

    tt_kan_forward.defvjp(_tt_kan_fwd, _tt_kan_bwd)

else:
    # Dummy fallbacks when JAX is not installed
    def cp_kan_forward(*args, **kwargs):
        check_jax_available()

    def tt_kan_forward(*args, **kwargs):
        check_jax_available()
