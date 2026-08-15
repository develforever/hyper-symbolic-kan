r"""
JAX Functional Layers & Ergonomic Interfaces for Hyper-Symbolic KAN.

Implements:
- ContinuousKANJAXLayer: CP-KAN continuous field layer with JAX functional API and closed-form ALS.
- TensorTrainKANJAXLayer: Tensor Train KAN (TT-KAN) layer for high-dimensional scaling (D >= 10).
- Fully compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.
"""

from typing import Tuple, List, Optional, Union, Dict, Any, Sequence
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from src.jax_kan.autograd_ops import (
        cp_kan_forward,
        tt_kan_forward,
        compute_chebyshev_jax,
        compute_chebyshev_and_deriv_jax,
        _HAS_JAX
    )
except ImportError:
    _HAS_JAX = False


class ContinuousKANJAXLayer:
    r"""
    JAX Functional Layer for Continuous CP-KAN (TDFF-Net).
    
    Represents continuous mapping f: \mathbb{R}^D \to \mathbb{R}^{out\_features}
    f(x_1, ..., x_D) = \sum_{r=1}^R \lambda_r \prod_{d=1}^D \left(\sum_{k=0}^K W_{r, k}^{(d)} T_k(x_d)\right)
    
    Compatible with `jax.jit`, `jax.vmap`, `jax.grad`, and Closed-Form ALS (0 epochs).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        rank: int = 16,
        degree: int = 5,
        dtype: Any = None
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.degree = degree
        self.dtype = dtype if dtype is not None else (jnp.float64 if _HAS_JAX else np.float64)

    def init_params(self, key: Optional[Any] = None) -> Dict[str, Any]:
        """
        Initializes parameters for CP-KAN layer.
        Returns dictionary with 'lambdas' and 'factors'.
        """
        if not _HAS_JAX:
            raise ImportError("JAX is required to use ContinuousKANJAXLayer")

        if key is None:
            key = jax.random.PRNGKey(42)

        k1, k2 = jax.random.split(key)
        
        if self.out_features == 1:
            lambdas = jnp.ones((self.rank,), dtype=self.dtype)
            factors = (
                jax.random.normal(
                    k2,
                    (self.in_features, self.rank, self.degree + 1),
                    dtype=self.dtype
                ) / np.sqrt(self.degree + 1)
            )
            return {"lambdas": lambdas, "factors": factors}
        else:
            lambdas = jnp.ones((self.out_features, self.rank), dtype=self.dtype)
            factors = (
                jax.random.normal(
                    k2,
                    (self.out_features, self.in_features, self.rank, self.degree + 1),
                    dtype=self.dtype
                ) / np.sqrt(self.degree + 1)
            )
            return {"lambdas": lambdas, "factors": factors}

    def apply(self, params: Dict[str, Any], x: "jnp.ndarray") -> "jnp.ndarray":
        """
        Forward evaluation compatible with jax.jit and jax.vmap.
        x: (*batch_shape, in_features)
        returns: (*batch_shape, out_features) or (*batch_shape, 1)
        """
        orig_shape = x.shape
        in_dim = orig_shape[-1]
        if in_dim != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {in_dim}")

        x_2d = jnp.reshape(x, (-1, self.in_features))
        if x_2d.dtype != self.dtype:
            x_2d = x_2d.astype(self.dtype)

        if self.out_features == 1:
            lambdas = params["lambdas"]
            factors = params["factors"]  # (D, R, K1)
            out_2d = cp_kan_forward(x_2d, lambdas, factors)  # (N,)
            out_2d = jnp.expand_dims(out_2d, axis=-1)  # (N, 1)
        else:
            lambdas = params["lambdas"]  # (out_features, R)
            factors = params["factors"]  # (out_features, D, R, K1)
            
            outs = []
            for c in range(self.out_features):
                out_c = cp_kan_forward(x_2d, lambdas[c], factors[c])
                outs.append(out_c)
            out_2d = jnp.stack(outs, axis=-1)  # (N, out_features)

        out_shape = (*orig_shape[:-1], self.out_features)
        return jnp.reshape(out_2d, out_shape)

    def fit_als(
        self,
        params: Dict[str, Any],
        X: Union["jnp.ndarray", np.ndarray],
        Y: Union["jnp.ndarray", np.ndarray],
        alpha: float = 1e-4,
        max_als_iters: int = 10
    ) -> Tuple[Dict[str, Any], float]:
        r"""
        Instantaneous closed-form Alternating Least Squares (ALS) optimization in 0 epochs.
        X: (N, in_features)
        Y: (N, out_features) or (N,)
        returns: (updated_params, final_rmse)
        """
        X_np = np.asarray(X, dtype=np.float64)
        Y_np = np.asarray(Y, dtype=np.float64)
        if Y_np.ndim == 1:
            Y_np = Y_np[:, None]

        N, D = X_np.shape
        K1 = self.degree + 1
        R = self.rank

        # Compute Chebyshev polynomial matrices T_d: (N, K1)
        T_list = []
        for d in range(D):
            x_d = np.clip(X_np[:, d], -1.0, 1.0)
            T_d = np.empty((N, K1), dtype=np.float64)
            T_d[:, 0] = 1.0
            if self.degree >= 1:
                T_d[:, 1] = x_d
            for k in range(2, K1):
                T_d[:, k] = 2.0 * x_d * T_d[:, k - 1] - T_d[:, k - 2]
            T_list.append(T_d)

        new_factors_all = []
        new_lambdas_all = []

        for out_idx in range(self.out_features):
            target_y = Y_np[:, out_idx]
            
            # Extract initial factor matrices for this output channel
            if self.out_features == 1:
                curr_factors = [np.array(params["factors"][d], dtype=np.float64, copy=True) for d in range(D)]
                curr_lambdas = np.array(params["lambdas"], dtype=np.float64, copy=True)
            else:
                curr_factors = [np.array(params["factors"][out_idx, d], dtype=np.float64, copy=True) for d in range(D)]
                curr_lambdas = np.array(params["lambdas"][out_idx], dtype=np.float64, copy=True)

            for _ in range(max_als_iters):
                # 1. Optymalizacja macierzy czynnikowych W^(d) dla każdego wymiaru d
                for d in range(D):
                    phi_other = np.ones((N, R), dtype=np.float64)
                    for j in range(D):
                        if j != d:
                            phi_j = T_list[j] @ curr_factors[j].T  # (N, R)
                            phi_other *= phi_j

                    Phi_d = np.zeros((N, R * K1), dtype=np.float64)
                    for r in range(R):
                        scale_r = curr_lambdas[r] * phi_other[:, r]
                        start_col = r * K1
                        end_col = (r + 1) * K1
                        Phi_d[:, start_col:end_col] = T_list[d] * scale_r[:, np.newaxis]

                    A_mat = Phi_d.T @ Phi_d + alpha * np.eye(R * K1)
                    B_vec = Phi_d.T @ target_y
                    w_flat = np.linalg.solve(A_mat, B_vec)

                    updated_factors = w_flat.reshape(R, K1)
                    norms = np.linalg.norm(updated_factors, axis=1, keepdims=True) + 1e-12
                    curr_factors[d] = updated_factors / norms
                    curr_lambdas = curr_lambdas * norms.ravel()

                # 2. Bezpośrednia optymalizacja wektora wag głównych \lambda
                P = np.ones((N, R), dtype=np.float64)
                for d in range(D):
                    P *= (T_list[d] @ curr_factors[d].T)
                A_lam = P.T @ P + alpha * np.eye(R)
                B_lam = P.T @ target_y
                curr_lambdas = np.linalg.solve(A_lam, B_lam)

            new_factors_all.append(np.stack(curr_factors, axis=0))
            new_lambdas_all.append(curr_lambdas)

        if self.out_features == 1:
            updated_params = {
                "lambdas": jnp.asarray(new_lambdas_all[0], dtype=self.dtype),
                "factors": jnp.asarray(new_factors_all[0], dtype=self.dtype)
            }
        else:
            updated_params = {
                "lambdas": jnp.asarray(np.stack(new_lambdas_all, axis=0), dtype=self.dtype),
                "factors": jnp.asarray(np.stack(new_factors_all, axis=0), dtype=self.dtype)
            }

        # Calculate RMSE
        pred = self.apply(updated_params, jnp.asarray(X_np, dtype=self.dtype))
        rmse = float(jnp.sqrt(jnp.mean((pred - jnp.asarray(Y_np, dtype=self.dtype)) ** 2)))
        return updated_params, rmse


class TensorTrainKANJAXLayer:
    r"""
    JAX Functional Layer for Tensor Train KAN (TT-KAN).
    
    Represents high-dimensional mapping f: \mathbb{R}^D \to \mathbb{R} (D >= 10)
    f(x) = G^(0)(x_1) G^(1)(x_2) ... G^(D-1)(x_D)
    
    Memory footprint O(D * R^2 * K) instead of O(R^D).
    """
    def __init__(
        self,
        in_features: int,
        ranks: Optional[Sequence[int]] = None,
        degree: int = 5,
        dtype: Any = None
    ):
        self.in_features = in_features
        self.degree = degree
        self.dtype = dtype if dtype is not None else (jnp.float64 if _HAS_JAX else np.float64)

        if ranks is None:
            R = 8
            self.ranks = [1] + [R] * (in_features - 1) + [1]
        else:
            assert len(ranks) == in_features + 1 and ranks[0] == 1 and ranks[-1] == 1
            self.ranks = list(ranks)

    def init_params(self, key: Optional[Any] = None) -> Dict[str, Any]:
        """Initializes TT cores for JAX."""
        if not _HAS_JAX:
            raise ImportError("JAX is required to use TensorTrainKANJAXLayer")

        if key is None:
            key = jax.random.PRNGKey(42)

        cores = []
        keys = jax.random.split(key, self.in_features)
        K1 = self.degree + 1
        for d in range(self.in_features):
            r_prev = self.ranks[d]
            r_next = self.ranks[d + 1]
            scale = 1.0 / np.sqrt(r_prev * r_next * K1)
            core = jax.random.normal(keys[d], (r_prev, K1, r_next), dtype=self.dtype) * scale
            cores.append(core)

        return {"cores": cores}

    def apply(self, params: Dict[str, Any], x: "jnp.ndarray") -> "jnp.ndarray":
        """
        Forward evaluation compatible with jax.jit and jax.vmap.
        x: (*batch_shape, in_features)
        returns: (*batch_shape,)
        """
        orig_shape = x.shape
        in_dim = orig_shape[-1]
        if in_dim != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {in_dim}")

        x_2d = jnp.reshape(x, (-1, self.in_features))
        if x_2d.dtype != self.dtype:
            x_2d = x_2d.astype(self.dtype)

        cores = params["cores"]
        out_flat = tt_kan_forward(x_2d, *cores)
        return jnp.reshape(out_flat, orig_shape[:-1])
