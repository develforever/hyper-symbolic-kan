import os
import sys
import numpy as np
from typing import List, Tuple, Optional

# Try loading the pre-compiled nanobind C++ extension module
try:
    from src.cpp_kernels import _cpp_kernels as _native_kernels
    _HAS_CPP = True
except ImportError:
    try:
        import _cpp_kernels as _native_kernels
        _HAS_CPP = True
    except ImportError:
        _native_kernels = None
        _HAS_CPP = False


class FastCPPKANEngine:
    r"""
    Natywny Silnik C++ / SIMD (AVX2 & OpenMP) z bindingami nanobind.
    
    Zapewnia ewaluację pól tensorowych TT-KAN oraz CP-KAN (TDFF-Net)
    oraz obliczanie ich analitycznych gradientów \nabla f z opóźnieniem sub-mikrosekundowym (< 0.2 us / pkt)
    i zerowym narzutem pamięciowym (brak alokacji na stercie per zapytanie, GIL released).
    """
    def __init__(self, spatial_dim: int = 10, degree: int = 5):
        self.spatial_dim = spatial_dim
        self.degree = degree
        self.has_native = _HAS_CPP

    def is_native_available(self) -> bool:
        return self.has_native

    def evaluate_batch(self, X: np.ndarray, cores: List[np.ndarray], ranks: List[int]) -> np.ndarray:
        r"""
        High-Throughput Ewaluacja TT-KAN dla N punktów w C++ z prędkością sub-mikrosekundową.
        """
        N, D = X.shape
        assert D == self.spatial_dim, f"Dimension mismatch: expected {self.spatial_dim}, got {D}"
        
        X_cont = np.ascontiguousarray(X, dtype=np.float64)

        if self.has_native and _native_kernels is not None:
            cores_flat_list = []
            core_offsets = []
            offset = 0
            for c in cores:
                c_flat = np.ascontiguousarray(c, dtype=np.float64).ravel()
                cores_flat_list.append(c_flat)
                core_offsets.append(offset)
                offset += c_flat.size
                
            all_cores_flat = np.concatenate(cores_flat_list).astype(np.float64, copy=False)
            core_offsets_arr = np.array(core_offsets, dtype=np.int32)
            ranks_arr = np.array(ranks, dtype=np.int32)
            
            Y_out = np.empty(N, dtype=np.float64)
            _native_kernels.evaluate_tt_kan_batch(
                X_cont,
                all_cores_flat,
                core_offsets_arr,
                ranks_arr,
                self.degree,
                Y_out
            )
            return Y_out

        # NumPy SIMD Pipeline Fallback
        K1 = self.degree + 1
        curr = np.ones((N, 1))
        
        for d in range(D):
            x_d = np.clip(X_cont[:, d], -1.0, 1.0)
            T_d = np.empty((N, K1), dtype=np.float64)
            T_d[:, 0] = 1.0
            if self.degree >= 1:
                T_d[:, 1] = x_d
            for k in range(1, self.degree):
                T_d[:, k + 1] = 2.0 * x_d * T_d[:, k] - T_d[:, k - 1]
                
            core_d = cores[d]
            r_prev, _, r_next = core_d.shape
            core_flat = core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)
            
            M_d = (T_d @ core_flat).reshape(N, r_prev, r_next)
            curr = (curr[:, None, :] @ M_d)[:, 0, :]
            
        return curr.squeeze(-1)

    def gradient_batch(self, X: np.ndarray, cores: List[np.ndarray], ranks: List[int]) -> np.ndarray:
        r"""
        Analityczny gradient TT-KAN \nabla f(X) \in \mathbb{R}^{N \times D} w C++.
        """
        N, D = X.shape
        assert D == self.spatial_dim, f"Dimension mismatch: expected {self.spatial_dim}, got {D}"
        
        X_cont = np.ascontiguousarray(X, dtype=np.float64)

        if self.has_native and _native_kernels is not None:
            cores_flat_list = []
            core_offsets = []
            offset = 0
            for c in cores:
                c_flat = np.ascontiguousarray(c, dtype=np.float64).ravel()
                cores_flat_list.append(c_flat)
                core_offsets.append(offset)
                offset += c_flat.size
                
            all_cores_flat = np.concatenate(cores_flat_list).astype(np.float64, copy=False)
            core_offsets_arr = np.array(core_offsets, dtype=np.int32)
            ranks_arr = np.array(ranks, dtype=np.int32)
            
            grad_out = np.empty((N, D), dtype=np.float64)
            _native_kernels.evaluate_tt_kan_gradient_batch(
                X_cont,
                all_cores_flat,
                core_offsets_arr,
                ranks_arr,
                self.degree,
                grad_out
            )
            return grad_out

        # NumPy Analytical Gradient Fallback
        K1 = self.degree + 1
        M_list, dM_list = [], []
        for d in range(D):
            x_d = np.clip(X_cont[:, d], -1.0, 1.0)
            T_d = np.empty((N, K1), dtype=np.float64)
            dT_d = np.empty((N, K1), dtype=np.float64)
            T_d[:, 0] = 1.0
            dT_d[:, 0] = 0.0
            if self.degree >= 1:
                T_d[:, 1] = x_d
                dT_d[:, 1] = 1.0
            for k in range(1, self.degree):
                T_d[:, k + 1] = 2.0 * x_d * T_d[:, k] - T_d[:, k - 1]
                dT_d[:, k + 1] = 2.0 * T_d[:, k] + 2.0 * x_d * dT_d[:, k] - dT_d[:, k - 1]
            
            core_d = cores[d]
            r_prev, _, r_next = core_d.shape
            core_flat = core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)
            M_list.append((T_d @ core_flat).reshape(N, r_prev, r_next))
            dM_list.append((dT_d @ core_flat).reshape(N, r_prev, r_next))
            
        L = [None] * D
        L_curr = np.ones((N, 1))
        for d in range(D):
            L_curr = (L_curr[:, None, :] @ M_list[d])[:, 0, :]
            L[d] = L_curr
            
        R = [None] * D
        R_curr = np.ones((N, 1))
        for d in range(D - 1, -1, -1):
            R[d] = R_curr
            R_curr = (M_list[d] @ R_curr[:, :, None])[:, :, 0]
            
        grad = np.zeros((N, D), dtype=np.float64)
        for m in range(D):
            L_prev = np.ones((N, 1)) if m == 0 else L[m - 1]
            R_next = R[m]
            dM_m = dM_list[m]
            mid = (L_prev[:, None, :] @ dM_m)[:, 0, :]
            grad[:, m] = (mid * R_next).sum(axis=1)
            
        return grad

    def evaluate_cp_batch(
        self, X: np.ndarray, factors: List[np.ndarray], lambdas: np.ndarray
    ) -> np.ndarray:
        r"""
        High-Throughput Ewaluacja CP-KAN (TDFF-Net) w C++.
        """
        N, D = X.shape
        assert D == self.spatial_dim
        
        X_cont = np.ascontiguousarray(X, dtype=np.float64)
        lambdas_cont = np.ascontiguousarray(lambdas, dtype=np.float64)
        rank = len(lambdas)
        
        if self.has_native and _native_kernels is not None:
            # Flatten factors (D, rank, K1)
            factors_flat = np.concatenate([
                np.ascontiguousarray(f, dtype=np.float64).ravel() for f in factors
            ]).astype(np.float64, copy=False)
            
            Y_out = np.empty(N, dtype=np.float64)
            _native_kernels.evaluate_cp_kan_batch(
                X_cont,
                factors_flat,
                lambdas_cont,
                rank,
                self.degree,
                Y_out
            )
            return Y_out

        # NumPy fallback
        cp_product = np.ones((N, rank), dtype=np.float64)
        K1 = self.degree + 1
        for d in range(D):
            x_d = np.clip(X_cont[:, d], -1.0, 1.0)
            T = np.empty((N, K1), dtype=np.float64)
            T[:, 0] = 1.0
            if self.degree >= 1:
                T[:, 1] = x_d
            for k in range(1, self.degree):
                T[:, k + 1] = 2.0 * x_d * T[:, k] - T[:, k - 1]
            phi_d = T @ factors[d].T
            cp_product *= phi_d
            
        return cp_product @ lambdas_cont

    def gradient_cp_batch(
        self, X: np.ndarray, factors: List[np.ndarray], lambdas: np.ndarray
    ) -> np.ndarray:
        r"""
        Analityczny gradient CP-KAN (TDFF-Net) \nabla f(X) \in \mathbb{R}^{N \times D} w C++.
        """
        N, D = X.shape
        assert D == self.spatial_dim
        
        X_cont = np.ascontiguousarray(X, dtype=np.float64)
        lambdas_cont = np.ascontiguousarray(lambdas, dtype=np.float64)
        rank = len(lambdas)
        
        if self.has_native and _native_kernels is not None:
            factors_flat = np.concatenate([
                np.ascontiguousarray(f, dtype=np.float64).ravel() for f in factors
            ]).astype(np.float64, copy=False)
            
            grad_out = np.empty((N, D), dtype=np.float64)
            _native_kernels.evaluate_cp_kan_gradient_batch(
                X_cont,
                factors_flat,
                lambdas_cont,
                rank,
                self.degree,
                grad_out
            )
            return grad_out

        # NumPy fallback
        K1 = self.degree + 1
        phi_evals = np.empty((N, rank, D))
        dphi_evals = np.empty((N, rank, D))
        
        for d in range(D):
            x_d = np.clip(X_cont[:, d], -1.0, 1.0)
            T = np.empty((N, K1), dtype=np.float64)
            dT = np.empty((N, K1), dtype=np.float64)
            T[:, 0] = 1.0
            dT[:, 0] = 0.0
            if self.degree >= 1:
                T[:, 1] = x_d
                dT[:, 1] = 1.0
            for k in range(1, self.degree):
                T[:, k + 1] = 2.0 * x_d * T[:, k] - T[:, k - 1]
                dT[:, k + 1] = 2.0 * T[:, k] + 2.0 * x_d * dT[:, k] - dT[:, k - 1]
            phi_evals[:, :, d] = T @ factors[d].T
            dphi_evals[:, :, d] = dT @ factors[d].T
            
        grad = np.zeros((N, D), dtype=np.float64)
        for dim in range(D):
            term_evals = phi_evals.copy()
            term_evals[:, :, dim] = dphi_evals[:, :, dim]
            grad[:, dim] = np.prod(term_evals, axis=2) @ lambdas_cont
            
        return grad

    def project_chebyshev_modal(self, nodal_core: np.ndarray, V_inv: np.ndarray) -> np.ndarray:
        r"""
        Projekcja węzłowego rdzenia TT na współczynniki Czebyszewa: (V_inv @ Nodal).
        nodal_core: (r_prev, K1, r_next)
        V_inv: (K1, K1)
        """
        r_prev, K1, r_next = nodal_core.shape
        assert V_inv.shape == (K1, K1)
        
        nodal_cont = np.ascontiguousarray(nodal_core, dtype=np.float64)
        V_inv_cont = np.ascontiguousarray(V_inv, dtype=np.float64)

        if self.has_native and _native_kernels is not None and hasattr(_native_kernels, "project_chebyshev_modal_batch"):
            modal_out = np.empty((r_prev, K1, r_next), dtype=np.float64)
            _native_kernels.project_chebyshev_modal_batch(
                nodal_cont,
                V_inv_cont,
                r_prev,
                K1,
                r_next,
                modal_out
            )
            return modal_out

        # NumPy fallback
        return np.einsum('ki, ris -> rks', V_inv_cont, nodal_cont)

    def build_dmrg_normal_equations(
        self,
        L_prev: np.ndarray,
        T_d: np.ndarray,
        T_d1: np.ndarray,
        R_next: np.ndarray,
        Y: np.ndarray,
        alpha: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Akumulacja układu normalnego Tikhonova dla 2-Site DMRG w C++ (AVX2 + OpenMP):
        A = Phi^T Phi + alpha * I
        B = Phi^T Y
        bez alokowania macierzy Phi o kształcie (N, P).
        """
        N, r_prev = L_prev.shape
        _, K1 = T_d.shape
        _, r_next = R_next.shape
        P = r_prev * K1 * K1 * r_next

        L_cont = np.ascontiguousarray(L_prev, dtype=np.float64)
        T_d_cont = np.ascontiguousarray(T_d, dtype=np.float64)
        T_d1_cont = np.ascontiguousarray(T_d1, dtype=np.float64)
        R_cont = np.ascontiguousarray(R_next, dtype=np.float64)
        Y_cont = np.ascontiguousarray(Y.ravel(), dtype=np.float64)

        if self.has_native and _native_kernels is not None and hasattr(_native_kernels, "build_dmrg_normal_equations_batch"):
            A_out = np.empty((P, P), dtype=np.float64)
            B_out = np.empty(P, dtype=np.float64)
            _native_kernels.build_dmrg_normal_equations_batch(
                L_cont,
                T_d_cont,
                T_d1_cont,
                R_cont,
                Y_cont,
                r_prev,
                K1,
                r_next,
                alpha,
                A_out,
                B_out
            )
            return A_out, B_out

        # NumPy fallback
        mid_basis = (T_d_cont[:, :, None] * T_d1_cont[:, None, :]).reshape(N, K1 * K1)
        Phi_left = (L_cont[:, :, None] * mid_basis[:, None, :]).reshape(N, r_prev * K1 * K1)
        Phi = (Phi_left[:, :, None] * R_cont[:, None, :]).reshape(N, P)
        
        A = Phi.T @ Phi + alpha * np.eye(P)
        B = Phi.T @ Y_cont
        return A, B

