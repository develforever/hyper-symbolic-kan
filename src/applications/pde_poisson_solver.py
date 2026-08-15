r"""
Mesh-Free Physics-Informed KAN Poisson / Laplace PDE Solver (0 Gradient Epochs).

Implements:
1. Analytical 2nd-order Chebyshev derivative recurrence d^2 T_k / dx^2.
2. Tensor Product & Spectral KAN Laplace Operator \nabla^2 \Psi(x) in arbitrary dimensions D.
3. Closed-Form Algebraic Collocation Solver for Poisson's equation \nabla^2 u = f with Dirichlet boundary conditions in 0 epochs.
4. Higher-Dimensional CP / Tensor-Train KAN Poisson ALS Solver (D >= 4).
5. Exact analytical verification benchmarks (Polynomial, Trigonometric, Multi-D).
"""

import time
import numpy as np
from typing import Tuple, Callable, Optional, Dict, Any, List
from itertools import product


def chebyshev_derivatives_2nd(x: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Wyznacza analityczne wielomiany Czebyszewa T_k(x), 1. pochodne T'_k(x) oraz 2. pochodne T''_k(x)
    dla stopni k = 0, 1, ..., degree przy użyciu 3-elementowej ścisłej rekurencji:
    
    T_0 = 1, T'_0 = 0, T''_0 = 0
    T_1 = x, T'_1 = 1, T''_1 = 0
    T_{k+1} = 2x T_k - T_{k-1}
    T'_{k+1} = 2 T_k + 2x T'_k - T'_{k-1}
    T''_{k+1} = 4 T'_k + 2x T''_k - T''_{k-1}
    """
    x_clamped = np.clip(x, -1.0, 1.0)
    N = len(x_clamped)
    K = degree
    
    T = np.zeros((N, K + 1), dtype=np.float64)
    dT = np.zeros((N, K + 1), dtype=np.float64)
    d2T = np.zeros((N, K + 1), dtype=np.float64)
    
    T[:, 0] = 1.0
    dT[:, 0] = 0.0
    d2T[:, 0] = 0.0
    
    if K >= 1:
        T[:, 1] = x_clamped
        dT[:, 1] = 1.0
        d2T[:, 1] = 0.0
        
    for k in range(1, K):
        T[:, k + 1] = 2.0 * x_clamped * T[:, k] - T[:, k - 1]
        dT[:, k + 1] = 2.0 * T[:, k] + 2.0 * x_clamped * dT[:, k] - dT[:, k - 1]
        d2T[:, k + 1] = 4.0 * dT[:, k] + 2.0 * x_clamped * d2T[:, k] - d2T[:, k - 1]
        
    return T, dT, d2T


class PoissonAnalyticalSolution:
    """Zestaw ścisłych rozwiązań analitycznych równania Poissona dla celów benchmarkingu."""
    
    @staticmethod
    def get_2d_polynomial():
        """u*(x, y) = (1 - x^2)(1 - y^2), \nabla^2 u = 2(x^2 + y^2 - 2), u=0 na brzegu."""
        def u_exact(X: np.ndarray) -> np.ndarray:
            x, y = X[:, 0], X[:, 1]
            return (1.0 - x**2) * (1.0 - y**2)
            
        def f_rhs(X: np.ndarray) -> np.ndarray:
            x, y = X[:, 0], X[:, 1]
            return 2.0 * (x**2 + y**2 - 2.0)
            
        def g_bc(X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X))
            
        return u_exact, f_rhs, g_bc

    @staticmethod
    def get_2d_trigonometric():
        """u*(x, y) = sin(pi*x) * sin(pi*y), \nabla^2 u = -2*pi^2 * u*, u=0 na brzegu [-1, 1]^2."""
        def u_exact(X: np.ndarray) -> np.ndarray:
            x, y = X[:, 0], X[:, 1]
            return np.sin(np.pi * x) * np.sin(np.pi * y)
            
        def f_rhs(X: np.ndarray) -> np.ndarray:
            return -2.0 * (np.pi ** 2) * u_exact(X)
            
        def g_bc(X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X))
            
        return u_exact, f_rhs, g_bc

    @staticmethod
    def get_3d_polynomial():
        """u*(x, y, z) = (1-x^2)(1-y^2)(1-z^2), u=0 na brzegu [-1, 1]^3."""
        def u_exact(X: np.ndarray) -> np.ndarray:
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            return (1.0 - x**2) * (1.0 - y**2) * (1.0 - z**2)
            
        def f_rhs(X: np.ndarray) -> np.ndarray:
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            term_x = -2.0 * (1.0 - y**2) * (1.0 - z**2)
            term_y = -2.0 * (1.0 - x**2) * (1.0 - z**2)
            term_z = -2.0 * (1.0 - x**2) * (1.0 - y**2)
            return term_x + term_y + term_z
            
        def g_bc(X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X))
            
        return u_exact, f_rhs, g_bc

    @staticmethod
    def get_3d_trigonometric():
        """u*(x, y, z) = cos(pi/2 * x) * cos(pi/2 * y) * cos(pi/2 * z), u=0 na brzegu."""
        def u_exact(X: np.ndarray) -> np.ndarray:
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            return np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) * np.cos(0.5 * np.pi * z)
            
        def f_rhs(X: np.ndarray) -> np.ndarray:
            return -3.0 * ((0.5 * np.pi) ** 2) * u_exact(X)
            
        def g_bc(X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X))
            
        return u_exact, f_rhs, g_bc


class SpectralKANPoissonSolver:
    r"""
    Bezsiatkowy Solver Równania Poissona \nabla^2 u(x) = f(x) w bazie Spektralnej KAN Czebyszewa.
    
    Rozwiązuje zadanie brzegowe Dirichleta w 0 epokach gradientowych metodą rzutu kolokacyjnego:
    u(x) = \sum_{k_1, ..., k_D} C_{k_1, ..., k_D} \prod_{d=1}^D T_{k_d}(x_d)
    
    \nabla^2 u(x) = \sum_{k_1, ..., k_D} C_{k} \left( \sum_{d=1}^D T''_{k_d}(x_d) \prod_{m \neq d} T_{k_m}(x_m) \right)
    """
    def __init__(self, spatial_dim: int = 2, degree: int = 6):
        self.spatial_dim = spatial_dim
        self.degree = degree
        self.num_basis_1d = degree + 1
        self.total_basis = self.num_basis_1d ** spatial_dim
        
        # Generacja indeksów wielowymiarowych (k_1, k_2, ..., k_D)
        self.multi_indices = np.array(list(product(range(self.num_basis_1d), repeat=spatial_dim)))
        self.coefficients = np.zeros(self.total_basis, dtype=np.float64)

    def _sample_domain_points(self, n_interior: int = 800, n_boundary: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Generuje punkty kolokacji wewnętrznej oraz punkty na brzegach domeny [-1, 1]^D."""
        # Wewnętrzne punkty losowe w (-0.98, 0.98)^D
        X_int = np.random.uniform(-0.96, 0.96, size=(n_interior, self.spatial_dim))
        
        # Punkty na 2*D ścianach brzegowych
        pts_per_face = max(4, n_boundary // (2 * self.spatial_dim))
        X_bc_list = []
        
        for d in range(self.spatial_dim):
            for val in [-1.0, 1.0]:
                face_pts = np.random.uniform(-1.0, 1.0, size=(pts_per_face, self.spatial_dim))
                face_pts[:, d] = val
                X_bc_list.append(face_pts)
                
        X_bc = np.vstack(X_bc_list)
        return X_int, X_bc

    def _evaluate_basis_and_laplacian(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca macierze ewaluacji bazy Phi(X) o kształcie (N, M) oraz
        laplasjanu bazy LapPhi(X) o kształcie (N, M).
        """
        N, D = X.shape
        M = self.total_basis
        
        # Obliczenie T, dT, d2T dla każdego wymiaru
        T_dims = []
        d2T_dims = []
        for d in range(D):
            T_d, _, d2T_d = chebyshev_derivatives_2nd(X[:, d], self.degree)
            T_dims.append(T_d)
            d2T_dims.append(d2T_d)
            
        Phi = np.ones((N, M), dtype=np.float64)
        LapPhi = np.zeros((N, M), dtype=np.float64)
        
        for idx, multi_idx in enumerate(self.multi_indices):
            # Ewaluacja iloczynu \prod_{d=1}^D T_{k_d}(x_d)
            prod_T = np.ones(N)
            for d in range(D):
                k_d = multi_idx[d]
                prod_T *= T_dims[d][:, k_d]
            Phi[:, idx] = prod_T
            
            # Ewaluacja laplasjanu: \sum_{d=1}^D T''_{k_d}(x_d) \prod_{m \neq d} T_{k_m}(x_m)
            lap_val = np.zeros(N)
            for d in range(D):
                term_d = d2T_dims[d][:, multi_idx[d]].copy()
                for m in range(D):
                    if m != d:
                        term_d *= T_dims[m][:, multi_idx[m]]
                lap_val += term_d
            LapPhi[:, idx] = lap_val
            
        return Phi, LapPhi

    def fit(
        self,
        f_rhs_fn: Callable[[np.ndarray], np.ndarray],
        g_bc_fn: Callable[[np.ndarray], np.ndarray],
        n_interior: int = 1200,
        n_boundary: int = 600,
        alpha_reg: float = 1e-9,
        beta_bc: float = 200.0
    ) -> Dict[str, Any]:
        """
        Rozwiązuje stacjonarne równanie Poissona w 0 epokach gradientowych
        za pomocą regularyzowanego układu normalnego Tikhonova.
        """
        t0 = time.perf_counter()
        
        X_int, X_bc = self._sample_domain_points(n_interior, n_boundary)
        
        # 1. Konstrukcja bloków macierzowych
        _, LapPhi_int = self._evaluate_basis_and_laplacian(X_int)
        Phi_bc, _ = self._evaluate_basis_and_laplacian(X_bc)
        
        f_int = f_rhs_fn(X_int)
        g_bc = g_bc_fn(X_bc)
        
        # Ważony blokowy układ równań: [LapPhi_int; sqrt(beta)*Phi_bc] C = [f_int; sqrt(beta)*g_bc]
        sqrt_beta = np.sqrt(beta_bc)
        A = np.vstack([LapPhi_int, sqrt_beta * Phi_bc])
        b = np.concatenate([f_int, sqrt_beta * g_bc])
        
        # 2. Bezpośrednie algebraiczne rozwiązanie w 0 epokach
        # A^T A C = A^T b + alpha * I
        AtA = A.T @ A + alpha_reg * np.eye(self.total_basis)
        Atb = A.T @ b
        
        try:
            self.coefficients = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(A, b, rcond=1e-12)[0]
            
        t1 = time.perf_counter()
        solve_time_ms = (t1 - t0) * 1000.0
        
        # Ewaluacja residuum PDE i błędu brzegowego
        res_pde = np.sqrt(np.mean((LapPhi_int @ self.coefficients - f_int) ** 2))
        res_bc = np.sqrt(np.mean((Phi_bc @ self.coefficients - g_bc) ** 2))
        
        return {
            "solve_time_ms": solve_time_ms,
            "pde_residual_rmse": float(res_pde),
            "bc_residual_rmse": float(res_bc),
            "basis_count": self.total_basis
        }

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Ewaluacja pola u(X) w punktach X (N, D)."""
        X_arr = np.atleast_2d(X)
        Phi, _ = self._evaluate_basis_and_laplacian(X_arr)
        u_val = Phi @ self.coefficients
        return u_val.ravel() if X.ndim == 1 else u_val

    def laplacian(self, X: np.ndarray) -> np.ndarray:
        """Ewaluacja laplasjanu \nabla^2 u(X) w punktach X (N, D)."""
        X_arr = np.atleast_2d(X)
        _, LapPhi = self._evaluate_basis_and_laplacian(X_arr)
        lap_val = LapPhi @ self.coefficients
        return lap_val.ravel() if X.ndim == 1 else lap_val

    def compute_l2_relative_error(self, X: np.ndarray, exact_u_fn: Callable[[np.ndarray], np.ndarray]) -> float:
        """Oblicza błąd względny L2 w stosunku do ścisłego rozwiązania analitycznego."""
        u_pred = self.evaluate(X)
        u_exact = exact_u_fn(X)
        l2_diff = np.sqrt(np.sum((u_pred - u_exact) ** 2))
        l2_exact = np.sqrt(np.sum(u_exact ** 2))
        return float(l2_diff / (l2_exact + 1e-12))


class TTPoissonSolver:
    r"""
    Wysokowymiarowy Tensor Train KAN Poisson Solver (D >= 4).
    
    Wykorzystuje analityczną dekompozycję operatora Laplace'a w łańcuchu TT
    oraz naprzemienne dopasowanie (ALS) bez konieczności stosowania gęstych siatek.
    """
    def __init__(self, spatial_dim: int = 4, ranks: Optional[List[int]] = None, degree: int = 5):
        self.spatial_dim = spatial_dim
        self.degree = degree
        self.num_basis = degree + 1
        
        if ranks is None:
            R = 6
            self.ranks = [1] + [R] * (spatial_dim - 1) + [1]
        else:
            self.ranks = ranks
            
        np.random.seed(42)
        self.cores = []
        for d in range(spatial_dim):
            r_prev = self.ranks[d]
            r_next = self.ranks[d + 1]
            core = np.random.randn(r_prev, self.num_basis, r_next) / np.sqrt(r_prev * r_next * self.num_basis)
            self.cores.append(core)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Ewaluacja pola u(X) w reprezentacji Tensor Train."""
        N, D = X.shape
        curr = np.ones((N, 1))
        
        for d in range(D):
            T_d, _, _ = chebyshev_derivatives_2nd(X[:, d], self.degree)
            core_d = self.cores[d]
            r_prev, K1, r_next = core_d.shape
            
            M_d = (T_d @ core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)).reshape(N, r_prev, r_next)
            curr = (curr[:, None, :] @ M_d)[:, 0, :]
            
        return curr.squeeze(-1)

    def laplacian(self, X: np.ndarray) -> np.ndarray:
        """Ewaluacja dokładnego analitycznego laplasjanu \nabla^2 u(X) w formacie TT."""
        N, D = X.shape
        
        M_list = []
        d2M_list = []
        for d in range(D):
            T_d, _, d2T_d = chebyshev_derivatives_2nd(X[:, d], self.degree)
            core_d = self.cores[d]
            r_prev, K1, r_next = core_d.shape
            core_flat = core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)
            
            M_list.append((T_d @ core_flat).reshape(N, r_prev, r_next))
            d2M_list.append((d2T_d @ core_flat).reshape(N, r_prev, r_next))
            
        # Lewe prefiksy
        L = [None] * D
        L_curr = np.ones((N, 1))
        for d in range(D):
            L_curr = (L_curr[:, None, :] @ M_list[d])[:, 0, :]
            L[d] = L_curr
            
        # Prawe prefiksy
        R = [None] * D
        R_curr = np.ones((N, 1))
        for d in range(D - 1, -1, -1):
            R[d] = R_curr
            R_curr = (M_list[d] @ R_curr[:, :, None])[:, :, 0]
            
        # Suma drugich pochodnych po wszystkich wymiarach
        lap = np.zeros(N)
        for m in range(D):
            L_prev = np.ones((N, 1)) if m == 0 else L[m - 1]
            R_next = R[m]
            d2M_m = d2M_list[m]
            
            mid = (L_prev[:, None, :] @ d2M_m)[:, 0, :]
            d2u_dxm2 = (mid * R_next).sum(axis=1)
            lap += d2u_dxm2
            
        return lap

    def fit_als(
        self,
        f_rhs_fn: Callable[[np.ndarray], np.ndarray],
        g_bc_fn: Callable[[np.ndarray], np.ndarray],
        n_interior: int = 1000,
        n_boundary: int = 500,
        max_sweeps: int = 3,
        alpha_reg: float = 1e-6,
        beta_bc: float = 100.0
    ) -> float:
        """
        Dopasowanie bezgradientowe TT-ALS pod residuum różniczkowe Poissona.
        """
        X_int = np.random.uniform(-0.95, 0.95, size=(n_interior, self.spatial_dim))
        
        # Punkty brzegowe
        X_bc_list = []
        pts_per_face = max(4, n_boundary // (2 * self.spatial_dim))
        for d in range(self.spatial_dim):
            for val in [-1.0, 1.0]:
                face_pts = np.random.uniform(-1.0, 1.0, size=(pts_per_face, self.spatial_dim))
                face_pts[:, d] = val
                X_bc_list.append(face_pts)
        X_bc = np.vstack(X_bc_list)
        
        f_int = f_rhs_fn(X_int)
        g_bc = g_bc_fn(X_bc)
        sqrt_beta = np.sqrt(beta_bc)
        
        D = self.spatial_dim
        
        for sweep in range(max_sweeps):
            for d in range(D):
                # 1. Obliczenie prefiksów L i R dla punktów wewnętrznych i brzegowych
                def get_prefixes(X_pts):
                    N_pts = len(X_pts)
                    M_pts = []
                    d2M_pts = []
                    for dim in range(D):
                        T_dim, _, d2T_dim = chebyshev_derivatives_2nd(X_pts[:, dim], self.degree)
                        core_dim = self.cores[dim]
                        r_p, K_1, r_n = core_dim.shape
                        flat = core_dim.transpose(1, 0, 2).reshape(K_1, r_p * r_n)
                        M_pts.append((T_dim @ flat).reshape(N_pts, r_p, r_n))
                        d2M_pts.append((d2T_dim @ flat).reshape(N_pts, r_p, r_n))
                        
                    L_pts = [None] * D
                    L_c = np.ones((N_pts, 1))
                    for dim in range(D):
                        L_c = (L_c[:, None, :] @ M_pts[dim])[:, 0, :]
                        L_pts[dim] = L_c
                        
                    R_pts = [None] * D
                    R_c = np.ones((N_pts, 1))
                    for dim in range(D - 1, -1, -1):
                        R_pts[dim] = R_c
                        R_c = (M_pts[dim] @ R_c[:, :, None])[:, :, 0]
                        
                    return T_dim, d2T_dim, L_pts, R_pts, M_pts, d2M_pts
                    
                _, _, L_int, R_int, _, _ = get_prefixes(X_int)
                _, _, L_bc, R_bc, _, _ = get_prefixes(X_bc)
                
                T_d_int, _, d2T_d_int = chebyshev_derivatives_2nd(X_int[:, d], self.degree)
                T_d_bc, _, _ = chebyshev_derivatives_2nd(X_bc[:, d], self.degree)
                
                L_prev_int = np.ones((len(X_int), 1)) if d == 0 else L_int[d - 1]
                R_next_int = R_int[d]
                
                L_prev_bc = np.ones((len(X_bc), 1)) if d == 0 else L_bc[d - 1]
                R_next_bc = R_bc[d]
                
                r_prev = self.ranks[d]
                r_next = self.ranks[d + 1]
                K1 = self.num_basis
                
                # Konstrukcja macierzy dla u(x_bc): L_{n, r} * T_{n, k} * R_{n, s}
                Phi_bc = (L_prev_bc[:, :, None, None] * T_d_bc[:, None, :, None] * R_next_bc[:, None, None, :]).reshape(len(X_bc), r_prev * K1 * r_next)
                
                # Konstrukcja macierzy dla laplasjanu: \nabla^2 u(x_int)
                # Część z d2T_d + części ze stałych pozostałych drugich pochodnych
                LapPhi_int = (L_prev_int[:, :, None, None] * d2T_d_int[:, None, :, None] * R_next_int[:, None, None, :]).reshape(len(X_int), r_prev * K1 * r_next)
                
                # Złożenie układu normalnego
                A = np.vstack([LapPhi_int, sqrt_beta * Phi_bc])
                b = np.concatenate([f_int, sqrt_beta * g_bc])
                
                AtA = A.T @ A + alpha_reg * np.eye(r_prev * K1 * r_next)
                Atb = A.T @ b
                
                core_flat = np.linalg.solve(AtA, Atb)
                self.cores[d] = core_flat.reshape(r_prev, K1, r_next)
                
        # Obliczenie końcowego błędu residuum
        lap_pred = self.laplacian(X_int)
        rmse_res = float(np.sqrt(np.mean((lap_pred - f_int) ** 2)))
        return rmse_res
