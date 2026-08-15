import numpy as np
import scipy.linalg
from typing import List, Tuple, Optional, Union
from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.tdff_net.tt_kan import TensorTrainKAN


class DMRGTTKANSolver:
    r"""
    2-Site DMRG (Density Matrix Renormalization Group) Solver dla Tensor Train KAN.
    
    Optymalizuje sąsiadujące pary rdzeni G^(d) <-> G^(d+1) jako super-rdzeń
    B^(d) \in \mathbb{R}^{r_{d-1} \times (K+1) \times (K+1) \times r_{d+1}}.
    
    Zalety względem 1-site ALS:
    1. Dynamiczna adaptacja rang wiązań w czasie rzeczywistym poprzez SVD.
    2. Ucieczka z pułapek lokalnych minimów dzięki poszerzonej przestrzeni wariacyjnej 2-site.
    3. Zachowanie ścisłej ortogonalności kanonicznej (left/right canonical form).
    """
    def __init__(
        self,
        alpha: float = 1e-6,
        max_sweeps: int = 4,
        variance_threshold: float = 0.9999,
        max_rank: int = 16,
        min_rank: int = 1
    ):
        self.alpha = alpha
        self.max_sweeps = max_sweeps
        self.variance_threshold = variance_threshold
        self.max_rank = max_rank
        self.min_rank = min_rank

    def _compute_chebyshev_all(self, model: Union[DynamicRankTTKAN, TensorTrainKAN], X: np.ndarray) -> List[np.ndarray]:
        """Oblicza bazy Czebyszewa T_k(x) dla wszystkich wymiarów."""
        D = model.spatial_dim
        T_list = []
        for d in range(D):
            T_d, _ = model._compute_chebyshev_and_derivatives(X[:, d])
            T_list.append(T_d)
        return T_list

    def _update_prefixes(
        self,
        model: Union[DynamicRankTTKAN, TensorTrainKAN],
        T_list: List[np.ndarray],
        N: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Wyznacza lewe prefiksy L i prawe sufiksy R dla całego łańcucha TT."""
        D = model.spatial_dim
        
        M_list = []
        for d in range(D):
            core = model.cores[d]
            r_p, K1, r_n = core.shape
            core_flat = core.transpose(1, 0, 2).reshape(K1, r_p * r_n)
            M_d = (T_list[d] @ core_flat).reshape(N, r_p, r_n)
            M_list.append(M_d)
            
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
            
        return L, R

    def fit(
        self,
        model: Union[DynamicRankTTKAN, TensorTrainKAN],
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        r"""
        Dopasowuje model do danych (X, Y) za pomocą 2-Site DMRG.
        X: shape (N, D), Y: shape (N,) lub (N, 1).
        Zwraca końcowy błąd RMSE.
        """
        N, D = X.shape
        assert D == model.spatial_dim
        Y = np.asarray(Y, dtype=np.float64).reshape(N, 1)
        K1 = model.num_basis
        
        T_list = self._compute_chebyshev_all(model, X)
        
        from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine
        engine = FastCPPKANEngine(spatial_dim=D, degree=model.degree)

        for sweep in range(self.max_sweeps):
            # 1. Left-to-Right 2-Site Sweep: d = 0 .. D - 2
            for d in range(D - 1):
                L, R = self._update_prefixes(model, T_list, N)
                
                L_prev = np.ones((N, 1)) if d == 0 else L[d - 1]
                R_next = np.ones((N, 1)) if d + 1 == D - 1 else R[d + 1]
                
                r_prev = model.ranks[d]
                r_next = model.ranks[d + 2]
                
                T_d = T_list[d]
                T_d1 = T_list[d + 1]
                
                A, B = engine.build_dmrg_normal_equations(
                    L_prev, T_d, T_d1, R_next, Y, alpha=self.alpha
                )
                try:
                    super_core_flat = np.linalg.solve(A, B).ravel()
                except np.linalg.LinAlgError:
                    super_core_flat = (np.linalg.pinv(A) @ B).ravel()
                    
                super_mat = super_core_flat.reshape(r_prev * K1, K1 * r_next)
                U, S, Vh = np.linalg.svd(super_mat, full_matrices=False)
                total_energy = np.sum(S ** 2)
                
                if total_energy < 1e-14:
                    new_rank = self.min_rank
                else:
                    cum_energy = np.cumsum(S ** 2) / total_energy
                    new_rank = int(np.searchsorted(cum_energy, self.variance_threshold)) + 1
                    new_rank = max(self.min_rank, min(new_rank, self.max_rank, len(S)))
                    
                U_trunc = U[:, :new_rank]
                model.cores[d] = U_trunc.reshape(r_prev, K1, new_rank)
                
                SV_trunc = np.diag(S[:new_rank]) @ Vh[:new_rank, :]
                model.cores[d + 1] = SV_trunc.reshape(new_rank, K1, r_next)
                
                model.ranks[d + 1] = new_rank

            # 2. Right-to-Left 2-Site Sweep: d = D - 2 .. 0
            for d in range(D - 2, -1, -1):
                L, R = self._update_prefixes(model, T_list, N)
                
                L_prev = np.ones((N, 1)) if d == 0 else L[d - 1]
                R_next = np.ones((N, 1)) if d + 1 == D - 1 else R[d + 1]
                
                r_prev = model.ranks[d]
                r_next = model.ranks[d + 2]
                
                T_d = T_list[d]
                T_d1 = T_list[d + 1]
                
                A, B = engine.build_dmrg_normal_equations(
                    L_prev, T_d, T_d1, R_next, Y, alpha=self.alpha
                )
                try:
                    super_core_flat = np.linalg.solve(A, B).ravel()
                except np.linalg.LinAlgError:
                    super_core_flat = (np.linalg.pinv(A) @ B).ravel()
                    
                super_mat = super_core_flat.reshape(r_prev * K1, K1 * r_next)
                U, S, Vh = np.linalg.svd(super_mat, full_matrices=False)
                total_energy = np.sum(S ** 2)
                
                if total_energy < 1e-14:
                    new_rank = self.min_rank
                else:
                    cum_energy = np.cumsum(S ** 2) / total_energy
                    new_rank = int(np.searchsorted(cum_energy, self.variance_threshold)) + 1
                    new_rank = max(self.min_rank, min(new_rank, self.max_rank, len(S)))
                    
                Vh_trunc = Vh[:new_rank, :]
                model.cores[d + 1] = Vh_trunc.reshape(new_rank, K1, r_next)
                
                US_trunc = U[:, :new_rank] @ np.diag(S[:new_rank])
                model.cores[d] = US_trunc.reshape(r_prev, K1, new_rank)
                
                model.ranks[d + 1] = new_rank

        Y_pred = model.evaluate(X)
        rmse = float(np.sqrt(np.mean((Y.ravel() - Y_pred) ** 2)))
        return rmse
