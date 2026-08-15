import numpy as np
from typing import Tuple, List
from src.tdff_net.dr_tt_kan import DynamicRankTTKAN

class DynamicRankTTALSSolver:
    r"""
    Beziteracyjny/Als Solver z Adaptacją Rang SVD dla DynamicRankTTKAN (0 epok gradientowych).
    
    Wykonywane kroki:
    1. Sekwencyjne dopasowanie rdzeni TT metodą Tikhonova (ALS).
    2. Adaptacyjne przycinanie rang TT (Truncated SVD Sweeping) na podstawie skumulowanej wariancji osobliwej.
    """
    def __init__(self, alpha: float = 1e-5, max_sweeps: int = 6, variance_threshold: float = 0.9999, max_rank: int = 16):
        self.alpha = alpha
        self.max_sweeps = max_sweeps
        self.variance_threshold = variance_threshold
        self.max_rank = max_rank

    def fit(self, model: DynamicRankTTKAN, X: np.ndarray, Y: np.ndarray, adapt_ranks: bool = True) -> float:
        """
        Dopasowanie modelu DR-TT-KAN do danych (X, Y) z 0 epokami gradientowymi i on-the-fly SVD truncating.
        """
        N, D = X.shape
        Y = Y.reshape(N, 1)
        
        for sweep in range(self.max_sweeps):
            for d in range(D):
                # Wyznaczenie prefiksów L (lewe) i R (prawe) dla obecnych rdzeni
                M_list = []
                for dim in range(D):
                    T_dim, _ = model._compute_chebyshev_and_derivatives(X[:, dim])
                    core_dim = model.cores[dim]
                    r_p, K1, r_n = core_dim.shape
                    core_flat = core_dim.transpose(1, 0, 2).reshape(K1, r_p * r_n)
                    M_list.append((T_dim @ core_flat).reshape(N, r_p, r_n))
                    
                L_curr = np.ones((N, 1))
                for dim in range(d):
                    L_curr = (L_curr[:, None, :] @ M_list[dim])[:, 0, :]
                    
                R_curr = np.ones((N, 1))
                for dim in range(D - 1, d, -1):
                    R_curr = (M_list[dim] @ R_curr[:, :, None])[:, :, 0]
                    
                T_d, _ = model._compute_chebyshev_and_derivatives(X[:, d]) # (N, K+1)
                
                r_prev = model.ranks[d]
                r_next = model.ranks[d + 1]
                K1 = model.num_basis
                
                # Macierz projektowa Phi_d: shape (N, r_prev * K1 * r_next)
                Phi_d = (L_curr[:, :, None, None] * T_d[:, None, :, None] * R_curr[:, None, None, :]).reshape(N, r_prev * K1 * r_next)
                
                # Tikhonov Ridge Solve
                A = Phi_d.T @ Phi_d + self.alpha * np.eye(r_prev * K1 * r_next)
                B = Phi_d.T @ Y
                
                core_flat = np.linalg.solve(A, B)
                model.cores[d] = core_flat.reshape(r_prev, K1, r_next)
                
            # SVD Rank Adaptation po każdym przebiegu ALS
            if adapt_ranks and sweep >= 1:
                model.truncate_ranks_svd(variance_threshold=self.variance_threshold, max_rank=self.max_rank)
                
        # Finałowa truncacja SVD
        if adapt_ranks:
            model.truncate_ranks_svd(variance_threshold=self.variance_threshold, max_rank=self.max_rank)
            
        Y_pred = model.evaluate(X)
        rmse = float(np.sqrt(np.mean((Y.ravel() - Y_pred) ** 2)))
        return rmse
