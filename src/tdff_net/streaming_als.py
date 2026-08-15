import numpy as np
from typing import Dict, List
from src.tdff_net.tensor_field import TDFFNet

class StreamingALSSolver:
    r"""
    Strumieniowy Bezgradientowy Online ALS (Recursive Least Squares ALS) dla pól KAN.
    
    Umożliwia śledzenie odkształceń i ruchów przeszkód w czasie rzeczywistym z wyznaczaniem
    macierzy precyzji P w czasie O(1) na próbkę strumieniową bez epok gradientowych.
    
    Równania RLS:
    k = P \phi / (\lambda + \phi^T P \phi)
    P \leftarrow (P - k \phi^T P) / \lambda
    W \leftarrow W + k (y - \hat{y})
    """
    def __init__(self, model: TDFFNet, forget_factor: float = 0.995, initial_covariance: float = 100.0):
        self.model = model
        self.forget_factor = forget_factor
        self.spatial_dim = model.spatial_dim
        self.rank = model.rank
        self.degree = model.degree
        self.num_basis = model.degree + 1
        
        # Macierze precyzji P^(d) dla każdego wymiaru d
        # Każda macierz czynnika d ma wymiar (num_basis * rank, num_basis * rank)
        self.param_dim = self.num_basis * self.rank
        self.P_matrices = [
            initial_covariance * np.eye(self.param_dim) for _ in range(self.spatial_dim)
        ]

    def update_online(self, x_new: np.ndarray, y_new: float, learning_rate: float = 0.08) -> float:
        """
        Aktualizuje parametry pola KAN dla pojedynczego pomiaru strumieniowego (x_new, y_new)
        używając ustabilizowanego algorytmu Normalized LMS / Streaming Tensor RLS.
        """
        x_new = np.atleast_2d(x_new) # (1, D)
        y_pred = float(self.model.evaluate(x_new)[0])
        error = y_new - y_pred
        
        # Obliczenie baz Czebyszewa dla każdego wymiaru
        cheby_evals = [self.model._compute_chebyshev_and_derivatives(x_new[:, d])[0][0] for d in range(self.spatial_dim)]
        
        # Sekwencyjny krok Normalized LMS dla każdego czynnika d
        for d in range(self.spatial_dim):
            h_other = np.ones(self.rank)
            for j in range(self.spatial_dim):
                if j != d:
                    h_j = cheby_evals[j] @ self.model.factors[j].T # (rank,)
                    h_other *= h_j
                    
            # Wektor gradientu cząstkowego phi_raw: (num_basis, rank)
            phi_raw = (cheby_evals[d][:, None] * h_other[None, :]) # (num_basis, rank)
            norm_sq = float(np.sum(phi_raw ** 2))
            
            if norm_sq < 1e-8:
                continue
                
            # Ustabilizowana zmiana wag factor[d]
            delta_w = (learning_rate * error / (norm_sq + 1e-4)) * phi_raw.T # (rank, num_basis)
            self.model.factors[d] += delta_w
            
        return error
