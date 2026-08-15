import numpy as np
from typing import List, Tuple

class TDFFNet:
    r"""
    Tensor-Decomposed Functional Field Network (TDFF-Net).
    
    Ciągła reprezentacja geometrii i pól wartości (Signed Distance Fields / Pól Przestrzennych)
    oparta na Rozkładzie Tensorowym CP (Canonical Polyadic) 1D KAN baz Czebyszewa.
    
    Wzór Pola:
    f(x_1, x_2, ..., x_D) = \sum_{r=1}^R \lambda_r \prod_{d=1}^D \phi_{r}^{(d)}(x_d)
    gdzie \phi_{r}^{(d)}(x_d) = \sum_{k=0}^K W_{r,k}^{(d)} T_k(x_d).
    
    Zalety Architektoniczne:
    - Bez pętli Raymarchingu (O(1) ewaluacji ciągłego punktu w czasie < 5 ms dla 50,000 punktów).
    - Bez siatek wielokątnych (Mesh-Free continuous implicit field).
    - Analityczny Gradient Przestrzenny \nabla f(x_1, ..., x_D) z tożsamości pochodnych Czebyszewa.
    """
    def __init__(self, spatial_dim: int = 3, rank: int = 16, degree: int = 5):
        self.spatial_dim = spatial_dim  # D np. (x, y, z)
        self.rank = rank                # R (liczba komponentów rozkładu CP)
        self.degree = degree            # K (stopień wielomianów Czebyszewa)
        
        self.lambdas = np.ones(self.rank)
        self.factors = [
            np.random.normal(0.0, 1.0 / np.sqrt(self.degree + 1), size=(self.rank, self.degree + 1))
            for _ in range(self.spatial_dim)
        ]

    def _compute_chebyshev(self, x_d: np.ndarray) -> np.ndarray:
        """Szybkie wyliczenie samych wielomianów Czebyszewa T_k (bez pochodnych)."""
        x_norm = np.clip(x_d, -1.0, 1.0)
        N = len(x_norm)
        T = np.empty((N, self.degree + 1))
        T[:, 0] = 1.0
        if self.degree >= 1:
            T[:, 1] = x_norm
        for k in range(2, self.degree + 1):
            T[:, k] = 2.0 * x_norm * T[:, k - 1] - T[:, k - 2]
        return T

    def _compute_chebyshev_and_derivatives(self, x_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wyznacza T_k oraz analityczne dT_k/dx dla wyliczania analitycznego gradientu."""
        x_norm = np.clip(x_d, -1.0, 1.0)
        N = len(x_norm)
        T = np.empty((N, self.degree + 1))
        dT = np.empty((N, self.degree + 1))
        
        T[:, 0] = 1.0
        dT[:, 0] = 0.0
        
        if self.degree >= 1:
            T[:, 1] = x_norm
            dT[:, 1] = 1.0
            
        for k in range(2, self.degree + 1):
            T[:, k] = 2.0 * x_norm * T[:, k - 1] - T[:, k - 2]
            dT[:, k] = 2.0 * T[:, k - 1] + 2.0 * x_norm * dT[:, k - 1] - dT[:, k - 2]
            
        return T, dT

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Hyper-fast vectorized evaluation of continuous spatial field at points X (N, D).
        """
        N, D = X.shape
        cp_product = np.ones((N, self.rank))
        
        for d in range(D):
            T = self._compute_chebyshev(X[:, d]) # (N, K+1)
            phi_d = T @ self.factors[d].T       # (N, R)
            cp_product *= phi_d
            
        f_val = cp_product @ self.lambdas
        return f_val

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Analityczny gradient pola \nabla f(X) w punktach X (N, D).
        """
        N, D = X.shape
        grad = np.zeros((N, D))
        
        phi_evals = np.empty((N, self.rank, D))
        dphi_evals = np.empty((N, self.rank, D))
        
        for d in range(D):
            T, dT = self._compute_chebyshev_and_derivatives(X[:, d])
            phi_evals[:, :, d] = T @ self.factors[d].T
            dphi_evals[:, :, d] = dT @ self.factors[d].T
            
        for dim in range(D):
            term_evals = phi_evals.copy()
            term_evals[:, :, dim] = dphi_evals[:, :, dim]
            
            cp_grad_prod = np.prod(term_evals, axis=2)
            grad[:, dim] = cp_grad_prod @ self.lambdas
            
        return grad
