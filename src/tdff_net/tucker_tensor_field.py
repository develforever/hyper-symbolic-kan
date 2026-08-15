import numpy as np
from typing import List, Tuple, Union

class TuckerTDFFNet:
    r"""
    Wielorozdzielcze Pole Tensorowe Tuckera (Hierarchical Tucker TDFF-Net).
    
    Ciągła reprezentacja geometrii o wysokiej częstotliwości (Sharp Boundary SDF)
    z rdzeniem tensorowym G \in \mathbb{R}^{R_1 \times ... \times R_D} oraz wielomianami Czebyszewa KAN.
    
    Wzór Pola:
    f(x_1, ..., x_D) = \sum_{r_1=1}^{R_1} ... \sum_{r_D=1}^{R_D} \mathcal{G}_{r_1,...,r_D} \prod_{d=1}^D \left( \sum_{k=0}^K W_{r_d, k}^{(d)} T_k(x_d) \right)
    """
    def __init__(self, spatial_dim: int = 3, ranks: Union[int, List[int]] = 8, degree: int = 5):
        self.spatial_dim = spatial_dim
        if isinstance(ranks, int):
            self.ranks = [ranks] * self.spatial_dim
        else:
            self.ranks = list(ranks)
        self.degree = degree
        
        # Inicjalizacja rdzenia tensorowego \mathcal{G}
        self.core = np.random.normal(0.0, 1.0 / np.sqrt(np.prod(self.ranks)), size=tuple(self.ranks))
        
        # Inicjalizacja macierzy czynnikowych W^(d) dla każdego wymiaru d
        self.factors = [
            np.random.normal(0.0, 1.0 / np.sqrt(self.degree + 1), size=(self.ranks[d], self.degree + 1))
            for d in range(self.spatial_dim)
        ]

    def _compute_chebyshev_and_derivatives(self, x_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wyznacza wielomiany T_k oraz analityczne pochodne dT_k/dx."""
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
        Wektorowa ewaluacja pola tensorowego Tuckera w punktach X (N, D).
        """
        N, D = X.shape
        phi_list = []
        for d in range(D):
            T, _ = self._compute_chebyshev_and_derivatives(X[:, d])
            phi_d = T @ self.factors[d].T # (N, R_d)
            phi_list.append(phi_d)
            
        if D == 2:
            # f(i) = sum_{r,s} G_{r,s} * phi_0(i,r) * phi_1(i,s)
            f_val = np.einsum('ir,is,rs->i', phi_list[0], phi_list[1], self.core)
        elif D == 3:
            # f(i) = sum_{r,s,t} G_{r,s,t} * phi_0(i,r) * phi_1(i,s) * phi_2(i,t)
            f_val = np.einsum('ir,is,it,rst->i', phi_list[0], phi_list[1], phi_list[2], self.core)
        else:
            # Pętla po punktach dla ogólnego wymiaru D
            f_val = np.zeros(N)
            for i in range(N):
                val = self.core.copy()
                for d in range(D):
                    val = np.tensordot(val, phi_list[d][i], axes=([0], [0]))
                f_val[i] = val
        return f_val

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Analityczny gradient pola \nabla f(X) w punktach X (N, D).
        """
        N, D = X.shape
        grad = np.zeros((N, D))
        
        phi_list = []
        dphi_list = []
        for d in range(D):
            T, dT = self._compute_chebyshev_and_derivatives(X[:, d])
            phi_list.append(T @ self.factors[d].T)
            dphi_list.append(dT @ self.factors[d].T)
            
        for dim in range(D):
            current_evals = [dphi_list[d] if d == dim else phi_list[d] for d in range(D)]
            if D == 2:
                grad[:, dim] = np.einsum('ir,is,rs->i', current_evals[0], current_evals[1], self.core)
            elif D == 3:
                grad[:, dim] = np.einsum('ir,is,it,rst->i', current_evals[0], current_evals[1], current_evals[2], self.core)
            else:
                for i in range(N):
                    val = self.core.copy()
                    for d in range(D):
                        val = np.tensordot(val, current_evals[d][i], axes=([0], [0]))
                    grad[i, dim] = val
                    
        return grad
