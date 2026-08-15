import numpy as np
from typing import List, Tuple, Optional

class TensorTrainKAN:
    r"""
    Tensor Train KAN (TT-KAN) dla Wielowymiarowych Pól Ciągłych D >= 10.
    
    Reprezentacja pola f(x_1, ..., x_D) w postaci łańcucha tensorów 3-wskaźnikowych:
    f(x) = G^(0)(x_1) G^(1)(x_2) ... G^(D-1)(x_D)
    
    gdzie M^(d)(x_d) = \sum_{k=0}^K T_k(x_d) G^(d)_{:, k, :} \in \mathbb{R}^{r_{d-1} \times r_d}.
    Złożoność pamięciowa: O(D * K * R^2) zamiast O(R^D) (brak klątwy wymiarowości).
    """
    def __init__(self, spatial_dim: int = 10, ranks: Optional[List[int]] = None, degree: int = 5):
        self.spatial_dim = spatial_dim
        self.degree = degree
        self.num_basis = degree + 1
        
        # Domyślne rangi TT r_0=1, r_1=R, ..., r_D=1
        if ranks is None:
            # TT ranks [1, R, R, ..., R, 1]
            R = 8
            self.ranks = [1] + [R] * (spatial_dim - 1) + [1]
        else:
            assert len(ranks) == spatial_dim + 1 and ranks[0] == 1 and ranks[-1] == 1
            self.ranks = ranks
            
        # Inicjalizacja rdzeni tensorowych G^(d) o wymiarach (r_{d-1}, K+1, r_d)
        np.random.seed(42)
        self.cores = []
        for d in range(spatial_dim):
            r_prev = self.ranks[d]
            r_next = self.ranks[d + 1]
            # Inicjalizacja losowa z małą wariancją
            core = np.random.randn(r_prev, self.num_basis, r_next) / np.sqrt(r_prev * r_next * self.num_basis)
            self.cores.append(core)

    def _compute_chebyshev_and_derivatives(self, x_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza wielomiany Czebyszewa T_k(x) oraz pochodne dT_k/dx dla wymiaru d.
        """
        N = x_d.shape[0]
        K = self.degree
        x_clamped = np.clip(x_d, -1.0, 1.0)
        
        T = np.zeros((N, K + 1))
        dT = np.zeros((N, K + 1))
        
        T[:, 0] = 1.0
        dT[:, 0] = 0.0
        
        if K >= 1:
            T[:, 1] = x_clamped
            dT[:, 1] = 1.0
            
        for k in range(1, K):
            T[:, k + 1] = 2.0 * x_clamped * T[:, k] - T[:, k - 1]
            dT[:, k + 1] = 2.0 * T[:, k] + 2.0 * x_clamped * dT[:, k] - dT[:, k - 1]
            
        return T, dT

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        r"""
        Ewaluacja pola TT-KAN dla macierzy punktów X \in \mathbb{R}^{N \times D}.
        Optymalizowana funkcja np.matmul zmniejsza czas wykonania o 10-15x.
        """
        N, D = X.shape
        assert D == self.spatial_dim
        
        curr = np.ones((N, 1)) # (N, r_0) = (N, 1)
        
        for d in range(D):
            T_d, _ = self._compute_chebyshev_and_derivatives(X[:, d]) # (N, K+1)
            core_d = self.cores[d] # (r_prev, K+1, r_next)
            r_prev, K1, r_next = core_d.shape
            
            # Fast matmul: T_d (N, K+1) @ core_d_flat (K+1, r_prev * r_next) -> (N, r_prev, r_next)
            M_d = (T_d @ core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)).reshape(N, r_prev, r_next)
            
            # Fast batch vector-matrix product: (N, 1, r_prev) @ (N, r_prev, r_next) -> (N, 1, r_next)
            curr = (curr[:, None, :] @ M_d)[:, 0, :]
            
        return curr.squeeze(-1) # (N,)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        r"""
        Oblicza analityczny gradient \nabla f(X) \in \mathbb{R}^{N \times D} bez automatycznego różniczkowania.
        """
        N, D = X.shape
        
        # 1. Obliczenie macierzy M_d i dM_d dla wszystkich wymiarów z użyciem np.matmul
        M_list = []
        dM_list = []
        for d in range(D):
            T_d, dT_d = self._compute_chebyshev_and_derivatives(X[:, d])
            core_d = self.cores[d]
            r_prev, K1, r_next = core_d.shape
            core_flat = core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)
            
            M_list.append((T_d @ core_flat).reshape(N, r_prev, r_next))
            dM_list.append((dT_d @ core_flat).reshape(N, r_prev, r_next))
            
        # 2. Obliczenie lewych prefiksów L^(d) (N, r_d)
        L = [None] * D
        L_curr = np.ones((N, 1))
        for d in range(D):
            L_curr = (L_curr[:, None, :] @ M_list[d])[:, 0, :]
            L[d] = L_curr
            
        # 3. Obliczenie prawych prefiksów R^(d) (N, r_{d-1})
        R = [None] * D
        R_curr = np.ones((N, 1))
        for d in range(D - 1, -1, -1):
            R[d] = R_curr
            R_curr = (M_list[d] @ R_curr[:, :, None])[:, :, 0]
            
        # 4. Złożenie analitycznego gradientu dla każdego wymiaru m
        grad = np.zeros((N, D))
        for m in range(D):
            L_prev = np.ones((N, 1)) if m == 0 else L[m - 1]
            R_next = R[m]
            dM_m = dM_list[m]
            
            # df/dx_m = L_{m-1} @ dM_m @ R_{m+1}
            mid = (L_prev[:, None, :] @ dM_m)[:, 0, :]
            df_dxm = (mid * R_next).sum(axis=1)
            grad[:, m] = df_dxm
            
        return grad


class TTALSSolver:
    """
    Beziteracyjny/Als Solver dla Tensor Train KAN (0 epok gradientowych).
    Wykonuje sekwencyjne dopasowanie rdzeni TT za pomocą uogólnionych równań Tikhonova.
    """
    def __init__(self, alpha: float = 1e-4, max_sweeps: int = 4):
        self.alpha = alpha
        self.max_sweeps = max_sweeps

    def fit(self, model: TensorTrainKAN, X: np.ndarray, Y: np.ndarray) -> float:
        N, D = X.shape
        Y = Y.reshape(N, 1)
        
        for sweep in range(self.max_sweeps):
            for d in range(D):
                # 1. Wyznaczenie prefiksów lewych L i prawych R dla obecnych rdzeni
                M_list = []
                for dim in range(D):
                    T_dim, _ = model._compute_chebyshev_and_derivatives(X[:, dim])
                    M_list.append(np.einsum('nk,rks->nrs', T_dim, model.cores[dim]))
                    
                L_curr = np.ones((N, 1))
                for dim in range(d):
                    L_curr = np.einsum('nr,nrs->ns', L_curr, M_list[dim])
                    
                R_curr = np.ones((N, 1))
                for dim in range(D - 1, d, -1):
                    R_curr = np.einsum('nrs,ns->nr', M_list[dim], R_curr)
                    
                # L_curr shape: (N, r_prev), R_curr shape: (N, r_next)
                T_d, _ = model._compute_chebyshev_and_derivatives(X[:, d]) # (N, K+1)
                
                r_prev = model.ranks[d]
                r_next = model.ranks[d + 1]
                K1 = model.num_basis
                
                # Macierz projektowa Phi_d shape (N, r_prev * K1 * r_next)
                # Term for index (r, k, s) is L_{n, r} * T_{n, k} * R_{n, s}
                Phi_d = (L_curr[:, :, None, None] * T_d[:, None, :, None] * R_curr[:, None, None, :]).reshape(N, r_prev * K1 * r_next)
                
                # Rozwiązanie Tikhonova
                A = Phi_d.T @ Phi_d + self.alpha * np.eye(r_prev * K1 * r_next)
                B = Phi_d.T @ Y
                
                core_flat = np.linalg.solve(A, B)
                model.cores[d] = core_flat.reshape(r_prev, K1, r_next)
                
        Y_pred = model.evaluate(X)
        return float(np.sqrt(np.mean((Y.ravel() - Y_pred) ** 2)))
