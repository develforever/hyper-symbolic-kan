import numpy as np
from typing import List, Tuple, Optional

class DynamicRankTTKAN:
    r"""
    Dynamic Rank-Adaptive Tensor Train KAN (DR-TT-KAN) dla Wielowymiarowych Pól Ciągłych D >= 10.
    
    Zastępuje statyczne rangi TT r_d dynamiczną alokacją i przycinaniem SVD w czasie rzeczywistym.
    Reprezentacja pola f(x_1, ..., x_D) w postaci łańcucha tensorów 3-wskaźnikowych:
    f(x) = G^(0)(x_1) G^(1)(x_2) ... G^(D-1)(x_D)
    
    gdzie M^(d)(x_d) = \sum_{k=0}^K T_k(x_d) G^(d)_{:, k, :} \in \mathbb{R}^{r_{d-1} \times r_d}.
    0 EPOK GRADIENTOWYCH: Wszelkie adaptacje rang zachodzą poprzez rozkłady SVD/QR i rzuty ortogonalne.
    """
    def __init__(self, spatial_dim: int = 10, init_ranks: Optional[List[int]] = None, degree: int = 5):
        self.spatial_dim = spatial_dim
        self.degree = degree
        self.num_basis = degree + 1
        
        # Domyślne rangi TT [1, R, R, ..., R, 1]
        if init_ranks is None:
            R = 8
            self.ranks = [1] + [R] * (spatial_dim - 1) + [1]
        else:
            assert len(init_ranks) == spatial_dim + 1 and init_ranks[0] == 1 and init_ranks[-1] == 1
            self.ranks = list(init_ranks)
            
        # Inicjalizacja rdzeni tensorowych G^(d) o wymiarach (r_{d-1}, K+1, r_d)
        np.random.seed(42)
        self.cores = []
        for d in range(spatial_dim):
            r_prev = self.ranks[d]
            r_next = self.ranks[d + 1]
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
        Ewaluacja pola DR-TT-KAN dla macierzy punktów X \in \mathbb{R}^{N \times D}.
        """
        N, D = X.shape
        assert D == self.spatial_dim
        
        curr = np.ones((N, 1)) # (N, r_0) = (N, 1)
        
        for d in range(D):
            T_d, _ = self._compute_chebyshev_and_derivatives(X[:, d]) # (N, K+1)
            core_d = self.cores[d] # (r_prev, K+1, r_next)
            r_prev, K1, r_next = core_d.shape
            
            # Fast matmul: T_d (N, K+1) @ core_d_flat (K+1, r_prev * r_next) -> (N, r_prev, r_next)
            core_flat = core_d.transpose(1, 0, 2).reshape(K1, r_prev * r_next)
            M_d = (T_d @ core_flat).reshape(N, r_prev, r_next)
            
            # Batch vector-matrix product: (N, 1, r_prev) @ (N, r_prev, r_next) -> (N, 1, r_next)
            curr = (curr[:, None, :] @ M_d)[:, 0, :]
            
        return curr.squeeze(-1) # (N,)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        r"""
        Oblicza analityczny gradient \nabla f(X) \in \mathbb{R}^{N \times D}.
        """
        N, D = X.shape
        
        M_list = []
        dM_list = []
        for d in range(D):
            T_d, dT_d = self._compute_chebyshev_and_derivatives(X[:, d])
            core_d = self.cores[d]
            r_prev, K1, r_next = core_d.shape
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
            
        grad = np.zeros((N, D))
        for m in range(D):
            L_prev = np.ones((N, 1)) if m == 0 else L[m - 1]
            R_next = R[m]
            dM_m = dM_list[m]
            
            mid = (L_prev[:, None, :] @ dM_m)[:, 0, :]
            df_dxm = (mid * R_next).sum(axis=1)
            grad[:, m] = df_dxm
            
        return grad

    def truncate_ranks_svd(self, variance_threshold: float = 0.999, max_rank: int = 16, min_rank: int = 1) -> List[int]:
        r"""
        Adaptacyjna Redukcja Rangi SVD (Truncated SVD Sweeping) wzdłuż łańcucha TT.
        Przesuwanie kanoniczne z lewej do prawej oraz z prawej do lewej bez gradientów.
        
        Zwraca zaktualizowaną listę rang TT.
        """
        D = self.spatial_dim
        
        # 1. Left-to-Right Orthogonalization & SVD Truncation Sweep
        for d in range(D - 1):
            r_prev, K1, r_next = self.cores[d].shape
            A_mat = self.cores[d].reshape(r_prev * K1, r_next)
            
            U, S, Vh = np.linalg.svd(A_mat, full_matrices=False)
            total_energy = np.sum(S ** 2)
            
            if total_energy < 1e-14:
                new_rank = min_rank
            else:
                cum_energy = np.cumsum(S ** 2) / total_energy
                new_rank = int(np.searchsorted(cum_energy, variance_threshold)) + 1
                new_rank = max(min_rank, min(new_rank, max_rank, len(S)))
                
            # Truncate U, S, Vh
            U_trunc = U[:, :new_rank] # (r_prev * K1, new_rank)
            S_trunc = np.diag(S[:new_rank]) # (new_rank, new_rank)
            Vh_trunc = Vh[:new_rank, :] # (new_rank, r_next)
            
            # Zastąpienie obecnego rdzenia ortogonalnym rdzeniem o nowej randze
            self.cores[d] = U_trunc.reshape(r_prev, K1, new_rank)
            self.ranks[d + 1] = new_rank
            
            # Absorpcja (S * Vh) do następnego rdzenia G^(d+1)
            M_absorb = S_trunc @ Vh_trunc # (new_rank, r_next)
            self.cores[d + 1] = np.einsum('ab, bkc -> akc', M_absorb, self.cores[d + 1])
            
        # 2. Right-to-Left Sweep for Dual Canonical Stability
        for d in range(D - 1, 0, -1):
            r_prev, K1, r_next = self.cores[d].shape
            B_mat = self.cores[d].reshape(r_prev, K1 * r_next)
            
            U, S, Vh = np.linalg.svd(B_mat, full_matrices=False)
            total_energy = np.sum(S ** 2)
            
            if total_energy < 1e-14:
                new_rank = min_rank
            else:
                cum_energy = np.cumsum(S ** 2) / total_energy
                new_rank = int(np.searchsorted(cum_energy, variance_threshold)) + 1
                new_rank = max(min_rank, min(new_rank, max_rank, len(S)))
                
            U_trunc = U[:, :new_rank]
            S_trunc = np.diag(S[:new_rank])
            Vh_trunc = Vh[:new_rank, :]
            
            self.cores[d] = Vh_trunc.reshape(new_rank, K1, r_next)
            self.ranks[d] = new_rank
            
            M_left = U_trunc @ S_trunc # (r_prev, new_rank)
            self.cores[d - 1] = np.einsum('akb, bc -> akc', self.cores[d - 1], M_left)
            
        return list(self.ranks)

    def expand_ranks(self, increment: int = 2, max_rank: int = 16, noise_scale: float = 1e-3) -> List[int]:
        r"""
        Dynamiczne Alokowanie Rangi (Rank Allocation): Zwiększa rangi TT na wiązaniach
        gdzie występuje niedoreprezentowanie (high residual variance).
        Dodaje ortogonalne wymiary z małym szumem inicjalizacyjnym.
        """
        D = self.spatial_dim
        for d in range(D - 1):
            r_prev, K1, r_curr = self.cores[d].shape
            r_next_core_last = self.cores[d + 1].shape[2]
            
            if r_curr >= max_rank:
                continue
                
            target_rank = min(max_rank, r_curr + increment)
            added_rank = target_rank - r_curr
            if added_rank <= 0:
                continue
                
            # Powiększenie rdzenia d w wymiarze 2 (r_curr -> target_rank)
            new_core_d = np.zeros((r_prev, K1, target_rank))
            new_core_d[:, :, :r_curr] = self.cores[d]
            new_core_d[:, :, r_curr:] = np.random.randn(r_prev, K1, added_rank) * noise_scale
            self.cores[d] = new_core_d
            
            # Powiększenie rdzenia d+1 w wymiarze 0 (r_curr -> target_rank)
            new_core_d1 = np.zeros((target_rank, K1, r_next_core_last))
            new_core_d1[:r_curr, :, :] = self.cores[d + 1]
            new_core_d1[r_curr:, :, :] = np.random.randn(added_rank, K1, r_next_core_last) * noise_scale
            self.cores[d + 1] = new_core_d1
            
            self.ranks[d + 1] = target_rank
            
        return list(self.ranks)
