import numpy as np
import scipy.linalg
from typing import Callable, List, Tuple, Optional, Union
from src.tdff_net.tt_kan import TensorTrainKAN


def maxvol(A: np.ndarray, tol: float = 1.05, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Algorytm MaxVol (Maximum Volume Submatrix Selection).
    
    Dla zadanej wysokiej macierzy A \in \mathbb{R}^{N \times r} (N >= r) o pełnym rzędzie kolumnowym,
    znajduje podzbiór r wierszy o indeksach I \subset {0, ..., N-1} maksymalizujący objętość
    (moduł wyznacznika |det(A[I, :])|).
    
    Zwraca:
        I: np.ndarray o kształcie (r,) - wybrane indeksy wierszy macierzy A.
        Z: np.ndarray o kształcie (N, r) - macierz współczynników interpolacji Z = A (A[I, :])^{-1}.
           Spełnia Z[I, :] = I_r oraz max_{i, j} |Z_{i, j}| <= tol.
    """
    N, r = A.shape
    if N < r:
        raise ValueError(f"MaxVol requires N >= r, got N={N}, r={r}")
    if N == r:
        return np.arange(r, dtype=np.int32), np.eye(r, dtype=np.float64)

    # 1. Inicjalizacja: dekompozycja QR z pivotingiem kolumnowym na A^T
    _, _, piv = scipy.linalg.qr(A.T, pivoting=True)
    I = np.array(piv[:r], dtype=np.int32)
    
    # 2. Wyznaczenie początkowej macierzy Z = A @ inv(A[I, :])
    A_I = A[I, :]
    try:
        Z = np.linalg.solve(A_I.T, A.T).T.copy()
    except np.linalg.LinAlgError:
        Z = (A @ np.linalg.pinv(A_I)).copy()
        
    # 3. Iteracyjna optymalizacja z formułą Shermana-Morrisona rzędu 1
    for iteration in range(max_iters):
        abs_Z = np.abs(Z)
        max_idx = np.unravel_index(np.argmax(abs_Z), Z.shape)
        i_max, j_max = max_idx
        val_max = abs_Z[i_max, j_max]
        
        if val_max <= tol:
            break
            
        I[j_max] = i_max
        gamma = Z[i_max, j_max]
        
        # Sherman-Morrison rank-1 update
        z_col = Z[:, j_max].copy()
        z_col[i_max] -= 1.0
        z_row = Z[i_max, :].copy()
        
        Z -= np.outer(z_col, z_row) / gamma

    return I, Z


class TTCrossSolver:
    r"""
    Solwer TT-Cross (Tensor Train Cross Approximation) dla Wielowymiarowych Pól KAN (D = 20..100).
    
    Umożliwia bezsiatkowe dopasowanie continuous field f(x_1, ..., x_D) do postaci Tensor Train KAN
    przy złożoności próbek O(D * R^2 * K) zamiast O(K^D).
    
    Matematyka:
    - Węzły próbkowania: Czebyszew-Gauss-Lobatto (CGL) \xi_k = -cos(\pi k / K) w [-1, 1].
    - Lewe i prawe zagnieżdżone wielowskaźniki MaxVol (\mathcal{I}_d, \mathcal{J}_d).
    - Transformacja z bazy węzłowej (nodal) do współczynników modalnych Czebyszewa:
      G^(d)_{:, k, :} = \sum_{i=0}^K (V^{-1})_{k, i} \hat{G}^(d)_{:, i, :}
    """
    def __init__(
        self,
        max_rank: int = 8,
        eps: float = 1e-5,
        max_sweeps: int = 4,
        tol_maxvol: float = 1.05,
        seed: int = 42
    ):
        self.max_rank = max_rank
        self.eps = eps
        self.max_sweeps = max_sweeps
        self.tol_maxvol = tol_maxvol
        self.seed = seed
        self.sample_count = 0

    def _get_cgl_nodes(self, degree: int) -> np.ndarray:
        r"""Zwraca węzły Czebyszewa-Gaussa-Lobatto \xi_k w zakresie [-1.0, 1.0]."""
        K = degree
        if K == 0:
            return np.array([0.0], dtype=np.float64)
        k = np.arange(K + 1, dtype=np.float64)
        nodes = -np.cos(np.pi * k / K)
        nodes[0] = -1.0
        nodes[-1] = 1.0
        return nodes

    def _compute_inverse_chebyshev_vandermonde(self, degree: int) -> np.ndarray:
        r"""
        Tworzy odwrotną macierz Vandermonde'a Czebyszewa V^{-1} o kształcie (K+1, K+1),
        gdzie V_{i, k} = T_k(\xi_i).
        """
        K = degree
        K1 = K + 1
        nodes = self._get_cgl_nodes(degree)
        
        V = np.zeros((K1, K1), dtype=np.float64)
        V[:, 0] = 1.0
        if K >= 1:
            V[:, 1] = nodes
        for k in range(1, K):
            V[:, k + 1] = 2.0 * nodes * V[:, k] - V[:, k - 1]
            
        V_inv = np.linalg.inv(V)
        return V_inv

    def fit_function(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        spatial_dim: int,
        degree: int = 5,
        target_ranks: Optional[Union[int, List[int]]] = None
    ) -> TensorTrainKAN:
        r"""
        Dopasowuje model TensorTrainKAN do czarnej skrzynki func(X) o sygnaturze X -> Y.
        X ma kształt (N, spatial_dim), zwraca Y o kształcie (N,).
        
        Złożoność: O(spatial_dim * max_rank^2 * (degree + 1)) zapytań do func.
        """
        rng = np.random.default_rng(self.seed)
        D = spatial_dim
        K = degree
        K1 = K + 1
        self.sample_count = 0
        
        cgl_nodes = self._get_cgl_nodes(degree)
        V_inv = self._compute_inverse_chebyshev_vandermonde(degree)
        
        # Określenie ograniczenia rang
        if target_ranks is None:
            max_r = self.max_rank
            r_limits = [1] + [max_r] * (D - 1) + [1]
        elif isinstance(target_ranks, int):
            r_limits = [1] + [target_ranks] * (D - 1) + [1]
        else:
            r_limits = list(target_ranks)
            
        # Inicjalizacja prawych wielowskaźników J_sets
        J_sets = [None] * (D + 1)
        J_sets[D] = np.zeros((1, 0), dtype=np.int32)
        
        for d in range(D - 1, 0, -1):
            length = D - d
            init_r = min(r_limits[d], K1 ** min(length, 2))
            init_pts = []
            init_pts.append(np.full(length, K // 2, dtype=np.int32))
            for _ in range(init_r - 1):
                pt = rng.integers(0, K1, size=length, dtype=np.int32)
                init_pts.append(pt)
            J_sets[d] = np.array(init_pts, dtype=np.int32)

        # Inicjalizacja lewych wielowskaźników I_sets
        I_sets = [None] * (D + 1)
        I_sets[0] = np.zeros((1, 0), dtype=np.int32)
        
        # Funkcja pomocnicza do bezpiecznej ewaluacji
        def evaluate_points(pts_indices: np.ndarray) -> np.ndarray:
            coords = cgl_nodes[pts_indices]
            self.sample_count += len(coords)
            vals = func(coords)
            return np.asarray(vals, dtype=np.float64).ravel()

        # Przebiegi TT-Cross (Sweeps)
        for sweep in range(self.max_sweeps):
            # 1. Left-to-Right Sweep: d = 0 .. D - 2
            for d in range(D - 1):
                I_curr = I_sets[d]       # (r_d, d)
                J_next = J_sets[d + 1]   # (r_{d+1}, D - d - 1)
                r_d = len(I_curr)
                r_next = len(J_next)
                
                # Budowa próbek o kształcie (r_d * K1 * r_next, D)
                pts = np.zeros((r_d * K1 * r_next, D), dtype=np.int32)
                row_idx = 0
                for ip in range(r_d):
                    for k in range(K1):
                        for jn in range(r_next):
                            if d > 0:
                                pts[row_idx, :d] = I_curr[ip]
                            pts[row_idx, d] = k
                            if d + 1 < D:
                                pts[row_idx, d + 1:] = J_next[jn]
                            row_idx += 1
                            
                evals = evaluate_points(pts)
                A_mat = evals.reshape(r_d * K1, r_next)
                
                U, S, Vh = np.linalg.svd(A_mat, full_matrices=False)
                
                target_r = r_limits[d + 1]
                if S[0] > 1e-14:
                    rel_s = S / S[0]
                    auto_r = int(np.sum(rel_s > self.eps))
                else:
                    auto_r = 1
                new_r = max(1, min(auto_r, target_r, len(S), r_d * K1))
                
                U_trunc = U[:, :new_r]
                I_piv, _ = maxvol(U_trunc, tol=self.tol_maxvol)
                
                # Aktualizacja I_sets[d + 1] (kształt: new_r, d + 1)
                new_I = np.zeros((new_r, d + 1), dtype=np.int32)
                for r_i, piv_idx in enumerate(I_piv):
                    ip = piv_idx // K1
                    k = piv_idx % K1
                    if d > 0:
                        new_I[r_i, :d] = I_curr[ip]
                    new_I[r_i, d] = k
                I_sets[d + 1] = new_I

            # 2. Right-to-Left Sweep: d = D - 1 .. 1
            for d in range(D - 1, 0, -1):
                I_prev = I_sets[d - 1]   # (r_{d-1}, d - 1)
                J_curr = J_sets[d]       # (r_d, D - d)
                r_prev = len(I_prev)
                r_d = len(J_curr)
                
                pts = np.zeros((r_prev * K1 * r_d, D), dtype=np.int32)
                row_idx = 0
                for ip in range(r_prev):
                    for k in range(K1):
                        for jn in range(r_d):
                            if d - 1 > 0:
                                pts[row_idx, :d - 1] = I_prev[ip]
                            pts[row_idx, d - 1] = k
                            if d < D:
                                pts[row_idx, d:] = J_curr[jn]
                            row_idx += 1
                            
                evals = evaluate_points(pts)
                A_mat = evals.reshape(r_prev, K1 * r_d)
                
                U, S, Vh = np.linalg.svd(A_mat, full_matrices=False)
                
                target_r = r_limits[d]
                if S[0] > 1e-14:
                    rel_s = S / S[0]
                    auto_r = int(np.sum(rel_s > self.eps))
                else:
                    auto_r = 1
                new_r = max(1, min(auto_r, target_r, len(S), K1 * r_d))
                
                V_trunc = Vh[:new_r, :].T  # (K1 * r_d, new_r)
                J_piv, _ = maxvol(V_trunc, tol=self.tol_maxvol)
                
                # Aktualizacja J_sets[d - 1] (kształt: new_r, D - d + 1)
                new_J = np.zeros((new_r, D - d + 1), dtype=np.int32)
                for r_j, piv_idx in enumerate(J_piv):
                    k = piv_idx // r_d
                    jn = piv_idx % r_d
                    new_J[r_j, 0] = k
                    if d < D:
                        new_J[r_j, 1:] = J_curr[jn]
                J_sets[d - 1] = new_J

        # 3. Konstrukcja Ostatecznych Rdzeni TT
        final_ranks = [1] + [len(I_sets[d]) for d in range(1, D)] + [1]
        
        # Obliczenie macierzy przecięcia P_d = A(I_sets[d], J_sets[d]) dla d = 1..D-1
        P_inv_list = [None] * D
        for d in range(1, D):
            I_d = I_sets[d]
            J_d = J_sets[d]
            r_I = len(I_d)
            r_J = len(J_d)
            pts_inter = np.zeros((r_I * r_J, D), dtype=np.int32)
            p_idx = 0
            for ip in range(r_I):
                for jp in range(r_J):
                    pts_inter[p_idx, :d] = I_d[ip]
                    pts_inter[p_idx, d:] = J_d[jp]
                    p_idx += 1
            inter_evals = evaluate_points(pts_inter).reshape(r_I, r_J)
            P_inv = np.linalg.pinv(inter_evals)  # shape: (r_J, r_I)
            P_inv_list[d] = P_inv

        nodal_cores = []
        for d in range(D):
            I_prev = I_sets[d]       # (r_d, d)
            J_next = J_sets[d + 1]   # (r_{d+1}, D - d - 1)
            r_d = len(I_prev)
            r_next = len(J_next)
            
            pts = np.zeros((r_d * K1 * r_next, D), dtype=np.int32)
            row_idx = 0
            for ip in range(r_d):
                for k in range(K1):
                    for jn in range(r_next):
                        if d > 0:
                            pts[row_idx, :d] = I_prev[ip]
                        pts[row_idx, d] = k
                        if d + 1 < D:
                            pts[row_idx, d + 1:] = J_next[jn]
                        row_idx += 1
                        
            evals = evaluate_points(pts)
            raw_core = evals.reshape(r_d, K1, r_next)
            
            if d < D - 1:
                # Absorb P_{d+1}^{-1} do prawego wymiaru: (r_d, K1, r_next) @ (r_next, r_I_{d+1})
                P_inv = P_inv_list[d + 1]
                core_d = np.einsum('rkj, js -> rks', raw_core, P_inv)
            else:
                core_d = raw_core
                
            nodal_cores.append(core_d)

        # 4. Transformacja Modalna do Współczynników Czebyszewa
        from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine
        engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
        modal_cores = []
        for d in range(D):
            n_core = nodal_cores[d]  # (r_d, K1, r_next)
            m_core = engine.project_chebyshev_modal(n_core, V_inv)
            modal_cores.append(m_core)

        # Złożenie modelu TensorTrainKAN
        model = TensorTrainKAN(spatial_dim=D, ranks=final_ranks, degree=degree)
        model.cores = modal_cores
        return model
