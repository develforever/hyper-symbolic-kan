import numpy as np
from typing import Tuple
from src.tdff_net.tucker_tensor_field import TuckerTDFFNet

class TuckerALSSolver:
    r"""
    Beziteracyjny/Analityczny Solver Alternating Least Squares (ALS) dla TuckerTDFFNet.
    
    Wykonywane kroki (0 Epok Gradientowych):
    1. Liniowe dopasowanie macierzy czynnikowych W^(d) i rdzenia tensorowego G metodą Tikhonova.
    2. Adaptacyjna Redukcja Rangi SVD (Adaptive Truncated SVD): odcina niepotrzebne komponenty w 0.1 ms.
    """
    def __init__(self, alpha: float = 1e-4, max_als_iters: int = 8, variance_threshold: float = 0.999):
        self.alpha = alpha
        self.max_als_iters = max_als_iters
        self.variance_threshold = variance_threshold

    def fit(self, model: TuckerTDFFNet, X: np.ndarray, Y: np.ndarray) -> float:
        """
        X: Punkty w przestrzeni (N, D)
        Y: Docelowe pola/SDF (N,)

        Raises:
            NotImplementedError: dla D != 2. Cała logika ALS poniżej jest
                warunkowana `if D == 2`; dla pozostałych D pętla wykonywała się
                bezczynnie i zwracała MSE losowej inicjalizacji jako metrykę
                sukcesu (audyt M3). Wariant ogólny nie jest zaimplementowany.
        """
        N, D = X.shape
        if D != 2:
            raise NotImplementedError(f"TuckerALS supports D=2, got D={D}")
        Y = Y.ravel()
        
        for it in range(self.max_als_iters):
            # 1. ALS po macierzach czynnikowych W^(d)
            for d in range(D):
                T_d, _ = model._compute_chebyshev_and_derivatives(X[:, d]) # (N, K+1)
                K1 = model.degree + 1
                R_d = model.ranks[d]
                
                # Wyznaczenie wkładu od pozostałych wymiarów
                phi_other = []
                for j in range(D):
                    if j != d:
                        T_j, _ = model._compute_chebyshev_and_derivatives(X[:, j])
                        phi_other.append(T_j @ model.factors[j].T)
                        
                # Konstrukcja macierzy projektowej Phi_d dla wymiaru d
                # D=2: phi_other[0] ma rozmiar (N, R_other)
                if D == 2:
                    other_phi = phi_other[0] # (N, R_1)
                    # Core shape: (R_0, R_1) if d=0, or (R_1, R_0) if d=1
                    # Matrix multiplication over core
                    if d == 0:
                        # factor W0 has shape (R0, K1). Phi0 = T0 @ W0^T (N, R0)
                        # f_val = einsum('ir,is,rs->i') = sum_r (phi0_ir * sum_s G_rs phi1_is)
                        # H_r(i) = sum_s G_rs phi1_is
                        H = other_phi @ model.core.T # (N, R0)
                    else:
                        H = other_phi @ model.core # (N, R1)
                        
                    Phi_design = np.zeros((N, R_d * K1))
                    for r in range(R_d):
                        scale_r = H[:, r]
                        start_c = r * K1
                        end_c = (r + 1) * K1
                        Phi_design[:, start_c:end_c] = T_d * scale_r[:, np.newaxis]
                        
                    A = Phi_design.T @ Phi_design + self.alpha * np.eye(R_d * K1)
                    B = Phi_design.T @ Y
                    w_flat = np.linalg.solve(A, B)
                    model.factors[d] = w_flat.reshape(R_d, K1)

            # 2. Optymalizacja czysto liniowa rdzenia tensorowego G
            phi_evals = []
            for d in range(D):
                T_d, _ = model._compute_chebyshev_and_derivatives(X[:, d])
                phi_evals.append(T_d @ model.factors[d].T)
                
            if D == 2:
                # Phi_core o rozmiarze (N, R0 * R1)
                R0, R1 = model.ranks
                Phi_core = (phi_evals[0][:, :, np.newaxis] * phi_evals[1][:, np.newaxis, :]).reshape(N, R0 * R1)
                A_core = Phi_core.T @ Phi_core + self.alpha * np.eye(R0 * R1)
                B_core = Phi_core.T @ Y
                g_flat = np.linalg.solve(A_core, B_core)
                model.core = g_flat.reshape(R0, R1)

        # 3. Adaptacyjna Redukcja Rangi SVD (Truncated SVD)
        self.adaptive_svd_truncate(model)

        # 4. Ponowne dopasowanie rdzenia tensorowego G dla zredukowanych rang
        phi_evals = [T_d @ model.factors[d].T for d, (T_d, _) in enumerate([model._compute_chebyshev_and_derivatives(X[:, d]) for d in range(D)])]
        if D == 2:
            R0, R1 = model.ranks
            Phi_core = (phi_evals[0][:, :, np.newaxis] * phi_evals[1][:, np.newaxis, :]).reshape(N, R0 * R1)
            A_core = Phi_core.T @ Phi_core + self.alpha * np.eye(R0 * R1)
            B_core = Phi_core.T @ Y
            g_flat = np.linalg.solve(A_core, B_core)
            model.core = g_flat.reshape(R0, R1)

        Y_pred = model.evaluate(X)
        mse = float(np.mean((Y - Y_pred) ** 2))
        return mse

    def adaptive_svd_truncate(self, model: TuckerTDFFNet):
        """
        Trunkacja SVD macierzy czynnikowych i rdzenia na podstawie wariancji 99.9%.
        """
        for d in range(model.spatial_dim):
            W = model.factors[d] # (R_d, K+1)
            if W.shape[0] <= 1:
                continue
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            total_var = np.sum(S ** 2)
            if total_var < 1e-12:
                continue
            cum_var = np.cumsum(S ** 2) / total_var
            new_rank = int(np.searchsorted(cum_var, self.variance_threshold)) + 1
            new_rank = max(1, min(new_rank, W.shape[0]))
            
            if new_rank < W.shape[0]:
                model.factors[d] = W[:new_rank, :]
                model.ranks[d] = new_rank
                # Truncate core along mode d
                if model.spatial_dim == 2:
                    if d == 0:
                        model.core = model.core[:new_rank, :]
                    else:
                        model.core = model.core[:, :new_rank]
