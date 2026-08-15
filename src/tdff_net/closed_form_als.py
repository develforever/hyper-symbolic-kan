import numpy as np
from src.tdff_net.tensor_field import TDFFNet

class ClosedFormALSSolver:
    """
    Beziteracyjny/Analityczny Solver Alternating Least Squares (ALS) dla TDFF-Net.
    Dopasowuje rozkład tensorowy CP pól funkcjonalnych bez gradientów i bez backpropagation.
    """
    def __init__(self, alpha: float = 1e-4, max_als_iters: int = 10):
        self.alpha = alpha
        self.max_als_iters = max_als_iters

    def fit(self, model: TDFFNet, X: np.ndarray, Y: np.ndarray) -> float:
        """
        X: Punkty próbkujące w przestrzeni (N, D)
        Y: Wartości docelowe pola funkcjonalnego/odległości (N,)
        
        Zwraca końcowy błąd średniokwadratowy MSE.
        """
        N, D = X.shape
        R = model.rank
        K = model.degree
        
        for it in range(self.max_als_iters):
            # 1. Optymalizacja macierzy czynnikowych W^(d) dla każdego wymiaru d
            for d in range(D):
                # Obliczenie iloczynów pozostałych wymiarów j != d
                phi_other = np.ones((N, R))
                for j in range(D):
                    if j != d:
                        T, _ = model._compute_chebyshev_and_derivatives(X[:, j])
                        phi_j = T @ model.factors[j].T # (N, R)
                        phi_other *= phi_j
                        
                # Wyznaczenie bazy Czebyszewa dla aktualnego wymiaru d
                T_d, _ = model._compute_chebyshev_and_derivatives(X[:, d]) # (N, K+1)
                
                # Budowanie macierzy projektowej Phi_d o rozmiarze (N, R * (K + 1))
                Phi_d = np.zeros((N, R * (K + 1)))
                for r in range(R):
                    # Dla każdego komponentu r: scaling = \lambda_r * phi_other[:, r]
                    scale_r = model.lambdas[r] * phi_other[:, r] # (N,)
                    start_col = r * (K + 1)
                    end_col = (r + 1) * (K + 1)
                    Phi_d[:, start_col:end_col] = T_d * scale_r[:, np.newaxis]
                    
                # Rozwiązanie równania grzbietowego: (Phi_d^T Phi_d + alpha I) w = Phi_d^T Y
                A = Phi_d.T @ Phi_d + self.alpha * np.eye(R * (K + 1))
                B = Phi_d.T @ Y
                w_flat = np.linalg.solve(A, B)
                
                # Przypisanie macierzy zaktualizowanych wag z normalizacją normy
                updated_factors = w_flat.reshape(R, K + 1)
                norms = np.linalg.norm(updated_factors, axis=1, keepdims=True) + 1e-12
                model.factors[d] = updated_factors / norms
                model.lambdas = model.lambdas * norms.ravel()

            # 2. Bezpośrednia optymalizacja wektora wag głównych \lambda
            P = np.ones((N, R))
            for j in range(D):
                T, _ = model._compute_chebyshev_and_derivatives(X[:, j])
                P *= (T @ model.factors[j].T)
                
            A_lam = P.T @ P + self.alpha * np.eye(R)
            B_lam = P.T @ Y
            model.lambdas = np.linalg.solve(A_lam, B_lam)

        # Wyznaczenie końcowego MSE
        Y_pred = model.evaluate(X)
        mse = float(np.mean((Y - Y_pred) ** 2))
        return mse
