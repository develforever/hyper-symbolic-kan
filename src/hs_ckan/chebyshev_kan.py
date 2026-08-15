import numpy as np

class ChebyshevKANBasis:
    """
    Krawędzie Kolmogorov-Arnold Networks (KAN) oparte na wielomianach Czebyszewa T_k(x).
    Pozwalają na analityczne odwzorowanie funkcji nieliniowych bez użycia siatek spline.
    """
    def __init__(self, degree: int = 5):
        self.degree = degree

    def compute_basis(self, X: np.ndarray) -> np.ndarray:
        """
        Przekształca wektor wejściowy X (N, D) do przestrzeni rozszerzonej wielomianami Czebyszewa T_0..T_deg.
        Zwraca macierz macierzy bazowych o kształcie (N, D * (degree + 1)).
        """
        # Normalizacja danych do przedziału [-1, 1] dla stabilności wielomianów Czebyszewa
        X_norm = np.clip(X, -1.0, 1.0)
        
        N, D = X_norm.shape
        features = []
        
        for k in range(self.degree + 1):
            if k == 0:
                T_k = np.ones_like(X_norm)
            elif k == 1:
                T_k = X_norm
            else:
                # Rekurencja Czebyszewa: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
                T_k_minus_1 = features[-1]
                T_k_minus_2 = features[-2]
                T_k = 2 * X_norm * T_k_minus_1 - T_k_minus_2
            features.append(T_k)
            
        # Łączymy wszystkie stopnie wielomianów
        phi_X = np.hstack(features)
        return phi_X
