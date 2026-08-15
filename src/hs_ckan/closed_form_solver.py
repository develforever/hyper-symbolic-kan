import numpy as np

class ClosedFormLayerSolver:
    """
    Analityczny Solver Warstwy w Czasie Zamkniętym (Beziteracyjny).
    Oblicza optymalne wagi W* metodą regresji grzbietowej / SVD w jednym kroku:
    W* = (Phi^T Phi + alpha I)^(-1) Phi^T Y
    """
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.W = None

    def fit(self, Phi: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Phi: Macierz cech bazowych (N, K)
        Y: Macierz docelowych wektorów stanów (N, M)
        """
        N, K = Phi.shape
        # Regularyzowana macierz kowariancji (K, K)
        A = Phi.T @ Phi + self.alpha * np.eye(K)
        B = Phi.T @ Y
        
        # Wyznaczenie wag metodą bezpośredniego rozwiązania układu równań liniowych
        self.W = np.linalg.solve(A, B)
        return self.W

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Solver nie został jeszcze skompilowany/przetrenowany.")
        return Phi @ self.W
