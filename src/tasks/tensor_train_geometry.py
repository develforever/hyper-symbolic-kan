import time
import numpy as np
from typing import Dict, Tuple

from src.tdff_net.tt_kan import TensorTrainKAN, TTALSSolver

class TensorTrainGeometryTask:
    r"""
    TASK 9: Benchmark Tensor Train KAN (TT-KAN) w Wielowymiarowej Przestrzeni D=10 (Faza 6).
    
    Aproksymuje hipersferyczne pole SDF f(x) = ||x||_2 - 0.5 w wymiarze D = 10.
    Mierzy:
    - Czas ALS w wymiarze D = 10 (0 epok gradientowych).
    - Przepustowość ewaluacji dla 50,000 punktów 10D.
    - Zgodność analitycznego gradientu 10D z różniczkowaniem skończonym.
    """
    def __init__(self, spatial_dim: int = 10, num_train: int = 4000, num_test: int = 50000):
        self.spatial_dim = spatial_dim
        self.num_train = num_train
        self.num_test = num_test

    def generate_hypersphere_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(42)
        # Próbkowanie w kostce [-0.8, 0.8]^D
        X_train = np.random.uniform(-0.8, 0.8, size=(self.num_train, self.spatial_dim))
        Y_train = np.linalg.norm(X_train, axis=1) - 0.5
        
        X_test = np.random.uniform(-0.8, 0.8, size=(self.num_test, self.spatial_dim))
        Y_test = np.linalg.norm(X_test, axis=1) - 0.5
        
        return X_train, Y_train, X_test, Y_test

    def run_benchmark(self, ranks: list = None, degree: int = 5) -> Dict[str, float]:
        X_train, Y_train, X_test, Y_test = self.generate_hypersphere_dataset()
        
        if ranks is None:
            # TT rangi r_0=1, r_1..r_9=8, r_10=1
            ranks = [1] + [8] * (self.spatial_dim - 1) + [1]
            
        model = TensorTrainKAN(spatial_dim=self.spatial_dim, ranks=ranks, degree=degree)
        solver = TTALSSolver(alpha=1e-4, max_sweeps=4)
        
        # 1. ALS Fit w wymiarze D=10
        t0_fit = time.perf_counter()
        rmse_train = solver.fit(model, X_train, Y_train)
        t_fit_ms = (time.perf_counter() - t0_fit) * 1000.0
        
        # 2. Ewaluacja wydajności dla 50,000 punktów 10D
        t0_eval = time.perf_counter()
        Y_pred = model.evaluate(X_test)
        t_eval_ms = (time.perf_counter() - t0_eval) * 1000.0
        
        rmse_test = float(np.sqrt(np.mean((Y_test - Y_pred) ** 2)))
        points_per_sec = (self.num_test / (t_eval_ms / 1000.0))
        
        # 3. Weryfikacja Analitycznego Gradientu 10D vs Różniczkowanie Skończone
        num_grad_samples = 200
        X_grad_test = X_test[:num_grad_samples]
        grad_analytical = model.gradient(X_grad_test) # (200, 10)
        
        eps = 1e-5
        grad_fd = np.zeros_like(grad_analytical)
        for d in range(self.spatial_dim):
            X_plus = X_grad_test.copy()
            X_minus = X_grad_test.copy()
            X_plus[:, d] += eps
            X_minus[:, d] -= eps
            grad_fd[:, d] = (model.evaluate(X_plus) - model.evaluate(X_minus)) / (2.0 * eps)
            
        max_grad_error = float(np.max(np.abs(grad_analytical - grad_fd)))
        
        return {
            "spatial_dim": float(self.spatial_dim),
            "num_test_points": float(self.num_test),
            "fit_time_ms": t_fit_ms,
            "eval_time_ms": t_eval_ms,
            "points_per_sec": points_per_sec,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "max_grad_error": max_grad_error
        }
