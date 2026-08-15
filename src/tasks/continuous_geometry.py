import time
import numpy as np
from typing import Dict, Any
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver

class ContinuousGeometryTask:
    """
    Zadanie Ciągłej Reprezentacji Geometrii 3D Bez Siatek i Bez Raymarchingu.
    
    Testuje:
    - Jakość aproksymacji ciągłych pól odległości (SDF / Continuous Geometry Fields).
    - Przepustowość zapytania (Szybkość ewaluacji w ms dla 50 000 punktów jednocześnie).
    - Dokładność analitycznego gradientu przestrzennego \nabla f(x,y,z) vs Różniczkowanie Numeryczne.
    """
    def __init__(self, num_train: int = 5000, num_test: int = 50000):
        self.num_train = num_train
        self.num_test = num_test

    def _target_field_function(self, X: np.ndarray) -> np.ndarray:
        """
        Złożone Ciągłe Pole Geometryczne 3D:
        Kombinacja Signed Distance Field sfery i falowej funkcji przestrzennej.
        SDF: d(x,y,z) = sqrt(x^2 + y^2 + z^2) - 0.65 + 0.1 * sin(3*x) * cos(3*y)
        """
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        sdf = r - 0.65 + 0.1 * np.sin(3.0 * x) * np.cos(3.0 * y)
        return sdf

    def run_benchmark(self, rank: int = 24, degree: int = 6) -> Dict[str, Any]:
        np.random.seed(42)
        
        # Generowanie punktów wewnątrz przestrzeni 3D [-1, 1]^3
        X_train = np.random.uniform(-1.0, 1.0, size=(self.num_train, 3))
        Y_train = self._target_field_function(X_train)
        
        X_test = np.random.uniform(-1.0, 1.0, size=(self.num_test, 3))
        Y_test = self._target_field_function(X_test)
        
        # 1. Inicjalizacja Modelu TDFF-Net
        model = TDFFNet(spatial_dim=3, rank=rank, degree=degree)
        solver = ClosedFormALSSolver(alpha=1e-5, max_als_iters=8)
        
        # 2. Dopasowanie Pola w Czasie Zamkniętym (Bez Backpropagation)
        t_fit_start = time.perf_counter()
        final_train_mse = solver.fit(model, X_train, Y_train)
        t_fit_ms = (time.perf_counter() - t_fit_start) * 1000.0
        
        # 3. Ewaluacja Przepustowości i Dokładności dla 50 000 Punktów (Zamiast Raymarchingu)
        t_eval_start = time.perf_counter()
        Y_pred = model.evaluate(X_test)
        t_eval_ms = (time.perf_counter() - t_eval_start) * 1000.0
        
        rmse = float(np.sqrt(np.mean((Y_test - Y_pred) ** 2)))
        mae = float(np.mean(np.abs(Y_test - Y_pred)))
        
        # 4. Weryfikacja Analitycznego Gradientu \nabla f(X) vs Finite Differences
        X_grad_sample = X_test[:1000]
        grad_analytic = model.gradient(X_grad_sample)
        
        # Numerical gradient computation (Finite Differences)
        eps = 1e-5
        grad_num = np.zeros_like(grad_analytic)
        for d in range(3):
            X_plus = X_grad_sample.copy()
            X_minus = X_grad_sample.copy()
            X_plus[:, d] += eps
            X_minus[:, d] -= eps
            grad_num[:, d] = (model.evaluate(X_plus) - model.evaluate(X_minus)) / (2.0 * eps)
            
        grad_error = float(np.mean(np.abs(grad_analytic - grad_num)))
        
        return {
            "fit_time_ms": t_fit_ms,
            "eval_time_ms": t_eval_ms,
            "num_test_points": self.num_test,
            "points_per_sec": (self.num_test / (t_eval_ms / 1000.0)),
            "rmse": rmse,
            "mae": mae,
            "grad_error": grad_error
        }
