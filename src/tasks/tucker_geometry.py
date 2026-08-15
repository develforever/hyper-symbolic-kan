import time
import numpy as np
from typing import Dict, Tuple

from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.tdff_net.tucker_tensor_field import TuckerTDFFNet
from src.tdff_net.tucker_als import TuckerALSSolver

class TuckerGeometryTask:
    r"""
    Task 7: Benchmark Wielorozdzielczych Pól Tensorowych Tuckera na Geometrii o Wysokiej Częstotliwości.
    
    Testuje pola SDF nie-gładkie z ostrymi krawędziami: f(x, y) = max(|x|, |y|) - 0.5 (Ostry Sześcian).
    Porównuje:
    - Standardowy Rozkład CP (TDFFNet R=16)
    - Rozkład Tuckera (TuckerTDFFNet ranks=[8,8]) z Adaptacyjną Redukcją Rangi SVD.
    """
    def __init__(self, num_train: int = 3000, num_test: int = 20000):
        self.num_train = num_train
        self.num_test = num_test

    def generate_sharp_geometry_dataset(self, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(seed)
        X_train = np.random.uniform(-1.0, 1.0, size=(self.num_train, 2))
        # SDF ostrego kwadratu o boku 1.0 (promień 0.5)
        Y_train = (np.maximum(np.abs(X_train[:, 0]), np.abs(X_train[:, 1])) - 0.5).reshape(-1, 1)

        np.random.seed(seed + 1000)
        X_test = np.random.uniform(-1.0, 1.0, size=(self.num_test, 2))
        Y_test = (np.maximum(np.abs(X_test[:, 0]), np.abs(X_test[:, 1])) - 0.5).reshape(-1, 1)

        return X_train, Y_train, X_test, Y_test

    def run_benchmark(self) -> Dict[str, float]:
        X_train, Y_train, X_test, Y_test = self.generate_sharp_geometry_dataset()
        
        # 1. Standardowy Rozkład CP
        cp_model = TDFFNet(spatial_dim=2, rank=16, degree=6)
        cp_solver = ClosedFormALSSolver(alpha=1e-3, max_als_iters=8)
        
        t0_cp = time.perf_counter()
        cp_solver.fit(cp_model, X_train, Y_train.ravel())
        t_cp_fit_ms = (time.perf_counter() - t0_cp) * 1000.0
        
        Y_pred_cp = cp_model.evaluate(X_test)
        rmse_cp = float(np.sqrt(np.mean((Y_test.ravel() - Y_pred_cp) ** 2)))
        
        # 2. Wielorozdzielczy Rozkład Tuckera (Faza 2)
        tucker_model = TuckerTDFFNet(spatial_dim=2, ranks=[10, 10], degree=6)
        tucker_solver = TuckerALSSolver(alpha=1e-3, max_als_iters=8, variance_threshold=0.9999)
        
        t0_tucker = time.perf_counter()
        tucker_solver.fit(tucker_model, X_train, Y_train.ravel())
        t_tucker_fit_ms = (time.perf_counter() - t0_tucker) * 1000.0
        
        Y_pred_tucker = tucker_model.evaluate(X_test)
        rmse_tucker = float(np.sqrt(np.mean((Y_test.ravel() - Y_pred_tucker) ** 2)))
        
        # 3. Weryfikacja Analitycznego Gradientu dla Tuckera
        test_sub = X_test[:100]
        analytic_grad = tucker_model.gradient(test_sub)
        
        eps = 1e-5
        fd_grad = np.zeros_like(analytic_grad)
        for d in range(2):
            X_plus = test_sub.copy()
            X_minus = test_sub.copy()
            X_plus[:, d] += eps
            X_minus[:, d] -= eps
            fd_grad[:, d] = (tucker_model.evaluate(X_plus) - tucker_model.evaluate(X_minus)) / (2 * eps)
            
        grad_err = float(np.mean(np.abs(analytic_grad - fd_grad)))
        
        return {
            "cp_fit_ms": t_cp_fit_ms,
            "cp_rmse": rmse_cp,
            "tucker_fit_ms": t_tucker_fit_ms,
            "tucker_rmse": rmse_tucker,
            "grad_error": grad_err,
            "final_ranks": tucker_model.ranks
        }
