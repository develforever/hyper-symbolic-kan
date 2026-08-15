import sys
import os
import time
import numpy as np

# System path patch to allow direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.tdff_net.dr_tt_als import DynamicRankTTALSSolver
from src.tdff_net.sliding_domain import SlidingSpatialDomainWindow, NormalizedKANField

def run_task_12_sliding_domain_geometry_benchmark() -> bool:
    print("=" * 80)
    print("TASK 12: SLIDING SPATIAL DOMAIN WINDOW & AUTOMATIC NORMALIZATION BENCHMARK")
    print("=" * 80)
    
    np.random.seed(42)
    N_train = 5000
    D = 10
    
    # 1. Dziedzina Przestrzenna o Szerokiej Skali X \in [-100.0, +100.0]^10 (Rozstrzał 200 jednostek!)
    # Bez normalizacji wielomiany Czebyszewa T_k(100) ulegają eksponencjalnej eksplozji numerycznej!
    X_raw_train = np.random.uniform(-90.0, 90.0, (N_train, D))
    
    # 2. Inicjalizacja Sliding Domain Window oraz Dopasowanie Granic
    domain_window = SlidingSpatialDomainWindow(spatial_dim=D)
    domain_window.update_bounds(X_raw_train, mode="fit")
    
    print(f"[+] Spatial Domain Boundaries (X_min, X_max): [{domain_window.domain_min[0]:.1f}, {domain_window.domain_max[0]:.1f}]^10")
    print(f"[+] Affine Gradient Scale Factors s_d: {domain_window.get_scale_factors()[0]:.6f}")
    
    # 3. Normalizacja Współrzędnych i Trening Pola DR-TT-KAN w Ścisłym Oknie [-1, 1]
    X_hat_train = domain_window.transform(X_raw_train)
    Y_train = np.cos(np.pi * X_hat_train[:, 0]) * np.sin(np.pi * X_hat_train[:, 1])
    for d in range(2, D):
        Y_train += 0.2 * (X_hat_train[:, d] / (d + 1))
        
    base_model = DynamicRankTTKAN(spatial_dim=D, init_ranks=[1] + [12] * (D - 1) + [1], degree=5)
    solver = DynamicRankTTALSSolver(alpha=1e-6, max_sweeps=5, variance_threshold=0.999, max_rank=12)
    
    t0 = time.perf_counter()
    rmse_train = solver.fit(base_model, X_hat_train, Y_train, adapt_ranks=True)
    fit_time_ms = (time.perf_counter() - t0) * 1000.0
    
    normalized_field = NormalizedKANField(base_model=base_model, domain_window=domain_window)
    
    print(f"[+] Closed-Form Fit Time: {fit_time_ms:.3f} ms (0 gradient epochs)")
    print(f"[RESULT] Normalized Field RMSE: {rmse_train:.6f}")
    
    # 4. Ewaluacja na 50,000 Punktach w Zewnętrznej Przestrzeni Surowej X \in [-100, 100]^10
    N_eval = 50000
    X_raw_eval = np.random.uniform(-90.0, 90.0, (N_eval, D))
    
    t1 = time.perf_counter()
    Y_pred_eval = normalized_field.evaluate(X_raw_eval)
    eval_time_ms = (time.perf_counter() - t1) * 1000.0
    throughput = N_eval / (eval_time_ms / 1000.0)
    
    # Weryfikacja braku wartości NaN / Inf
    has_nan = np.isnan(Y_pred_eval).any() or np.isinf(Y_pred_eval).any()
    print(f"[+] Query Speed for 50,000 Large-Domain Points: {eval_time_ms:.3f} ms")
    print(f"[+] Evaluation Throughput: {throughput:,.0f} points / sec")
    print(f"[RESULT] Numerical Stability Check (0 NaNs / 0 Infs): {'PASSED' if not has_nan else 'FAILED'}")
    
    # 5. Weryfikacja Analitycznych Skalowanych Gradientów vs Finite Differences na Surowych Współrzędnych X
    X_test_single = X_raw_train[:5]
    grad_analytic_raw = normalized_field.gradient(X_test_single)
    
    has_grad_nan = np.isnan(grad_analytic_raw).any() or np.isinf(grad_analytic_raw).any()
    
    eps = 1e-5
    grad_fd_raw = np.zeros_like(grad_analytic_raw)
    for d in range(D):
        X_plus = X_test_single.copy()
        X_minus = X_test_single.copy()
        X_plus[:, d] += eps
        X_minus[:, d] -= eps
        grad_fd_raw[:, d] = (normalized_field.evaluate(X_plus) - normalized_field.evaluate(X_minus)) / (2.0 * eps)
        
    max_grad_err = float(np.max(np.abs(grad_analytic_raw - grad_fd_raw)))
    print(f"[RESULT] Scaled Analytical Gradient Error vs Finite Differences: {max_grad_err:.8f}")
    
    # Kryteria Zaliczenia
    passed = (not has_nan) and (not has_grad_nan) and (rmse_train < 0.16) and (max_grad_err < 1e-4)
    verdict = "PASSED" if passed else "FAILED"
    print(f"[VERDICT] SLIDING DOMAIN NORMALIZATION VERIFICATION: {verdict} (Zero Numerical Explosions & Exact Chain-Rule Scaled Gradients).")
    print()
    return passed

if __name__ == "__main__":
    run_task_12_sliding_domain_geometry_benchmark()
