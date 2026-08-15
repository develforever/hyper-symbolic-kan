import sys
import os
import time
import numpy as np

# System path patch to allow direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.tdff_net.dr_tt_als import DynamicRankTTALSSolver

def run_dynamic_rank_tt_geometry_benchmark() -> bool:
    print("=" * 80)
    print("TASK 11: DYNAMIC RANK-ADAPTIVE TENSOR TRAIN KAN (DR-TT-KAN) BENCHMARK")
    print("=" * 80)
    
    np.random.seed(42)
    N_train = 5000
    D = 10
    
    # 1. Syntetyczne 10D Pole Ciągłe: f(X) = cos(pi*X_0)*sin(pi*X_1) + 0.2 * sum_{d=2}^9 (X_d / (d+1))
    X_train = np.random.uniform(-0.9, 0.9, (N_train, D))
    Y_train = np.cos(np.pi * X_train[:, 0]) * np.sin(np.pi * X_train[:, 1])
    for d in range(2, D):
        Y_train += 0.2 * (X_train[:, d] / (d + 1))
        
    # 2. Inicjalizacja Prze-paramentryzowanego Modelu DR-TT-KAN z wysokimi rangami R_in = 12
    init_ranks = [1] + [12] * (D - 1) + [1]
    model = DynamicRankTTKAN(spatial_dim=D, init_ranks=init_ranks, degree=5)
    
    init_num_params = sum(r_p * (model.degree + 1) * r_n for r_p, r_n in zip(model.ranks[:-1], model.ranks[1:]))
    print(f"[+] Initial TT Ranks (Over-parameterized): {init_ranks}")
    print(f"[+] Initial Parameter Count: {init_num_params:,}")
    
    # 3. Dopasowanie ALS + Adaptacja Rang SVD (0 epok gradientowych)
    solver = DynamicRankTTALSSolver(alpha=1e-5, max_sweeps=6, variance_threshold=0.999, max_rank=16)
    
    t0 = time.perf_counter()
    final_rmse = solver.fit(model, X_train, Y_train, adapt_ranks=True)
    fit_time_ms = (time.perf_counter() - t0) * 1000.0
    
    final_ranks = list(model.ranks)
    final_num_params = sum(r_p * (model.degree + 1) * r_n for r_p, r_n in zip(model.ranks[:-1], model.ranks[1:]))
    compression_ratio = init_num_params / max(1, final_num_params)
    
    print(f"[+] Closed-Form SVD-ALS Fit Time: {fit_time_ms:.3f} ms (0 gradient epochs)")
    print(f"[+] Truncated Final TT Ranks: {final_ranks}")
    print(f"[+] Compressed Parameter Count: {final_num_params:,}")
    print(f"[+] Parameter Compression Ratio: {compression_ratio:.2f}x reduction")
    print(f"[RESULT] 10D Field Reconstruction RMSE: {final_rmse:.6f}")
    
    # 4. Ewaluacja Przepustowości dla 50,000 Punktów w 10D
    N_eval = 50000
    X_eval = np.random.uniform(-0.9, 0.9, (N_eval, D))
    
    t1 = time.perf_counter()
    Y_eval = model.evaluate(X_eval)
    eval_time_ms = (time.perf_counter() - t1) * 1000.0
    throughput = N_eval / (eval_time_ms / 1000.0)
    
    print(f"[+] Query Speed for 50,000 Points in 10D: {eval_time_ms:.3f} ms")
    print(f"[+] 10D Evaluation Throughput: {throughput:,.0f} points / sec")
    
    # 5. Weryfikacja Analitycznych Gradientów 10D vs Finite Differences
    X_test_single = X_train[:5]
    grad_analytic = model.gradient(X_test_single)
    
    eps = 1e-6
    grad_fd = np.zeros_like(grad_analytic)
    for d in range(D):
        X_plus = X_test_single.copy()
        X_minus = X_test_single.copy()
        X_plus[:, d] += eps
        X_minus[:, d] -= eps
        grad_fd[:, d] = (model.evaluate(X_plus) - model.evaluate(X_minus)) / (2.0 * eps)
        
    max_grad_err = float(np.max(np.abs(grad_analytic - grad_fd)))
    print(f"[RESULT] 10D Analytical Gradient Error vs Finite Differences: {max_grad_err:.8f}")
    
    # Kryteria Zaliczenia
    passed = (final_rmse < 0.15) and (max_grad_err < 1e-4) and (final_num_params < init_num_params)
    verdict = "PASSED" if passed else "FAILED"
    print(f"[VERDICT] DYNAMIC RANK-ADAPTIVE TT-KAN VERIFICATION: {verdict} (Dynamic SVD Rank Adaptation & Zero Gradient Epochs).")
    print()
    return passed

if __name__ == "__main__":
    run_dynamic_rank_tt_geometry_benchmark()
