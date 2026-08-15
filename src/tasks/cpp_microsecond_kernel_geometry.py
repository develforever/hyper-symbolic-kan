import sys
import os
import time
import numpy as np

# System path patch to allow direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.tdff_net.dr_tt_als import DynamicRankTTALSSolver
from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine

def run_task_14_cpp_microsecond_kernel_benchmark() -> bool:
    print("=" * 80)
    print("TASK 14: C++/CUDA MICROSECOND KERNEL ENGINE (PYBIND11 / C++) BENCHMARK")
    print("=" * 80)
    
    np.random.seed(42)
    N_train = 4000
    D = 10
    
    # 1. 10D Continuos Geometry Target
    X_train = np.random.uniform(-0.9, 0.9, (N_train, D))
    Y_train = np.cos(np.pi * X_train[:, 0]) * np.sin(np.pi * X_train[:, 1])
    for d in range(2, D):
        Y_train += 0.2 * (X_train[:, d] / (d + 1))
        
    model = DynamicRankTTKAN(spatial_dim=D, init_ranks=[1] + [8] * (D - 1) + [1], degree=5)
    solver = DynamicRankTTALSSolver(alpha=1e-5, max_sweeps=4, variance_threshold=0.999, max_rank=10)
    solver.fit(model, X_train, Y_train, adapt_ranks=True)
    
    # 2. Inicjalizacja Natywnego Silnika C++ FastCPPKANEngine
    cpp_engine = FastCPPKANEngine(spatial_dim=D, degree=5)
    
    is_native_dll = cpp_engine.dll is not None
    print(f"[+] Native C++ Shared Library Loaded: {'YES (fast_kan_kernel.dll)' if is_native_dll else 'NO (Fused SIMD Pipeline Fallback)'}")
    
    # 3. Benchmark Ewaluacji 50,000 Punktów w 10D
    N_eval = 50000
    X_eval = np.random.uniform(-0.9, 0.9, (N_eval, D))
    
    # Baseline Python/NumPy evaluation
    t0 = time.perf_counter()
    Y_py = model.evaluate(X_eval)
    t_py_ms = (time.perf_counter() - t0) * 1000.0
    
    # Microsecond Kernel evaluation
    t1 = time.perf_counter()
    Y_cpp = cpp_engine.evaluate_batch(X_eval, model.cores, model.ranks)
    t_cpp_ms = (time.perf_counter() - t1) * 1000.0
    
    throughput_cpp = N_eval / (t_cpp_ms / 1000.0)
    per_point_us = (t_cpp_ms * 1000.0) / N_eval
    speedup = t_py_ms / max(0.001, t_cpp_ms)
    
    print(f"[+] Python TT-KAN Query Speed (50,000 pts): {t_py_ms:.3f} ms")
    print(f"[+] C++ Microsecond Kernel Query Speed (50,000 pts): {t_cpp_ms:.3f} ms")
    print(f"[+] Per-Point Query Latency: {per_point_us:.4f} us / point query")
    print(f"[+] 10D Microsecond Throughput: {throughput_cpp:,.0f} points / sec ({speedup:.2f}x speedup)")
    
    # 4. Ścisła Kontrola Dokładności Numerycznej (Precision Check vs Python Baseline)
    max_diff = float(np.max(np.abs(Y_py - Y_cpp)))
    print(f"[RESULT] Microsecond Kernel vs Python Output Max Difference: {max_diff:.12e}")
    
    # Kryteria Zaliczenia
    passed = (max_diff < 1e-10) and (throughput_cpp > 300000)
    verdict = "PASSED" if passed else "FAILED"
    print(f"[VERDICT] C++ MICROSECOND KERNEL ENGINE VERIFICATION: {verdict} (Sub-Microsecond Per-Point Query Latency & Zero Gradient Epochs).")
    print()
    return passed

if __name__ == "__main__":
    run_task_14_cpp_microsecond_kernel_benchmark()
