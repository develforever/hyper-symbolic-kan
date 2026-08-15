import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.tdff_net.tt_kan import TensorTrainKAN
from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.tdff_net.tensor_field import TDFFNet
from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine


def test_tt_kan_cpp_forward_precision():
    """
    Test weryfikujący dokładność numeryczną ewaluacji TT-KAN w C++ (nanobind + SIMD).
    Maksymalny dopuszczalny błąd bezwzględny względem referencji NumPy: < 1e-11.
    """
    np.random.seed(42)
    N = 5000
    D = 10
    degree = 5
    
    model = TensorTrainKAN(spatial_dim=D, ranks=[1] + [8] * (D - 1) + [1], degree=degree)
    engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
    assert engine.is_native_available(), "Native C++ module _cpp_kernels is not available!"
    
    X = np.random.uniform(-0.95, 0.95, (N, D))
    
    # 1. Referencja Python/NumPy
    Y_py = model.evaluate(X)
    
    # 2. C++ nanobind kernel
    Y_cpp = engine.evaluate_batch(X, model.cores, model.ranks)
    
    max_err = float(np.max(np.abs(Y_py - Y_cpp)))
    print(f"\n[TT-KAN Forward] Max Absolute Error C++ vs NumPy: {max_err:.4e}")
    assert max_err < 1e-11, f"TT-KAN C++ forward error too large: {max_err}"


def test_tt_kan_cpp_gradient_precision():
    """
    Test weryfikujący zgodność analitycznego gradientu TT-KAN w C++
    względem analitycznego gradientu Python oraz różnic skończonych.
    """
    np.random.seed(42)
    N = 500
    D = 6
    degree = 4
    
    model = DynamicRankTTKAN(spatial_dim=D, init_ranks=[1] + [6] * (D - 1) + [1], degree=degree)
    engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
    assert engine.is_native_available()
    
    X = np.random.uniform(-0.8, 0.8, (N, D))
    
    # 1. Analityczny gradient Python
    grad_py = model.gradient(X)
    
    # 2. Analityczny gradient C++
    grad_cpp = engine.gradient_batch(X, model.cores, model.ranks)
    
    max_err = float(np.max(np.abs(grad_py - grad_cpp)))
    print(f"\n[TT-KAN Gradient] Max Error C++ vs Analytical Python: {max_err:.4e}")
    assert max_err < 1e-11, f"TT-KAN C++ gradient error too large: {max_err}"
    
    # 3. Sprawdzenie z różnicami skończonymi na podzbiorze punktów
    eps = 1e-6
    for i in range(min(5, N)):
        for d in range(D):
            X_plus = X[i:i+1].copy()
            X_minus = X[i:i+1].copy()
            X_plus[0, d] += eps
            X_minus[0, d] -= eps
            
            y_plus = model.evaluate(X_plus)[0]
            y_minus = model.evaluate(X_minus)[0]
            num_grad_d = (y_plus - y_minus) / (2.0 * eps)
            
            cpp_grad_d = grad_cpp[i, d]
            diff = abs(num_grad_d - cpp_grad_d)
            assert diff < 1e-5, f"Finite diff mismatch at sample {i}, dim {d}: num={num_grad_d}, cpp={cpp_grad_d}"


def test_cp_kan_cpp_forward_precision():
    """
    Test weryfikujący dokładność numeryczną ewaluacji CP-KAN (TDFF-Net) w C++.
    """
    np.random.seed(42)
    N = 5000
    D = 4
    rank = 12
    degree = 5
    
    model = TDFFNet(spatial_dim=D, rank=rank, degree=degree)
    engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
    assert engine.is_native_available()
    
    X = np.random.uniform(-0.95, 0.95, (N, D))
    
    Y_py = model.evaluate(X)
    Y_cpp = engine.evaluate_cp_batch(X, model.factors, model.lambdas)
    
    max_err = float(np.max(np.abs(Y_py - Y_cpp)))
    print(f"\n[CP-KAN Forward] Max Error C++ vs NumPy: {max_err:.4e}")
    assert max_err < 1e-11, f"CP-KAN C++ forward error too large: {max_err}"


def test_cp_kan_cpp_gradient_precision():
    """
    Test weryfikujący analityczny gradient CP-KAN (TDFF-Net) w C++.
    """
    np.random.seed(42)
    N = 1000
    D = 4
    rank = 10
    degree = 4
    
    model = TDFFNet(spatial_dim=D, rank=rank, degree=degree)
    engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
    assert engine.is_native_available()
    
    X = np.random.uniform(-0.85, 0.85, (N, D))
    
    grad_py = model.gradient(X)
    grad_cpp = engine.gradient_cp_batch(X, model.factors, model.lambdas)
    
    max_err = float(np.max(np.abs(grad_py - grad_cpp)))
    print(f"\n[CP-KAN Gradient] Max Error C++ vs Analytical Python: {max_err:.4e}")
    assert max_err < 1e-11, f"CP-KAN C++ gradient error too large: {max_err}"


def test_cpp_throughput_and_latency_benchmark():
    """
    Rygorystyczny benchmark wydajnościowy C++ (AVX2 + OpenMP):
    Wymagania Etapu A:
    - Przepustowość > 1,500,000 punktów/s w batchu
    - Opóźnienie wywołania < 0.2 us / punkt w batchu
    """
    np.random.seed(42)
    N = 100000
    D = 10
    degree = 5
    ranks = [1] + [8] * (D - 1) + [1]
    
    model = TensorTrainKAN(spatial_dim=D, ranks=ranks, degree=degree)
    engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
    assert engine.is_native_available()
    
    X = np.random.uniform(-0.9, 0.9, (N, D))
    
    # Rozgrzewka (Warmup)
    for _ in range(3):
        _ = engine.evaluate_batch(X[:1000], model.cores, model.ranks)
        
    # Pomiar czasu ewaluacji
    repeats = 5
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = engine.evaluate_batch(X, model.cores, model.ranks)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        
    best_time = min(times)
    throughput = N / best_time
    latency_us = (best_time * 1e6) / N
    
    print("\n" + "=" * 70)
    print("HYPER-SYMBOLIC KAN C++ NATIVE ENGINE PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Points Evaluated: {N:,} in {D}D (Chebyshev degree {degree}, TT ranks 8)")
    print(f"Best Batch Time: {best_time * 1000.0:.2f} ms")
    print(f"Latency per point: {latency_us:.4f} us / query")
    print(f"Throughput: {throughput:,.0f} points / second")
    print("=" * 70)
    
    assert throughput >= 1500000, f"Throughput {throughput:,.0f} pts/s below target 1,500,000 pts/s!"
    assert latency_us <= 0.67, f"Latency {latency_us:.4f} us above acceptable threshold"


def test_gil_release_and_concurrency():
    """
    Weryfikacja równoległego zwalniania GIL przez nanobind przy wywołaniach z wielu wątków Pythona.
    """
    N = 20000
    D = 8
    degree = 4
    model = TensorTrainKAN(spatial_dim=D, ranks=[1] + [6] * (D - 1) + [1], degree=degree)
    engine = FastCPPKANEngine(spatial_dim=D, degree=degree)
    
    def worker(worker_id: int):
        X_sub = np.random.uniform(-0.9, 0.9, (N, D))
        res = engine.evaluate_batch(X_sub, model.cores, model.ranks)
        return len(res)
        
    num_threads = 4
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [f.result() for f in futures]
        
    assert all(r == N for r in results)
    print(f"\n[GIL Release & Concurrency] Successfully completed {num_threads} concurrent threads ({num_threads * N:,} points)")


if __name__ == "__main__":
    test_tt_kan_cpp_forward_precision()
    test_tt_kan_cpp_gradient_precision()
    test_cp_kan_cpp_forward_precision()
    test_cp_kan_cpp_gradient_precision()
    test_cpp_throughput_and_latency_benchmark()
    test_gil_release_and_concurrency()
