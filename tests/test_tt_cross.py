import time
import numpy as np
from typing import Callable

from src.tdff_net.tt_cross import maxvol, TTCrossSolver
from src.tdff_net.dmrg_kan import DMRGTTKANSolver
from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine


def test_maxvol_submatrix_selection():
    r"""
    Weryfikacja algorytmu MaxVol:
    1. Wybór r wierszy o maksymalnym wyznaczniku podmacierzy.
    2. Właściwość Z[I, :] == I_r oraz max |Z_{i, j}| <= tol.
    """
    np.random.seed(42)
    N = 100
    r = 6
    tol = 1.05

    A = np.random.randn(N, r)
    I, Z = maxvol(A, tol=tol)

    assert len(I) == r, f"Expected {r} row indices, got {len(I)}"
    assert len(np.unique(I)) == r, "Row indices must be unique"
    
    # Sprawdzenie Z[I, :] == I_r
    np.testing.assert_allclose(Z[I, :], np.eye(r), atol=1e-5)
    
    # Sprawdzenie ograniczenia modułu
    max_val = np.max(np.abs(Z))
    assert max_val <= tol + 1e-4, f"MaxVol bound violated: max |Z| = {max_val} > {tol}"
    
    # Sprawdzenie czy wyznacznik A[I, :] jest większy niż dla losowych podzbiorów
    det_opt = np.abs(np.linalg.det(A[I, :]))
    for _ in range(20):
        rand_idx = np.random.choice(N, size=r, replace=False)
        det_rand = np.abs(np.linalg.det(A[rand_idx, :]))
        assert det_opt >= det_rand * 0.5, "MaxVol submatrix determinant should dominate random selection"


def test_tt_cross_reconstruction_20d():
    r"""
    Weryfikacja próbkowania bezsiatkowego TT-Cross w przestrzeni D = 20:
    Dopasowanie funkcji nieliniowej przy budżecie próbek O(D * R^2 * K) zamiast O(K^D).
    """
    D = 20
    degree = 4
    
    def target_func(X: np.ndarray) -> np.ndarray:
        # Sprzężenie parzyste między kolejnymi wymiarami
        val = 0.0
        for d in range(X.shape[1] - 1):
            val += np.sin(np.pi * X[:, d]) * np.cos(np.pi * X[:, d + 1])
        return val

    solver = TTCrossSolver(max_rank=6, eps=1e-4, max_sweeps=2, seed=42)
    
    t0 = time.perf_counter()
    model = solver.fit_function(target_func, spatial_dim=D, degree=degree)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"\n[TT-Cross 20D] Dopasowano w {elapsed_ms:.2f} ms przy {solver.sample_count} próbkach")
    print(f"[TT-Cross 20D] Osiągnięte rangi TT: {model.ranks}")
    
    # Weryfikacja budżetu próbek O(D * R^2 * K)
    max_allowed_samples = D * (solver.max_rank ** 2) * (degree + 1) * 3
    assert solver.sample_count <= max_allowed_samples, f"Sample count {solver.sample_count} exceeded budget {max_allowed_samples}"
    assert elapsed_ms < 1000.0, f"Fitting time {elapsed_ms:.2f} ms exceeded 1.0s limit"
    
    # Weryfikacja dokładności na zbiorze walidacyjnym
    np.random.seed(123)
    X_val = np.random.uniform(-0.8, 0.8, (200, D))
    Y_true = target_func(X_val)
    Y_pred = model.evaluate(X_val)
    
    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))
    rel_err = rmse / (np.std(Y_true) + 1e-8)
    print(f"[TT-Cross 20D] RMSE: {rmse:.4f} | Błąd względny: {rel_err:.4f}")
    assert rel_err < 0.25, f"Relative error {rel_err:.4f} too high for TT-Cross 20D"


def test_tt_cross_high_dimensional_50d():
    r"""
    Weryfikacja skalowalności algorytmu TT-Cross w przestrzeni D = 50.
    """
    D = 50
    degree = 4
    
    def field_50d(X: np.ndarray) -> np.ndarray:
        val = 0.0
        for d in range(X.shape[1] - 1):
            val += np.sin(np.pi * X[:, d]) * np.cos(np.pi * X[:, d + 1])
        return val

    solver = TTCrossSolver(max_rank=4, eps=1e-4, max_sweeps=2, seed=42)
    
    t0 = time.perf_counter()
    model = solver.fit_function(field_50d, spatial_dim=D, degree=degree)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"\n[TT-Cross 50D] Dopasowano w {elapsed_ms:.2f} ms przy {solver.sample_count} próbkach")
    
    assert elapsed_ms < 1000.0, f"Fitting time {elapsed_ms:.2f} ms exceeded 1.0s limit"
    assert len(model.cores) == D
    assert model.ranks[0] == 1 and model.ranks[-1] == 1
    
    # Ewaluacja na 100 punktach testowych
    X_test = np.random.uniform(-0.75, 0.75, (100, D))
    Y_pred = model.evaluate(X_test)
    assert np.all(np.isfinite(Y_pred)), "Model predictions contain NaN or Inf"


def test_dmrg_2site_rank_adaptation():
    r"""
    Weryfikacja 2-Site DMRG z dynamiczną adaptacją rang i ucinaniem SVD na wiązaniach.
    """
    np.random.seed(42)
    N = 3000
    D = 4
    degree = 4
    
    X = np.random.uniform(-0.8, 0.8, (N, D))
    # Nieliniowa funkcja sprzężona
    Y = np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + np.sin(np.pi * X[:, 2]) * np.cos(np.pi * X[:, 3])
    
    model = DynamicRankTTKAN(spatial_dim=D, init_ranks=[1, 4, 4, 4, 1], degree=degree)
    solver = DMRGTTKANSolver(alpha=1e-6, max_sweeps=5, variance_threshold=0.9999, max_rank=8)
    
    t0 = time.perf_counter()
    rmse = solver.fit(model, X, Y)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"\n[2-Site DMRG] RMSE: {rmse:.6f} w {elapsed_ms:.2f} ms | Rangi: {model.ranks}")
    
    assert rmse < 0.06, f"2-Site DMRG RMSE {rmse:.6f} exceeds tolerance 0.06"
    assert model.ranks[0] == 1 and model.ranks[-1] == 1


def test_cpp_kernels_stage_b_precision():
    r"""
    Weryfikacja dokładności numerycznej operacji C++ z Etapu B:
    1. project_chebyshev_modal (błąd < 1e-12)
    2. build_dmrg_normal_equations (błąd < 1e-11)
    """
    np.random.seed(42)
    engine = FastCPPKANEngine(spatial_dim=3, degree=3)
    assert engine.is_native_available(), "Native C++ kernels module must be available"
    
    # 1. Test projekcji modalnej Czebyszewa
    r_p, K1, r_n = 4, 5, 4
    nodal = np.random.randn(r_p, K1, r_n)
    V_inv = np.random.randn(K1, K1)
    
    modal_cpp = engine.project_chebyshev_modal(nodal, V_inv)
    modal_py = np.einsum('ki, ris -> rks', V_inv, nodal)
    
    err_proj = float(np.max(np.abs(modal_cpp - modal_py)))
    print(f"\n[C++ Modal Projection] Max Absolute Error: {err_proj:.4e}")
    assert err_proj < 1e-12, f"Modal projection error too large: {err_proj}"
    
    # 2. Test akumulacji układu normalnego DMRG
    N = 200
    L_prev = np.random.randn(N, 3)
    T_d = np.random.randn(N, 4)
    T_d1 = np.random.randn(N, 4)
    R_next = np.random.randn(N, 3)
    Y = np.random.randn(N)
    alpha = 1e-5
    
    A_cpp, B_cpp = engine.build_dmrg_normal_equations(L_prev, T_d, T_d1, R_next, Y, alpha=alpha)
    
    # Referencja NumPy
    P = 3 * 4 * 4 * 3
    mid = (T_d[:, :, None] * T_d1[:, None, :]).reshape(N, 16)
    Phi = ((L_prev[:, :, None] * mid[:, None, :]).reshape(N, 48)[:, :, None] * R_next[:, None, :]).reshape(N, P)
    A_py = Phi.T @ Phi + alpha * np.eye(P)
    B_py = Phi.T @ Y
    
    err_A = float(np.max(np.abs(A_cpp - A_py)))
    err_B = float(np.max(np.abs(B_cpp - B_py)))
    print(f"[C++ DMRG Normal Equations] Max Error A: {err_A:.4e} | B: {err_B:.4e}")
    assert err_A < 1e-11, f"DMRG Matrix A error too large: {err_A}"
    assert err_B < 1e-11, f"DMRG Vector B error too large: {err_B}"


if __name__ == "__main__":
    test_maxvol_submatrix_selection()
    test_tt_cross_reconstruction_20d()
    test_tt_cross_high_dimensional_50d()
    test_dmrg_2site_rank_adaptation()
    test_cpp_kernels_stage_b_precision()
