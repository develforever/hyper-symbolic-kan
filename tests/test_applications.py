r"""
Unit and Integration Tests for Stage E: Robotics CBF Planner & Mesh-Free PDE Solvers.
"""

import pytest
import numpy as np
import sympy as sp
import time

from src.applications.robotics_cbf_planner import (
    CBFConfig,
    ContinuousKANObstacleField,
    SyntheticSphereObstacle,
    CBFPlanner,
    DomainBoxCBF,
    InterAgentCBF
)
from src.applications.pde_poisson_solver import (
    chebyshev_derivatives_2nd,
    SpectralKANPoissonSolver,
    TTPoissonSolver,
    PoissonAnalyticalSolution
)
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver


def test_chebyshev_second_derivative_analytical_exactness():
    """
    Test weryfikujący analityczną rekurencję dla wielomianów Czebyszewa T_k,
    1. pochodnych T'_k oraz 2. pochodnych T''_k względem ścisłych obliczeń symbolicznych SymPy.
    """
    degree = 8
    x_sym = sp.Symbol('x', real=True)
    
    # Punkty testowe w przedziale [-1, 1]
    x_pts = np.linspace(-0.95, 0.95, 50)
    T_arr, dT_arr, d2T_arr = chebyshev_derivatives_2nd(x_pts, degree)
    
    for k in range(degree + 1):
        T_sym = sp.chebyshevt(k, x_sym)
        dT_sym = sp.diff(T_sym, x_sym)
        d2T_sym = sp.diff(dT_sym, x_sym)
        
        T_sym_fn = sp.lambdify(x_sym, T_sym, 'numpy')
        dT_sym_fn = sp.lambdify(x_sym, dT_sym, 'numpy')
        d2T_sym_fn = sp.lambdify(x_sym, d2T_sym, 'numpy')
        
        T_exact = np.asarray(T_sym_fn(x_pts), dtype=np.float64)
        dT_exact = np.asarray(dT_sym_fn(x_pts), dtype=np.float64)
        d2T_exact = np.asarray(d2T_sym_fn(x_pts), dtype=np.float64)
        
        if T_exact.ndim == 0:
            T_exact = np.full_like(x_pts, float(T_exact))
        if dT_exact.ndim == 0:
            dT_exact = np.full_like(x_pts, float(dT_exact))
        if d2T_exact.ndim == 0:
            d2T_exact = np.full_like(x_pts, float(d2T_exact))
            
        err_T = np.max(np.abs(T_arr[:, k] - T_exact))
        err_dT = np.max(np.abs(dT_arr[:, k] - dT_exact))
        err_d2T = np.max(np.abs(d2T_arr[:, k] - d2T_exact))
        
        assert err_T < 1e-11, f"T_{k} błąd przekracza 1e-11: {err_T}"
        assert err_dT < 1e-10, f"T'_{k} błąd przekracza 1e-10: {err_dT}"
        assert err_d2T < 1e-9, f"T''_{k} błąd przekracza 1e-9: {err_d2T}"


def test_cbf_kinematic_single_agent_3d():
    r"""
    Test planera CBF dla kinematyki 1. rzędu (\dot{p} = u):
    Dron 3D przelatuje z (-0.7, -0.7, 0.0) do (0.7, 0.7, 0.0) omijając
    przeszkodę opartą na ciągłym polu KAN (TDFFNet) umieszczoną w (0, 0, 0).
    """
    np.random.seed(42)
    config = CBFConfig(alpha=3.0, v_max=1.5, d_safe=0.05)
    planner = CBFPlanner(config)
    
    # 1. Tworzymy pole KAN reprezentujące ciągłą przeszkodę sferyczną
    tdff = TDFFNet(spatial_dim=3, rank=8, degree=5)
    # Dopasowanie ALS pola dystansu do sfery w środku
    als = ClosedFormALSSolver(alpha=1e-5, max_als_iters=3)
    grid_pts = np.random.uniform(-0.9, 0.9, size=(500, 3))
    # SDF sfery o promieniu 0.35 w (0, 0, 0): dist - 0.35
    sdf_vals = np.linalg.norm(grid_pts, axis=1) - 0.35
    als.fit(tdff, grid_pts, sdf_vals)
    
    kan_obstacle = ContinuousKANObstacleField(tdff, threshold=0.0, invert=False, name="kan_central_sphere")
    
    start = np.array([-0.7, -0.7, 0.0])
    goal = np.array([0.7, 0.7, 0.0])
    
    res = planner.simulate_kinematic_trajectory(
        start=start,
        goal=goal,
        obstacles=[kan_obstacle],
        dt=0.01,
        max_steps=350,
        goal_tolerance=0.08
    )
    
    assert res["collision"] is False, f"Wykryto kolizję! Min h: {np.min(res['h_min_history'])}"
    assert res["success"] is True, "Dron nie dotarł do celu w limicie kroków!"


def test_cbf_dynamic_hocbf_drone_flight():
    r"""
    Test dynamicznego planera HOCBF 2. rzędu (\ddot{p} = a):
    Dron z dużą prędkością początkową lecący prosto na przeszkodę wyhamowuje
    i omija barierę bez naruszenia warunku h(x) >= 0.
    """
    config = CBFConfig(alpha_hocbf=(6.0, 4.0), a_max=10.0, v_max=2.0)
    planner = CBFPlanner(config)
    
    obstacle = SyntheticSphereObstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.3)
    
    start_pos = np.array([-0.7, 0.0, 0.0])
    start_vel = np.array([1.5, 0.0, 0.0]) # Prędkość skierowana bezpośrednio na przeszkodę
    goal = np.array([0.7, 0.0, 0.0])
    
    res = planner.simulate_dynamic_trajectory(
        start_pos=start_pos,
        start_vel=start_vel,
        goal=goal,
        obstacles=[obstacle],
        dt=0.01,
        max_steps=400,
        goal_tolerance=0.1
    )
    
    assert res["collision"] is False, f"Wykryto kolizję w teście dynamicznym! Min h: {np.min(res['h_min_history'])}"
    assert res["success"] is True, "Dron dynamiczny nie osiągnął celu!"


def test_cbf_multi_agent_swarm_avoidance():
    """
    Test bezkolizyjnego lotu roju dronów (N=8 agentów) krzyżujących trajektorie
    z centralną przeszkodą KAN i barierami wzajemnymi.
    """
    np.random.seed(123)
    config = CBFConfig(alpha=2.5, v_max=1.2, d_safe=0.06)
    planner = CBFPlanner(config)
    
    N = 8
    radius = 0.65
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    starts = np.zeros((N, 3))
    goals = np.zeros((N, 3))
    for i in range(N):
        starts[i] = [radius * np.cos(angles[i]), radius * np.sin(angles[i]), 0.0]
        goals[i] = [-radius * np.cos(angles[i]), -radius * np.sin(angles[i]), 0.0]
        
    central_obstacle = SyntheticSphereObstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.2)
    
    res = planner.simulate_swarm(
        start_positions=starts,
        goal_positions=goals,
        obstacles=[central_obstacle],
        dt=0.015,
        max_steps=180
    )
    
    assert res["inter_agent_collision"] is False, f"Kolizja między agentami! Min dystans: {res['min_inter_agent_dist']}"
    assert res["obstacle_collision"] is False, f"Kolizja z przeszkodą! Min dystans: {res['min_obstacle_dist']}"


def test_poisson_solver_2d_analytical_benchmark():
    """
    Test bezsiatkowego solvera Poissona w 2D (0 epok gradientowych).
    Porównanie z dokładnym rozwiązaniem analitycznym u*(x, y) = (1-x^2)(1-y^2).
    """
    u_exact, f_rhs, g_bc = PoissonAnalyticalSolution.get_2d_polynomial()
    
    solver = SpectralKANPoissonSolver(spatial_dim=2, degree=6)
    fit_info = solver.fit(
        f_rhs_fn=f_rhs,
        g_bc_fn=g_bc,
        n_interior=1000,
        n_boundary=400,
        alpha_reg=1e-10,
        beta_bc=300.0
    )
    
    # Weryfikacja na gęstej siatce testowej (2500 punktów)
    x_test = np.linspace(-0.95, 0.95, 50)
    y_test = np.linspace(-0.95, 0.95, 50)
    XX, YY = np.meshgrid(x_test, y_test)
    X_eval = np.column_stack([XX.ravel(), YY.ravel()])
    
    rel_l2_err = solver.compute_l2_relative_error(X_eval, u_exact)
    
    print(f"\n[2D Poisson Polynomial] Solve Time: {fit_info['solve_time_ms']:.2f} ms | L2 Relative Error: {rel_l2_err:.4e}")
    assert rel_l2_err < 1e-4, f"Błąd względny L2 zbyt wysoki: {rel_l2_err}"
    assert fit_info["pde_residual_rmse"] < 1e-3, f"Residuum PDE zbyt wysokie: {fit_info['pde_residual_rmse']}"


def test_poisson_solver_3d_zero_epochs():
    """
    Test bezsiatkowego solvera Poissona w 3D (0 epok gradientowych).
    Rozwiązanie równania \nabla^2 u = f dla u*(x, y, z) = (1-x^2)(1-y^2)(1-z^2).
    """
    u_exact, f_rhs, g_bc = PoissonAnalyticalSolution.get_3d_polynomial()
    
    solver = SpectralKANPoissonSolver(spatial_dim=3, degree=5)
    fit_info = solver.fit(
        f_rhs_fn=f_rhs,
        g_bc_fn=g_bc,
        n_interior=1500,
        n_boundary=800,
        alpha_reg=1e-9,
        beta_bc=250.0
    )
    
    np.random.seed(999)
    X_test = np.random.uniform(-0.92, 0.92, size=(2000, 3))
    
    rel_l2_err = solver.compute_l2_relative_error(X_test, u_exact)
    
    print(f"\n[3D Poisson 0-Epochs] Solve Time: {fit_info['solve_time_ms']:.2f} ms | L2 Error: {rel_l2_err:.4e}")
    assert rel_l2_err < 1e-4, f"Błąd L2 w 3D przekracza próg: {rel_l2_err}"


def test_poisson_high_dim_tensor_train():
    """
    Test solvera TT-KAN dla równania Poissona w przestrzeni 4D (D=4) w 0 epokach gradientowych.
    """
    def f_rhs(X: np.ndarray) -> np.ndarray:
        return -4.0 * (np.pi ** 2) * np.sum(np.cos(np.pi * X), axis=1)
        
    def g_bc(X: np.ndarray) -> np.ndarray:
        return np.sum(np.cos(np.pi * X), axis=1)
        
    tt_solver = TTPoissonSolver(spatial_dim=4, ranks=[1, 4, 4, 4, 1], degree=4)
    
    t0 = time.perf_counter()
    rmse_res = tt_solver.fit_als(
        f_rhs_fn=f_rhs,
        g_bc_fn=g_bc,
        n_interior=400,
        n_boundary=200,
        max_sweeps=2,
        alpha_reg=1e-5,
        beta_bc=50.0
    )
    t1 = time.perf_counter()
    solve_time_ms = (t1 - t0) * 1000.0
    
    print(f"\n[4D TT-KAN Poisson] Solve Time: {solve_time_ms:.2f} ms | PDE Residual RMSE: {rmse_res:.4f}")
    assert np.isfinite(rmse_res), "Residuum TT-Poisson zawiera NaN/Inf!"
