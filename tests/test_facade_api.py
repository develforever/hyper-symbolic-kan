"""
Unit tests for the High-Level Facade API (import hyper_kan as hk).
"""

import os
import sys
import tempfile
import pathlib
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hyper_kan as hk

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def test_facade_tensor_field_fit_predict_gradient():
    """Tests hk.TensorField end-to-end (ALS fit, forward, gradient, save/load)."""
    np.random.seed(42)
    N = 150
    X = np.random.uniform(-0.8, 0.8, size=(N, 3))
    # Non-linear 3D target function
    y = np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + 0.5 * X[:, 2] ** 2

    # Initialize model via facade
    model = hk.TensorField(spatial_dim=3, rank=12, degree=5)
    model.fit(X, y, alpha=1e-4, max_iters=10)

    # Prediction & call syntax
    y_pred = model.predict(X)
    y_call = model(X)
    np.testing.assert_allclose(y_pred, y_call)

    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    assert rmse < 0.05, f"High-level TensorField ALS did not reach expected accuracy: RMSE={rmse}"

    # Gradient computation
    grad = model.gradient(X[:5])
    assert grad.shape == (5, 3)
    assert not np.isnan(grad).any()

    # Serialization roundtrip
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = pathlib.Path(tmp_dir) / "field_model.json"
        model.save(json_path)
        assert json_path.exists()

        loaded_model = hk.TensorField.load(json_path)
        y_loaded = loaded_model(X)
        np.testing.assert_allclose(y_pred, y_loaded, rtol=1e-12, atol=1e-12)


def test_facade_tensor_field_torch_jax_bridges():
    """Tests converting hk.TensorField to PyTorch and JAX layers."""
    field = hk.TensorField(spatial_dim=2, rank=8, degree=3)

    # Convert to PyTorch
    torch_layer = field.to_torch()
    assert torch_layer.in_features == 2
    assert torch_layer.rank == 8
    assert torch_layer.degree == 3

    # Convert to JAX
    if not _HAS_JAX:
        pytest.skip("JAX not installed")
    jax_params, jax_layer = field.to_jax()
    assert jax_layer.in_features == 2
    assert "lambdas" in jax_params
    assert "factors" in jax_params


def test_facade_tensor_train_field_fit_cross():
    """Tests hk.TensorTrainField fitting a 10D continuous function via TT-Cross."""
    def target_10d(x):
        # Continuous anisotropic 10D function
        return np.sum(x[:, :5] ** 2, axis=1) + np.prod(np.cos(x[:, 5:]), axis=1)

    field = hk.TensorTrainField(spatial_dim=10, degree=4)
    field.fit_cross(target_10d, max_rank=8, eps=1e-2)

    X_test = np.random.uniform(-0.6, 0.6, size=(20, 10))
    y_exact = target_10d(X_test)
    y_pred = field(X_test)

    rel_err = np.linalg.norm(y_exact - y_pred) / (np.linalg.norm(y_exact) + 1e-12)
    assert rel_err < 0.25, f"TT-Cross relative error too high: {rel_err}"

    # Gradient check
    grads = field.gradient(X_test[:3])
    assert grads.shape == (3, 10)
    assert not np.isnan(grads).any()


def test_facade_poisson_solver():
    """Tests hk.PoissonSolver on 2D polynomial Poisson benchmark."""
    solver = hk.PoissonSolver(dim=2, degree=8)
    
    # \nabla^2 u = 2(x^2 + y^2 - 2) with exact solution u*(x, y) = (1 - x^2)(1 - y^2)
    def f_rhs(x, y):
        return 2.0 * (x**2 + y**2 - 2.0)
    
    res = solver.solve(f_rhs)
    assert res["pde_residual_rmse"] < 1e-3
    assert res["solve_time_ms"] < 50.0

    # Continuous evaluation
    coords = np.array([[0.0, 0.0], [0.5, 0.5], [0.2, -0.3]])
    u_eval = solver(coords)
    assert len(u_eval) == 3


def test_facade_cbf_planner():
    """Tests hk.CBFPlanner obstacle avoidance with continuous KAN obstacle field."""
    field = hk.TensorField(spatial_dim=3, rank=8, degree=4)
    # Fit obstacle field (e.g. sphere centered at [0.5, 0.5, 0.5])
    N = 200
    X = np.random.uniform(-1, 1, (N, 3))
    # h(x) = dist^2 - R^2 (positive outside, negative inside)
    h_target = np.sum((X - np.array([0.5, 0.5, 0.5])) ** 2, axis=1) - 0.3 ** 2
    field.fit(X, h_target, alpha=1e-4, max_iters=5)

    planner = hk.CBFPlanner(safety_margin=0.05, alpha_cbf=2.0)
    planner.add_obstacle_field(field)

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 1.0, 1.0])
    result = planner.plan_trajectory(start, goal, dt=0.05, max_steps=100)

    assert result["success"] or len(result["trajectory"]) > 0
    assert not result["collision"]
