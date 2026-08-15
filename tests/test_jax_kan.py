"""
Unit tests for JAX Backend (ContinuousKANJAXLayer, TensorTrainKANJAXLayer, Custom VJP).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import jax
    import jax.numpy as jnp
    from src.jax_kan.autograd_ops import (
        cp_kan_forward,
        tt_kan_forward,
        compute_chebyshev_jax
    )
    from src.jax_kan.layers import ContinuousKANJAXLayer, TensorTrainKANJAXLayer
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _reference_cp_kan_pure_jax(X, lambdas, factors):
    """Reference pure JAX implementation without custom VJP for gradient verification."""
    N, D = X.shape
    R = lambdas.shape[0]
    K1 = factors.shape[2]
    degree = K1 - 1

    cp_prod = jnp.ones((N, R), dtype=X.dtype)
    for d in range(D):
        T_d = compute_chebyshev_jax(X[:, d], degree)
        phi_d = T_d @ factors[d].T
        cp_prod = cp_prod * phi_d

    return cp_prod @ lambdas


def test_jax_cp_kan_custom_vjp_vs_autodiff():
    """Test custom VJP gradients for CP-KAN vs standard autodiff in float64."""
    if not _HAS_JAX:
        pytest.skip("JAX not installed")

    # Enable float64 for JAX
    from jax import config
    config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(123)
    N, D, R, K = 16, 3, 4, 3

    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.uniform(k1, (N, D), minval=-0.9, maxval=0.9, dtype=jnp.float64)
    lambdas = jax.random.normal(k2, (R,), dtype=jnp.float64)
    factors = jax.random.normal(k3, (D, R, K + 1), dtype=jnp.float64)

    # Loss function for custom VJP
    def loss_custom(x, lam, fact):
        y = cp_kan_forward(x, lam, fact)
        return jnp.sum(y ** 2)

    # Loss function for reference autodiff
    def loss_ref(x, lam, fact):
        y = _reference_cp_kan_pure_jax(x, lam, fact)
        return jnp.sum(y ** 2)

    # Check forward match
    y_custom = cp_kan_forward(X, lambdas, factors)
    y_ref = _reference_cp_kan_pure_jax(X, lambdas, factors)
    np.testing.assert_allclose(np.array(y_custom), np.array(y_ref), rtol=1e-12, atol=1e-12)

    # Compute gradients
    gx_c, glam_c, gfact_c = jax.grad(loss_custom, argnums=(0, 1, 2))(X, lambdas, factors)
    gx_r, glam_r, gfact_r = jax.grad(loss_ref, argnums=(0, 1, 2))(X, lambdas, factors)

    # Assert exact match between analytical VJP and automatic differentiation
    np.testing.assert_allclose(np.array(gx_c), np.array(gx_r), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.array(glam_c), np.array(glam_r), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.array(gfact_c), np.array(gfact_r), rtol=1e-9, atol=1e-9)


def _reference_tt_kan_pure_jax(X, *cores):
    """Reference pure JAX implementation of TT-KAN for autodiff verification."""
    N, D = X.shape
    K1 = cores[0].shape[1]
    degree = K1 - 1

    curr = jnp.ones((N, 1), dtype=X.dtype)
    for d in range(D):
        T_d = compute_chebyshev_jax(X[:, d], degree)
        M_d = jnp.einsum('nk, rks -> nrs', T_d, cores[d])
        curr = jnp.squeeze(jnp.matmul(jnp.expand_dims(curr, 1), M_d), axis=1)

    return jnp.squeeze(curr, axis=-1)


def test_jax_tt_kan_custom_vjp_vs_autodiff():
    """Test custom VJP gradients for TT-KAN vs standard autodiff in float64."""
    if not _HAS_JAX:
        pytest.skip("JAX not installed")

    from jax import config
    config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(42)
    N, D, K = 12, 4, 3
    ranks = [1, 3, 4, 2, 1]

    keys = jax.random.split(key, D + 1)
    X = jax.random.uniform(keys[0], (N, D), minval=-0.8, maxval=0.8, dtype=jnp.float64)

    cores = []
    for d in range(D):
        c = jax.random.normal(keys[d + 1], (ranks[d], K + 1, ranks[d + 1]), dtype=jnp.float64)
        cores.append(c)

    def loss_custom(x, *c_list):
        y = tt_kan_forward(x, *c_list)
        return jnp.sum(y ** 2)

    def loss_ref(x, *c_list):
        y = _reference_tt_kan_pure_jax(x, *c_list)
        return jnp.sum(y ** 2)

    # Check forward match
    y_c = tt_kan_forward(X, *cores)
    y_r = _reference_tt_kan_pure_jax(X, *cores)
    np.testing.assert_allclose(np.array(y_c), np.array(y_r), rtol=1e-12, atol=1e-12)

    # Gradients w.r.t X and cores
    grads_c = jax.grad(loss_custom, argnums=tuple(range(D + 1)))(X, *cores)
    grads_r = jax.grad(loss_ref, argnums=tuple(range(D + 1)))(X, *cores)

    # Gradient w.r.t X
    np.testing.assert_allclose(np.array(grads_c[0]), np.array(grads_r[0]), rtol=1e-9, atol=1e-9)

    # Gradients w.r.t cores
    for d in range(D):
        np.testing.assert_allclose(np.array(grads_c[d + 1]), np.array(grads_r[d + 1]), rtol=1e-9, atol=1e-9)


def test_jax_jit_compilation_and_shapes():
    """Verify jax.jit compilation and multi-dimensional batch forward pass."""
    if not _HAS_JAX:
        pytest.skip("JAX not installed")

    layer = ContinuousKANJAXLayer(in_features=3, out_features=2, rank=8, degree=4)
    params = layer.init_params()

    @jax.jit
    def jitted_forward(p, x):
        return layer.apply(p, x)

    # Test 2D shape (N, D)
    X_2d = jnp.zeros((10, 3), dtype=jnp.float64)
    out_2d = jitted_forward(params, X_2d)
    assert out_2d.shape == (10, 2)

    # Test 3D batch shape (B, T, D)
    X_3d = jnp.zeros((4, 8, 3), dtype=jnp.float64)
    out_3d = jitted_forward(params, X_3d)
    assert out_3d.shape == (4, 8, 2)


def test_jax_continuous_kan_layer_fit_als():
    """Verify that Closed-Form ALS in JAX layer fits non-linear 3D function in 0 epochs."""
    if not _HAS_JAX:
        pytest.skip("JAX not installed")

    layer = ContinuousKANJAXLayer(in_features=3, out_features=1, rank=12, degree=5)
    params = layer.init_params()

    np.random.seed(42)
    N = 200
    X = np.random.uniform(-0.9, 0.9, size=(N, 3))
    # Target non-linear function
    Y = (np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + X[:, 2] ** 2)[:, None]

    initial_pred = layer.apply(params, jnp.asarray(X, dtype=jnp.float64))
    initial_rmse = float(jnp.sqrt(jnp.mean((initial_pred - jnp.asarray(Y, dtype=jnp.float64)) ** 2)))

    updated_params, final_rmse = layer.fit_als(params, X, Y, alpha=1e-4, max_als_iters=8)

    assert final_rmse < initial_rmse
    assert final_rmse < 0.05, f"ALS did not converge sufficiently: RMSE={final_rmse}"


def test_jax_gradient_descent_optimization():
    """Verify that JAX ContinuousKAN layer converges under gradient descent."""
    if not _HAS_JAX:
        pytest.skip("JAX not installed")

    layer = ContinuousKANJAXLayer(in_features=2, out_features=1, rank=6, degree=3)
    params = layer.init_params()

    np.random.seed(99)
    N = 100
    X = jnp.asarray(np.random.uniform(-0.8, 0.8, size=(N, 2)), dtype=jnp.float64)
    Y = jnp.asarray((X[:, 0] ** 2 + X[:, 1] ** 2)[:, None], dtype=jnp.float64)

    def mse_loss(p):
        pred = layer.apply(p, X)
        return jnp.mean((pred - Y) ** 2)

    grad_fn = jax.jit(jax.grad(mse_loss))
    initial_loss = float(mse_loss(params))

    lr = 0.02
    for _ in range(30):
        grads = grad_fn(params)
        params["lambdas"] = params["lambdas"] - lr * grads["lambdas"]
        params["factors"] = params["factors"] - lr * grads["factors"]

    final_loss = float(mse_loss(params))
    assert final_loss < initial_loss
