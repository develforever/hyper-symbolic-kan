import os
import tempfile
import pathlib
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.torch_kan.autograd_ops import ContinuousKANAutograd, TensorTrainKANAutograd
from src.torch_kan.layers import ContinuousKANLayer, TensorTrainKANLayer
from src.torch_kan.safetensors_io import save_kan_safetensors, load_kan_safetensors
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tt_kan import TensorTrainKAN


def test_continuous_kan_autograd_gradcheck():
    """
    Rygorystyczna weryfikacja analitycznych gradientów ContinuousKANAutograd
    względem różnic skończonych przy użyciu torch.autograd.gradcheck (precyzja float64).
    """
    N, D, R, K = 6, 3, 4, 3
    torch.manual_seed(42)

    X = torch.empty(N, D, dtype=torch.float64).uniform_(-0.8, 0.8).requires_grad_(True)
    lambdas = torch.randn(R, dtype=torch.float64, requires_grad=True)
    factors = [
        torch.randn(R, K + 1, dtype=torch.float64, requires_grad=True)
        for _ in range(D)
    ]

    def func(x, l, *f):
        return ContinuousKANAutograd.apply(x, l, K, *f)

    assert torch.autograd.gradcheck(
        func, (X, lambdas, *factors), eps=1e-6, atol=1e-4, rtol=1e-3
    ), "ContinuousKANAutograd gradcheck failed!"


def test_tensor_train_kan_autograd_gradcheck():
    """
    Rygorystyczna weryfikacja analitycznych gradientów TensorTrainKANAutograd
    względem różnic skończonych przy użyciu torch.autograd.gradcheck.
    """
    N, D, K = 6, 3, 3
    ranks = (1, 3, 4, 1)
    torch.manual_seed(42)

    X = torch.empty(N, D, dtype=torch.float64).uniform_(-0.8, 0.8).requires_grad_(True)
    cores = [
        torch.randn(ranks[d], K + 1, ranks[d + 1], dtype=torch.float64, requires_grad=True)
        for d in range(D)
    ]

    def func(x, *c):
        return TensorTrainKANAutograd.apply(x, K, ranks, *c)

    assert torch.autograd.gradcheck(
        func, (X, *cores), eps=1e-6, atol=1e-4, rtol=1e-3
    ), "TensorTrainKANAutograd gradcheck failed!"


def test_continuous_kan_layer_forward_and_shapes():
    """
    Weryfikacja zachowania kształtów dla ContinuousKANLayer (1D, 2D, 3D batching, multi-output).
    """
    layer_single = ContinuousKANLayer(in_features=4, out_features=1, rank=8, degree=3)
    x_2d = torch.randn(10, 4, dtype=torch.float64)
    out_2d = layer_single(x_2d)
    assert out_2d.shape == (10, 1)

    x_3d = torch.randn(5, 7, 4, dtype=torch.float64)
    out_3d = layer_single(x_3d)
    assert out_3d.shape == (5, 7, 1)

    layer_multi = ContinuousKANLayer(in_features=4, out_features=3, rank=8, degree=3)
    out_multi = layer_multi(x_2d)
    assert out_multi.shape == (10, 3)

    out_multi_3d = layer_multi(x_3d)
    assert out_multi_3d.shape == (5, 7, 3)


def test_tensor_train_kan_layer_forward_and_shapes():
    """
    Weryfikacja zachowania kształtów dla TensorTrainKANLayer (2D, 3D batching, multi-output).
    """
    ranks = [1, 4, 4, 1]
    layer_single = TensorTrainKANLayer(in_features=3, out_features=1, ranks=ranks, degree=3)
    x_2d = torch.randn(12, 3, dtype=torch.float64)
    out_2d = layer_single(x_2d)
    assert out_2d.shape == (12, 1)

    x_3d = torch.randn(4, 6, 3, dtype=torch.float64)
    out_3d = layer_single(x_3d)
    assert out_3d.shape == (4, 6, 1)

    layer_multi = TensorTrainKANLayer(in_features=3, out_features=2, ranks=ranks, degree=3)
    out_multi = layer_multi(x_2d)
    assert out_multi.shape == (12, 2)


def test_torch_kan_optimization_loop():
    """
    Test pętli optymalizacyjnej torch.optim.Adam z ContinuousKANLayer:
    Aproksymacja nieliniowej funkcji f(x, y) = sin(pi * x) * cos(pi * y).
    """
    torch.manual_seed(42)
    layer = ContinuousKANLayer(in_features=2, out_features=1, rank=12, degree=5)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)

    # Dane treningowe
    X_train = torch.empty(150, 2, dtype=torch.float64).uniform_(-0.9, 0.9)
    Y_train = (torch.sin(np.pi * X_train[:, 0]) * torch.cos(np.pi * X_train[:, 1])).unsqueeze(-1)

    initial_loss = nn.MSELoss()(layer(X_train), Y_train).item()

    for step in range(50):
        optimizer.zero_grad()
        pred = layer(X_train)
        loss = nn.MSELoss()(pred, Y_train)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, f"Optimizer did not converge sufficiently: initial={initial_loss}, final={final_loss}"


def test_tensor_train_kan_optimization_loop():
    """
    Test pętli optymalizacyjnej torch.optim.Adam z TensorTrainKANLayer.
    """
    torch.manual_seed(42)
    layer = TensorTrainKANLayer(in_features=3, out_features=1, ranks=[1, 4, 4, 1], degree=4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)

    X_train = torch.empty(100, 3, dtype=torch.float64).uniform_(-0.8, 0.8)
    Y_train = (X_train[:, 0] * torch.cos(X_train[:, 1]) + X_train[:, 2]**2).unsqueeze(-1)

    initial_loss = nn.MSELoss()(layer(X_train), Y_train).item()

    for step in range(50):
        optimizer.zero_grad()
        pred = layer(X_train)
        loss = nn.MSELoss()(pred, Y_train)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.2, f"TT-KAN Optimizer failed: initial={initial_loss}, final={final_loss}"


def test_hybrid_als_gradient_fine_tuning():
    """
    Test hybrydowego dopasowania:
    1. ALS (0 epok gradientowych) -> szybkie dopasowanie bazowe.
    2. Gradient Fine-Tuning z Adamem -> dopracowanie precyzji.
    """
    torch.manual_seed(42)
    layer = ContinuousKANLayer(in_features=3, out_features=1, rank=10, degree=4)

    X = torch.empty(200, 3, dtype=torch.float64).uniform_(-0.9, 0.9)
    Y = (torch.sin(np.pi * X[:, 0]) * torch.exp(-X[:, 1]**2) + 0.5 * X[:, 2]).unsqueeze(-1)

    # 1. ALS initial fit
    als_mse = layer.fit_als(X, Y, alpha=1e-4, max_als_iters=8)
    assert als_mse < 0.1, f"ALS fit MSE too high: {als_mse}"

    # 2. Gradient fine tuning
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.005)
    for _ in range(25):
        optimizer.zero_grad()
        loss = nn.MSELoss()(layer(X), Y)
        loss.backward()
        optimizer.step()

    fine_loss = nn.MSELoss()(layer(X), Y).item()
    assert fine_loss <= als_mse + 1e-4


def test_hybrid_tt_cross_gradient_fine_tuning():
    """
    Test hybrydowego dopasowania TT-Cross -> PyTorch Layer:
    1. TT-Cross dopasowanie czarnej skrzynki.
    2. Eksport do TensorTrainKANLayer i ewaluacja.
    """
    def target_fn(x):
        return np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) + x[:, 2]

    layer = TensorTrainKANLayer(in_features=3, out_features=1, ranks=[1, 6, 6, 1], degree=4)
    layer.fit_tt_cross(target_fn, max_rank=6, eps=1e-4)

    X_test = torch.empty(50, 3, dtype=torch.float64).uniform_(-0.8, 0.8)
    Y_expected = torch.from_numpy(target_fn(X_test.numpy())).unsqueeze(-1)

    Y_pred = layer(X_test)
    error = torch.mean((Y_pred - Y_expected)**2).item()
    assert error < 0.05, f"TT-Cross reconstruction error too high: {error}"


def test_safetensors_serialization_roundtrip():
    """
    Weryfikacja bezpiecznej serializacji i deserializacji SafeTensors dla ContinuousKANLayer,
    TensorTrainKANLayer oraz formatów NumPy TDFFNet i TensorTrainKAN.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # 1. ContinuousKANLayer roundtrip
        layer_cp = ContinuousKANLayer(in_features=3, out_features=1, rank=8, degree=4)
        cp_file = str(tmp_path / "model_cp.safetensors")
        save_kan_safetensors(layer_cp, cp_file)

        loaded_cp = load_kan_safetensors(cp_file, as_torch=True)
        assert isinstance(loaded_cp, ContinuousKANLayer)
        assert loaded_cp.in_features == layer_cp.in_features
        assert loaded_cp.rank == layer_cp.rank
        assert loaded_cp.degree == layer_cp.degree
        assert torch.equal(loaded_cp.lambdas, layer_cp.lambdas)
        for d in range(layer_cp.in_features):
            assert torch.equal(loaded_cp.factors[d], layer_cp.factors[d])

        X_test = torch.randn(10, 3, dtype=torch.float64)
        assert torch.allclose(layer_cp(X_test), loaded_cp(X_test))

        # 2. TensorTrainKANLayer roundtrip
        layer_tt = TensorTrainKANLayer(in_features=3, out_features=1, ranks=[1, 5, 5, 1], degree=3)
        tt_file = str(tmp_path / "model_tt.safetensors")
        save_kan_safetensors(layer_tt, tt_file)

        loaded_tt = load_kan_safetensors(tt_file, as_torch=True)
        assert isinstance(loaded_tt, TensorTrainKANLayer)
        assert loaded_tt.in_features == layer_tt.in_features
        assert loaded_tt.ranks == layer_tt.ranks
        assert loaded_tt.degree == layer_tt.degree
        for d in range(layer_tt.in_features):
            assert torch.equal(loaded_tt.cores[d], layer_tt.cores[d])

        assert torch.allclose(layer_tt(X_test), loaded_tt(X_test))

        # 3. NumPy TDFFNet roundtrip
        tdff = TDFFNet(spatial_dim=3, rank=6, degree=3)
        tdff_file = str(tmp_path / "tdff.safetensors")
        save_kan_safetensors(tdff, tdff_file)

        loaded_tdff = load_kan_safetensors(tdff_file, as_torch=False)
        assert isinstance(loaded_tdff, TDFFNet)
        assert np.array_equal(tdff.lambdas, loaded_tdff.lambdas)
        for d in range(3):
            assert np.array_equal(tdff.factors[d], loaded_tdff.factors[d])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
