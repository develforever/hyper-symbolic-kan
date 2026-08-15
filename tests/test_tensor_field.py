import numpy as np
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.tdff_net.tt_kan import TensorTrainKAN

def test_tdff_net_analytical_gradient():
    """Weryfikacja czy analityczny gradient Czebyszewa zgadza się z różnicami skończonymi."""
    model = TDFFNet(spatial_dim=3, rank=8, degree=4)
    X = np.random.uniform(-0.8, 0.8, (20, 3))
    
    grad_analytic = model.gradient(X)
    
    eps = 1e-6
    grad_num = np.zeros_like(grad_analytic)
    for d in range(3):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, d] += eps
        X_minus[:, d] -= eps
        grad_num[:, d] = (model.evaluate(X_plus) - model.evaluate(X_minus)) / (2.0 * eps)
        
    diff = np.max(np.abs(grad_analytic - grad_num))
    assert diff < 1e-5, f"Błąd gradientu przekracza tolerancję: {diff}"

def test_closed_form_als_convergence_and_stability():
    """Weryfikacja czy ALS z normalizacją zbiega i nie eksploduje numerycznie."""
    model = TDFFNet(spatial_dim=3, rank=12, degree=5)
    solver = ClosedFormALSSolver(alpha=1e-4, max_als_iters=8)
    
    # Cel: funkcja nieliniowa w 3D (kombinacja sferyczna)
    X_train = np.random.uniform(-0.9, 0.9, (1000, 3))
    Y_train = np.sin(np.pi * X_train[:, 0]) * np.cos(np.pi * X_train[:, 1]) + 0.5 * X_train[:, 2]**2
    
    mse = solver.fit(model, X_train, Y_train)
    assert not np.isnan(mse)
    assert not np.isinf(mse)
    assert mse < 0.05, f"Błąd MSE dopasowania ALS zbyt wysoki: {mse}"
    
    # Sprawdzenie czy wagi i lambdy są ograniczone (brak eksplozji)
    assert np.all(np.isfinite(model.lambdas))
    for f in model.factors:
        assert np.all(np.isfinite(f))
        # Normy wierszy powinny być znormalizowane do ~1.0
        row_norms = np.linalg.norm(f, axis=1)
        np.testing.assert_allclose(row_norms, 1.0, atol=1e-4)

def test_tt_kan_gradient_exactness():
    """Weryfikacja analitycznego gradientu TT-KAN w D=5."""
    model = TensorTrainKAN(spatial_dim=5, ranks=[1, 4, 4, 4, 4, 1], degree=3)
    X = np.random.uniform(-0.7, 0.7, (15, 5))
    
    grad_analytic = model.gradient(X)
    eps = 1e-6
    grad_num = np.zeros_like(grad_analytic)
    for d in range(5):
        X_p = X.copy()
        X_m = X.copy()
        X_p[:, d] += eps
        X_m[:, d] -= eps
        grad_num[:, d] = (model.evaluate(X_p) - model.evaluate(X_m)) / (2.0 * eps)
        
    diff = np.max(np.abs(grad_analytic - grad_num))
    assert diff < 1e-4, f"Błąd gradientu TT-KAN: {diff}"
