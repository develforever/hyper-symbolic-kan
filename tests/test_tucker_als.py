"""
Testy kontraktu TuckerALSSolver (audyt M3).

Przed naprawa cala logika ALS w `fit` byla warunkowana `if D == 2`. Dla D != 2
petla wykonywala sie bezczynnie, rdzen i czynniki nigdy nie byly aktualizowane,
a zwracana wartosc to MSE losowej inicjalizacji -- zgloszona jako metryka
sukcesu. Ponizsze testy pilnuja, ze nieobslugiwany wymiar konczy sie jawnym
wyjatkiem, a nie cicha awaria.
"""
import numpy as np
import pytest

from src.tdff_net.tucker_tensor_field import TuckerTDFFNet
from src.tdff_net.tucker_als import TuckerALSSolver


def _sample(D: int, N: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.9, 0.9, (N, D))
    Y = np.prod(np.cos(X), axis=1)
    return X, Y


@pytest.mark.parametrize("D", [1, 3, 4, 5])
def test_fit_raises_for_unsupported_dimension(D: int):
    """Dla D != 2 `fit` musi rzucic NotImplementedError z aktualnym D w komunikacie."""
    model = TuckerTDFFNet(spatial_dim=D, ranks=[4] * D, degree=4)
    solver = TuckerALSSolver(alpha=1e-4, max_als_iters=4)
    X, Y = _sample(D)

    with pytest.raises(NotImplementedError) as exc:
        solver.fit(model, X, Y)

    assert f"D={D}" in str(exc.value), f"Komunikat nie zawiera aktualnego D: {exc.value}"


def test_fit_raises_before_touching_model_state():
    """Wyjatek pada w pierwszej linii -- model pozostaje nietkniety."""
    D = 3
    model = TuckerTDFFNet(spatial_dim=D, ranks=[4] * D, degree=4)
    core_before = model.core.copy()
    factors_before = [f.copy() for f in model.factors]
    ranks_before = list(model.ranks)

    X, Y = _sample(D)
    with pytest.raises(NotImplementedError):
        TuckerALSSolver().fit(model, X, Y)

    np.testing.assert_array_equal(model.core, core_before)
    for f, f0 in zip(model.factors, factors_before):
        np.testing.assert_array_equal(f, f0)
    assert list(model.ranks) == ranks_before


def test_fit_still_works_for_supported_dimension():
    """D=2 to jedyna zaimplementowana sciezka -- musi nadal realnie dopasowywac."""
    D = 2
    model = TuckerTDFFNet(spatial_dim=D, ranks=[8, 8], degree=6)
    X, Y = _sample(D, N=2000, seed=7)

    mse_random_init = float(np.mean((Y - model.evaluate(X)) ** 2))
    mse_fitted = TuckerALSSolver(alpha=1e-6, max_als_iters=8).fit(model, X, Y)

    assert mse_fitted < 1e-3, f"MSE po dopasowaniu zbyt duze: {mse_fitted}"
    assert mse_fitted < mse_random_init, (
        f"Dopasowanie nie poprawilo MSE losowej inicjalizacji "
        f"({mse_fitted} vs {mse_random_init})"
    )
