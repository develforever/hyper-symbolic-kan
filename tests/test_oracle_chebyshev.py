r"""
Wyrocznie ZEWNĘTRZNE dla bazy Czebyszewa i jej pochodnych (audyt V3, PROTOKOL zasada 1 i 3).

Do tej pory ``tests/`` porównywały backend C++ z pythonowym fallbackiem napisanym
przez tego samego autora z tego samego wzoru rekurencyjnego. To jest test SPÓJNOŚCI,
nie POPRAWNOŚCI: jeśli obie strony mylą się tak samo, zgodność niczego nie dowodzi.
Dokładnie tak przeszły trzy błędy matematyczne (audyt M1/M2/M7).

Ten plik weryfikuje kod repozytorium przeciw dwóm niezależnym wyroczniom SPOZA repo:

  * wartości bazy T_k(x)      -> ``scipy.special.eval_chebyt`` (algorytm własny SciPy),
  * pierwsza pochodna T'_k(x) -> ``numpy.polynomial.chebyshev.chebder(e_k, m=1)`` + ``chebval``,
  * druga pochodna  T''_k(x)  -> ``numpy.polynomial.chebyshev.chebder(e_k, m=2)`` + ``chebval``.

Wyrocznia pochodnej działa na współczynnikach: dla pojedynczego wielomianu bazowego
T_k jego wektor współczynników to e_k (one-hot). ``chebder`` różniczkuje szereg
w przestrzeni współczynników niezależnie od naszej trójelementowej rekurencji, a
``chebval`` ewaluuje wynik. Żadna z tych funkcji nie dzieli z repo ani jednej linii
kodu liczącego pochodną, więc wspólny błąd wzoru jest wykrywalny.

Testowany kod repozytorium:
  * ``src.hs_ckan.chebyshev_kan.ChebyshevKANBasis`` (kanoniczna baza NumPy),
  * ``src.applications.pde_poisson_solver.chebyshev_derivatives_2nd`` (jedyna ścieżka
    w repo licząca T, T' i T'' w jednym przebiegu -- używana przez solver Poissona).

Siatka celowo obejmuje końce przedziału (x = +-1) i punkty tuż przy nich. Tam
|T'_K| osiąga K^2, a |T''_K| osiąga K^2 (K^2 - 1) / 3 -- to najostrzejsze miejsce
na rozjazd numeryczny i jednocześnie miejsce, w którym clipping dziedziny [-1, 1]
robi największą różnicę.

Uzasadnienie tolerancji (nie strojone pod wynik -- zmierzone i ograniczone od góry):

  Rekurencja T_{k+1} = 2x T_k - T_{k-1} wykonuje O(K) operacji zmiennoprzecinkowych;
  błąd zaokrągleń rośnie jak ~ C * K * eps (eps = 2.22e-16). Dla K = 16 daje to
  rząd 1e-14. Pochodne dziedziczą ten błąd przeskalowany przez rząd wielkości samej
  pochodnej (do K^2 dla T', do ~K^4/3 dla T'').

  Zmierzone maksima (degree = 16, siatka poniżej), zaokrąglone w górę:
    wartości : abs 8.0e-15                       -> ATOL_VAL   = 1e-13  (~12x zapas)
    T'       : abs 4.5e-13, rel 2.4e-14 (|o|>1)  -> RTOL/ATOL_D1 = 1e-11 (rel ~400x,
                                                    abs ~20x zapas nad 4.5e-13)
    T''      : abs 4.4e-11, rel 5.7e-14 (|o|>1)  -> RTOL_D2 = 1e-10, ATOL_D2 = 1e-9
                                                    (abs ~23x zapas nad 4.4e-11)

  Progi są ustawione powyżej zmierzonego maksimum z jawnym, opisanym zapasem,
  a nie dobrane tak, żeby przeszło. Jeżeli którykolwiek przestanie przechodzić,
  to jest wynik do zaraportowania (PROTOKOL zasada 1), nie powód do podniesienia progu.
"""

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebder, chebval
from scipy.special import eval_chebyt

from src.applications.pde_poisson_solver import chebyshev_derivatives_2nd
from src.hs_ckan.chebyshev_kan import ChebyshevKANBasis

# --- tolerancje (patrz uzasadnienie w docstringu modułu) ---
ATOL_VAL = 1e-13
RTOL_D1 = 1e-11
ATOL_D1 = 1e-11
RTOL_D2 = 1e-10
ATOL_D2 = 1e-9

DEGREES = [5, 8, 16]


def _in_domain_grid() -> np.ndarray:
    """Siatka w [-1, 1]: dokładne końce +-1, punkty tuż przy nich, i wnętrze.

    Clipping [-1, 1] jest tu tożsamościowy, więc rekurencja repo daje prawdziwy
    wielomian i można ją bezpośrednio porównać z wyrocznią bez zaciemniania przez
    projekcję dziedziny (ta jest testowana osobno w test_backend_parity.py).
    """
    edge = np.array([1e-12, 1e-9, 1e-6, 1e-3, 1e-2])
    pts = np.concatenate([
        np.array([-1.0, 1.0, 0.0]),
        1.0 - edge,          # tuż przy prawym końcu, gdzie |T'_K| -> K^2
        -1.0 + edge,         # tuż przy lewym końcu
        np.linspace(-0.999, 0.999, 51),
    ])
    return np.unique(pts).astype(np.float64)


def _oracle_values(x: np.ndarray, degree: int) -> np.ndarray:
    """T_k(x) dla k = 0..degree z SciPy: kształt (N, degree+1)."""
    return np.stack([eval_chebyt(k, x) for k in range(degree + 1)], axis=1)


def _oracle_derivative(x: np.ndarray, degree: int, m: int) -> np.ndarray:
    """m-ta pochodna T_k(x) przez numpy.polynomial: kształt (N, degree+1).

    Współczynnik pojedynczego T_k to e_k; chebder(e_k, m) różniczkuje w przestrzeni
    współczynników, chebval ewaluuje. Niezależne od rekurencji repo.
    """
    eye = np.eye(degree + 1)
    cols = [chebval(x, chebder(eye[k], m=m)) for k in range(degree + 1)]
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("degree", DEGREES)
def test_numpy_basis_values_vs_scipy_eval_chebyt(degree: int) -> None:
    """Kanoniczna baza NumPy (ChebyshevKANBasis) vs scipy.special.eval_chebyt."""
    x = _in_domain_grid()
    T_repo = ChebyshevKANBasis(degree).compute_basis(x[:, None])  # (N, degree+1) dla D=1
    T_oracle = _oracle_values(x, degree)

    assert T_repo.shape == T_oracle.shape
    err = np.max(np.abs(T_repo - T_oracle))
    assert err < ATOL_VAL, f"degree={degree}: max|T_repo - T_scipy| = {err:.3e} >= {ATOL_VAL:.0e}"


@pytest.mark.parametrize("degree", DEGREES)
def test_recurrence_values_vs_scipy_eval_chebyt(degree: int) -> None:
    """T z chebyshev_derivatives_2nd (ścieżka solvera Poissona) vs SciPy."""
    x = _in_domain_grid()
    T_repo, _, _ = chebyshev_derivatives_2nd(x, degree)
    T_oracle = _oracle_values(x, degree)
    err = np.max(np.abs(T_repo - T_oracle))
    assert err < ATOL_VAL, f"degree={degree}: max|T - T_scipy| = {err:.3e} >= {ATOL_VAL:.0e}"


@pytest.mark.parametrize("degree", DEGREES)
def test_first_derivative_vs_numpy_chebder(degree: int) -> None:
    """T' z chebyshev_derivatives_2nd vs numpy.polynomial.chebyshev.chebder(m=1)."""
    x = _in_domain_grid()
    _, dT_repo, _ = chebyshev_derivatives_2nd(x, degree)
    dT_oracle = _oracle_derivative(x, degree, m=1)
    assert np.allclose(dT_repo, dT_oracle, rtol=RTOL_D1, atol=ATOL_D1), (
        f"degree={degree}: max abs = {np.max(np.abs(dT_repo - dT_oracle)):.3e}"
    )


@pytest.mark.parametrize("degree", DEGREES)
def test_second_derivative_vs_numpy_chebder_m2(degree: int) -> None:
    """T'' z chebyshev_derivatives_2nd vs numpy.polynomial.chebyshev.chebder(m=2)."""
    x = _in_domain_grid()
    _, _, d2T_repo = chebyshev_derivatives_2nd(x, degree)
    d2T_oracle = _oracle_derivative(x, degree, m=2)
    assert np.allclose(d2T_repo, d2T_oracle, rtol=RTOL_D2, atol=ATOL_D2), (
        f"degree={degree}: max abs = {np.max(np.abs(d2T_repo - d2T_oracle)):.3e}"
    )


def test_edge_derivative_magnitude_is_K_squared() -> None:
    """Kontrola pozytywna (PROTOKOL zasada 1): przy x = +-1 wyrocznia daje T'_K(1) = K^2,
    T'_K(-1) = (-1)^(K+1) K^2. Gdyby siatka lub wyrocznia liczyły coś banalnego, ten
    warunek by nie zaszedł -- tu weryfikuje, że najostrzejszy punkt jest faktycznie badany.
    """
    degree = 16
    x = np.array([-1.0, 1.0])
    dT_oracle = _oracle_derivative(x, degree, m=1)  # (2, degree+1)
    K = degree
    assert dT_oracle[1, K] == pytest.approx(K * K)              # x = +1
    assert dT_oracle[0, K] == pytest.approx(((-1) ** (K + 1)) * K * K)  # x = -1

    # I nasza rekurencja trafia w tę samą wartość K^2 na krawędzi.
    _, dT_repo, _ = chebyshev_derivatives_2nd(x, degree)
    assert dT_repo[1, K] == pytest.approx(K * K, rel=RTOL_D1, abs=ATOL_D1)
