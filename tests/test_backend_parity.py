r"""
Parzystość backendów przeciw WYROCZNI ZEWNĘTRZNEJ (audyt V3, PROTOKOL zasada 1 i 3).

Cztery backendy liczą tę samą bazę Czebyszewa T_k(x) i jej pochodną: NumPy, PyTorch,
JAX, C++. Każdy jest tu porównany z zewnętrzną wyrocznią (``scipy.special.eval_chebyt``
oraz ``numpy.polynomial.chebyshev.chebder``) na TYM SAMYM wejściu -- NIE ze sobą
nawzajem. Zgodność backendów między sobą jest testem spójności; zgodność z niezależną
wyrocznią jest testem poprawności.

DWA testy tu PADAJĄ i to jest zaplanowany wynik, nie awaria (oba oznaczone
``xfail(strict=True)`` z pełnym opisem: zmierzona rozbieżność, przyczyna, klasyfikacja
defekt/ograniczenie). ``strict=True`` oznacza, że gdy przyczyna zniknie (naprawa),
test zamieni się w XPASS -> twardy błąd, zmuszając do zdjęcia ``xfail`` -- działa więc
jak żywy strażnik regresji, nie jak wyciszenie.

  (1) JAX bez ``jax_enable_x64`` liczy w float32. Zmierzone: wartości abs ~4.2e-6,
      T' abs ~3.5e-4 -- nie ma prawa trafić w próg float64 (1e-13 / 1e-11).
      Klasyfikacja: UDOKUMENTOWANE OGRANICZENIE BACKENDU (nie defekt matematyczny).
      Dowód: test_jax_matches_oracle_at_float32_tolerance PRZECHODZI -- wzór jest
      poprawny, brakuje tylko precyzji. Naprawa (audyt N8: ``jax.config.update(
      'jax_enable_x64', True)``) idzie osobnym commitem po decyzji.

  (2) PyTorch NIE przycina dziedziny do [-1, 1] (``_compute_chebyshev_torch`` używa
      x bezpośrednio), podczas gdy NumPy i C++ przycinają. Poza [-1, 1] rozjazd jest
      JAKOŚCIOWY: dla x = 2, degree = 16 wyrocznia kontraktowa dziedziny (T_k(clip x))
      daje |wartość| <= 1, a torch daje prawdziwy wielomian ~7.1e8. To nie kwestia
      tolerancji. Klasyfikacja: RÓŻNICA KONTRAKTU DZIEDZINY, nie defekt wzoru.
      Dowód: test_torch_matches_true_polynomial_outside_domain PRZECHODZI -- torch
      liczy prawdziwy T_k(x) dokładnie, po prostu nie rzutuje na dziedzinę.
      Rozstrzygnięcie (ujednolicić clipping wszystkich backendów albo assertować
      wejście, audyt M8/M9) idzie osobnym commitem po decyzji.

Backend C++ nie eksportuje surowej bazy, tylko ewaluację TT-KAN. T_k(x) wyciągamy
przez rdzeń one-hot: dla D=1, rdzeń (1, K1, 1) z jedynką na pozycji k daje
sum_j core_j T_j(x) = T_k(x); gradient tego samego rdzenia daje T'_k(x). To realny
kod jądra C++, nie druga implementacja Pythona.
"""

import importlib.util

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebder, chebval
from scipy.special import eval_chebyt

from src.hs_ckan.chebyshev_kan import ChebyshevKANBasis
from src.applications.pde_poisson_solver import chebyshev_derivatives_2nd

from tests._native import requires_native

# Wykrywanie opcjonalnych backendow bez warunkowego wiazania nazw (czyste dla pyright);
# faktyczny import zywie lokalnie w ewaluatorach ponizej, wolanych tylko gdy backend jest.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_JAX = importlib.util.find_spec("jax") is not None

# Tolerancje float64 -- identyczne jak w test_oracle_chebyshev.py (patrz tam uzasadnienie).
ATOL_VAL = 1e-13
RTOL_D1 = 1e-11
ATOL_D1 = 1e-11
# float32: eps_f32 ~ 1.2e-7; po O(K) krokach rekurencji rzedu 1e-5 dla wartosci,
# wieksze dla pochodnych skalowanych przez K^2. Prog jawnie luzny, bo to inna precyzja.
ATOL_VAL_F32 = 1e-4
ATOL_OUT_OF_DOMAIN = 1e-9

DEGREES = [8, 16]


def _in_domain_grid() -> np.ndarray:
    edge = np.array([1e-12, 1e-9, 1e-6, 1e-3, 1e-2])
    pts = np.concatenate([
        np.array([-1.0, 1.0, 0.0]),
        1.0 - edge, -1.0 + edge,
        np.linspace(-0.999, 0.999, 41),
    ])
    return np.unique(pts).astype(np.float64)


def _out_of_domain_grid() -> np.ndarray:
    return np.array([-3.0, -2.0, -1.5, -1.2, 1.2, 1.5, 2.0, 3.0], dtype=np.float64)


# --- wyrocznie ---
def _oracle_values(x: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([eval_chebyt(k, x) for k in range(degree + 1)], axis=1)


def _oracle_first_derivative(x: np.ndarray, degree: int) -> np.ndarray:
    eye = np.eye(degree + 1)
    return np.stack([chebval(x, chebder(eye[k], m=1)) for k in range(degree + 1)], axis=1)


def _oracle_values_domain_contract(x: np.ndarray, degree: int) -> np.ndarray:
    """Wyrocznia kontraktu dziedziny: prawdziwy wielomian na x rzutowanym na [-1, 1].

    To jest zachowanie, którego oczekuje model KAN zdefiniowany na [-1, 1]: NumPy i C++
    je realizują (clipping), PyTorch nie. ``np.clip`` to projekcja dziedziny z
    definicji modelu, nie reimplementacja wielomianu -- cała matematyka wielomianu
    pochodzi z SciPy.
    """
    return _oracle_values(np.clip(x, -1.0, 1.0), degree)


# --- ewaluatory backendów: zwracaja (N, degree+1) ---
def _numpy_values(x: np.ndarray, degree: int) -> np.ndarray:
    return ChebyshevKANBasis(degree).compute_basis(x[:, None])


def _numpy_first_derivative(x: np.ndarray, degree: int) -> np.ndarray:
    _, dT, _ = chebyshev_derivatives_2nd(x, degree)
    return dT


def _torch_values(x: np.ndarray, degree: int) -> np.ndarray:
    import torch

    from src.torch_kan.autograd_ops import _compute_chebyshev_torch
    return _compute_chebyshev_torch(torch.from_numpy(x), degree).detach().numpy()


def _torch_first_derivative(x: np.ndarray, degree: int) -> np.ndarray:
    import torch

    from src.torch_kan.autograd_ops import _compute_chebyshev_and_deriv_torch
    _, dT = _compute_chebyshev_and_deriv_torch(torch.from_numpy(x), degree)
    return dT.detach().numpy()


def _jax_values(x: np.ndarray, degree: int) -> np.ndarray:
    import jax.numpy as jnp  # type: ignore

    from src.jax_kan.autograd_ops import compute_chebyshev_jax
    return np.asarray(compute_chebyshev_jax(jnp.asarray(x), degree))


def _jax_first_derivative(x: np.ndarray, degree: int) -> np.ndarray:
    import jax.numpy as jnp  # type: ignore

    from src.jax_kan.autograd_ops import compute_chebyshev_and_deriv_jax
    _, dT = compute_chebyshev_and_deriv_jax(jnp.asarray(x), degree)
    return np.asarray(dT)


def _cpp_values(x: np.ndarray, degree: int) -> np.ndarray:
    from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine
    eng = FastCPPKANEngine(spatial_dim=1, degree=degree)
    xc = x[:, None]
    out = np.empty((x.size, degree + 1), dtype=np.float64)
    for k in range(degree + 1):
        core = np.zeros((1, degree + 1, 1)); core[0, k, 0] = 1.0
        out[:, k] = eng.evaluate_batch(xc, [core], [1, 1])
    return out


def _cpp_first_derivative(x: np.ndarray, degree: int) -> np.ndarray:
    from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine
    eng = FastCPPKANEngine(spatial_dim=1, degree=degree)
    xc = x[:, None]
    out = np.empty((x.size, degree + 1), dtype=np.float64)
    for k in range(degree + 1):
        core = np.zeros((1, degree + 1, 1)); core[0, k, 0] = 1.0
        out[:, k] = eng.gradient_batch(xc, [core], [1, 1])[:, 0]
    return out


# ======================================================================
# In-domain: wartosci vs czysta wyrocznia eval_chebyt (prog float64)
# ======================================================================
@pytest.mark.parametrize("degree", DEGREES)
def test_numpy_values_vs_oracle(degree: int) -> None:
    x = _in_domain_grid()
    err = np.max(np.abs(_numpy_values(x, degree) - _oracle_values(x, degree)))
    assert err < ATOL_VAL, f"numpy degree={degree}: {err:.3e}"


@pytest.mark.parametrize("degree", DEGREES)
def test_numpy_first_derivative_vs_oracle(degree: int) -> None:
    x = _in_domain_grid()
    assert np.allclose(_numpy_first_derivative(x, degree), _oracle_first_derivative(x, degree),
                       rtol=RTOL_D1, atol=ATOL_D1)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("degree", DEGREES)
def test_torch_values_vs_oracle(degree: int) -> None:
    x = _in_domain_grid()
    err = np.max(np.abs(_torch_values(x, degree) - _oracle_values(x, degree)))
    assert err < ATOL_VAL, f"torch degree={degree}: {err:.3e}"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("degree", DEGREES)
def test_torch_first_derivative_vs_oracle(degree: int) -> None:
    x = _in_domain_grid()
    assert np.allclose(_torch_first_derivative(x, degree), _oracle_first_derivative(x, degree),
                       rtol=RTOL_D1, atol=ATOL_D1)


@requires_native
@pytest.mark.parametrize("degree", DEGREES)
def test_cpp_values_vs_oracle(degree: int) -> None:
    x = _in_domain_grid()
    err = np.max(np.abs(_cpp_values(x, degree) - _oracle_values(x, degree)))
    assert err < ATOL_VAL, f"cpp degree={degree}: {err:.3e}"


@requires_native
@pytest.mark.parametrize("degree", DEGREES)
def test_cpp_first_derivative_vs_oracle(degree: int) -> None:
    x = _in_domain_grid()
    assert np.allclose(_cpp_first_derivative(x, degree), _oracle_first_derivative(x, degree),
                       rtol=RTOL_D1, atol=ATOL_D1)


# ======================================================================
# ZAPLANOWANE PADNIECIE (1): JAX float32 nie trafia w prog float64.
# ======================================================================
@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "JAX bez jax_enable_x64 liczy w float32: zmierzone wartosci abs ~4.2e-6 "
        "(prog float64 ATOL_VAL=1e-13). UDOKUMENTOWANE OGRANICZENIE BACKENDU, nie "
        "defekt matematyczny -- test_jax_matches_oracle_at_float32_tolerance przechodzi. "
        "Naprawa: jax.config.update('jax_enable_x64', True) (audyt N8), osobny commit."
    ),
)
@pytest.mark.parametrize("degree", DEGREES)
def test_jax_values_vs_oracle_float64_threshold(degree: int) -> None:
    x = _in_domain_grid()
    err = np.max(np.abs(_jax_values(x, degree) - _oracle_values(x, degree)))
    assert err < ATOL_VAL, f"jax degree={degree}: {err:.3e}"


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("degree", DEGREES)
def test_jax_matches_oracle_at_float32_tolerance(degree: int) -> None:
    """Dowod, ze wzor JAX jest poprawny -- rozjazd to wylacznie precyzja float32."""
    x = _in_domain_grid()
    err = np.max(np.abs(_jax_values(x, degree) - _oracle_values(x, degree)))
    assert err < ATOL_VAL_F32, f"jax degree={degree} @f32: {err:.3e}"


# ======================================================================
# ZAPLANOWANE PADNIECIE (2): torch nie przycina dziedziny -> rozjazd poza [-1,1].
# ======================================================================
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "torch nie przycina wejscia do [-1, 1] (_compute_chebyshev_torch uzywa x "
        "bezposrednio), NumPy i C++ przycinaja. Poza dziedzina rozjazd JAKOSCIOWY: "
        "dla x=2, degree=16 kontrakt dziedziny daje |T_k|<=1, torch ~7.1e8. To nie "
        "kwestia tolerancji. ROZNICA KONTRAKTU DZIEDZINY, nie defekt wzoru -- "
        "test_torch_matches_true_polynomial_outside_domain przechodzi. Ujednolicenie "
        "(audyt M8/M9) osobnym commitem po decyzji."
    ),
)
@pytest.mark.parametrize("degree", DEGREES)
def test_torch_respects_domain_contract_outside_interval(degree: int) -> None:
    x = _out_of_domain_grid()
    err = np.max(np.abs(_torch_values(x, degree) - _oracle_values_domain_contract(x, degree)))
    assert err < ATOL_OUT_OF_DOMAIN, f"torch out-of-domain degree={degree}: {err:.3e}"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("degree", DEGREES)
def test_torch_matches_true_polynomial_outside_domain(degree: int) -> None:
    """Dowod, ze matematyka torch jest poprawna: poza [-1,1] torch = prawdziwy T_k(x)
    (wyrocznia BEZ clipu). Rozni sie od NumPy/C++ tylko brakiem projekcji dziedziny."""
    x = _out_of_domain_grid()
    err = np.max(np.abs(_torch_values(x, degree) - _oracle_values(x, degree)))
    assert err < 1e-8, f"torch vs true polynomial degree={degree}: {err:.3e}"


# Kontrola pozytywna kontraktu dziedziny: NumPy i C++ RESPEKTUJA go (przechodza).
@pytest.mark.parametrize("degree", DEGREES)
def test_numpy_respects_domain_contract_outside_interval(degree: int) -> None:
    x = _out_of_domain_grid()
    err = np.max(np.abs(_numpy_values(x, degree) - _oracle_values_domain_contract(x, degree)))
    assert err < ATOL_OUT_OF_DOMAIN, f"numpy out-of-domain degree={degree}: {err:.3e}"


@requires_native
@pytest.mark.parametrize("degree", DEGREES)
def test_cpp_respects_domain_contract_outside_interval(degree: int) -> None:
    x = _out_of_domain_grid()
    err = np.max(np.abs(_cpp_values(x, degree) - _oracle_values_domain_contract(x, degree)))
    assert err < ATOL_OUT_OF_DOMAIN, f"cpp out-of-domain degree={degree}: {err:.3e}"


# ======================================================================
# Tabela rozbieznosci backend x regime (zawsze przechodzi; widoczna przy `pytest -s`).
# ======================================================================
def test_report_backend_divergence_table() -> None:
    degree = 16
    xi = _in_domain_grid()
    xo = _out_of_domain_grid()
    ov_i = _oracle_values(xi, degree)
    ov_contract = _oracle_values_domain_contract(xo, degree)

    rows = [("numpy", _numpy_values(xi, degree), _numpy_values(xo, degree))]
    if _HAS_TORCH:
        rows.append(("torch", _torch_values(xi, degree), _torch_values(xo, degree)))
    if _HAS_JAX:
        rows.append(("jax(f32)", _jax_values(xi, degree), _jax_values(xo, degree)))
    try:
        rows.append(("cpp", _cpp_values(xi, degree), _cpp_values(xo, degree)))
    except Exception as exc:  # native niedostepny
        print(f"\n[parity] cpp pominiety: {exc}")

    print(f"\n[parity] degree={degree}  max|backend - oracle|")
    print(f"{'backend':10s} {'in-domain(pure)':>18s} {'out-of-domain(contract)':>26s}")
    for name, vi, vo in rows:
        di = np.max(np.abs(vi - ov_i))
        do = np.max(np.abs(vo - ov_contract))
        print(f"{name:10s} {di:>18.2e} {do:>26.2e}")
