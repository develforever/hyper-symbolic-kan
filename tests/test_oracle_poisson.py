r"""
Wyrocznia ANALITYCZNA dla solvera Poissona (audyt V3, PROTOKOL zasada 1).

Zamiast porównywać solver z drugą własną implementacją, porównujemy go z zamkniętym
rozwiązaniem analitycznym o twardym progu błędu. To jest oracle spoza numeryki repo:
funkcja u*(x, y) = (1 - x^2)(1 - y^2) spełnia dokładnie

    nabla^2 u* = 2 (x^2 + y^2 - 2),   u* = 0 na brzegu [-1, 1]^2,

i jest wielomianem stopnia 2 w każdej zmiennej, więc jest DOKŁADNIE reprezentowalna
w bazie Czebyszewa T_0..T_deg (x^2 = (T_2 + 1)/2). Poprawny solver kolokacyjny musi
ją odtworzyć z dokładnością ograniczoną wyłącznie uwarunkowaniem układu normalnego,
nie błędem aproksymacji bazy. Stąd twardy próg:

    L2_rel < 1e-8.

Zmierzone (seed 0/1/2, degree=6, siatka ewaluacyjna 40x40 we wnętrzu): L2_rel rzędu
3e-16..9e-16 -- ~15 rzędów pod progiem. Test przechodzi z dużym zapasem, co znaczy,
że rekurencja drugiej pochodnej T''_k oraz montaż operatora Laplace'a w
``SpectralKANPoissonSolver`` są poprawne dla tego przypadku.

Zakres: testujemy WYŁĄCZNIE ``SpectralKANPoissonSolver`` (2D, ścieżka closed-form).
NIE testujemy ``TTPoissonSolver`` (D >= 4, ALS) -- ma otwarty błąd M2 (macierz
projektowa pomija człony m != d laplasjanu) i jest poza zakresem tej sesji.

PROTOKOL zasada 1: gdyby próg nie przechodził, wynikiem jest ZARAPORTOWANA liczba
L2_rel, nie podniesienie progu ani rozluźnienie asercji.

Uwaga o determinizmie: solver losuje punkty kolokacji globalnym ``np.random`` (audyt
A4). Ustawiamy ziarno przed ``fit`` wyłącznie po to, by test był powtarzalny; wynik
jest i tak odporny na wybór ziarna (zmierzone trzy ziarna, wszystkie ~1e-16).
"""

import numpy as np

from src.applications.pde_poisson_solver import (
    PoissonAnalyticalSolution,
    SpectralKANPoissonSolver,
)

L2_REL_HARD_THRESHOLD = 1e-8


def _interior_eval_grid(n: int = 40, half: float = 0.95) -> np.ndarray:
    """Deterministyczna siatka ewaluacyjna n x n we wnętrzu (-half, half)^2."""
    g = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel()])


def test_poisson_2d_polynomial_l2_rel_below_1e_minus_8() -> None:
    """u* = (1-x^2)(1-y^2): dokładnie reprezentowalna, twardy próg L2_rel < 1e-8."""
    u_exact, f_rhs, g_bc = PoissonAnalyticalSolution.get_2d_polynomial()

    np.random.seed(0)  # tylko powtarzalność punktów kolokacji (audyt A4)
    solver = SpectralKANPoissonSolver(spatial_dim=2, degree=6)
    info = solver.fit(f_rhs, g_bc, n_interior=1500, n_boundary=800, alpha_reg=1e-12, beta_bc=200.0)

    x_eval = _interior_eval_grid()
    l2_rel = solver.compute_l2_relative_error(x_eval, u_exact)

    # Zawsze raportuj zmierzoną liczbę (PROTOKOL zasada 1) -- widoczne przy `pytest -s`.
    print(
        f"\n[oracle-poisson] 2d_polynomial degree=6 L2_rel={l2_rel:.3e} "
        f"pde_rmse={info['pde_residual_rmse']:.2e} bc_rmse={info['bc_residual_rmse']:.2e}"
    )

    assert l2_rel < L2_REL_HARD_THRESHOLD, (
        f"L2_rel = {l2_rel:.3e} >= prog {L2_REL_HARD_THRESHOLD:.0e}. "
        "To jest wynik do zaraportowania, nie powod do podniesienia progu."
    )
