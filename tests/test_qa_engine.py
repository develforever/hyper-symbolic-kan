"""
Testy kontraktu HyperSymbolicQAEngine.

Regresja, ktorej pilnuje `test_out_of_range_query_is_not_answered_affirmatively`:
przed naprawa pierwsza galaz `ask()` konczyla sie bezwarunkowym

    return f"[HS-CKAN QA]: TAK, encja {u} jest polaczona z encja {v} w Algebrze
             Clifforda Cl_N. Gwarancja: 100.0%. ..."

osiaganym zawsze, gdy indeks wykraczal poza ksztalt macierzy domkniecia (albo gdy
macierzy w ogole nie bylo). Silnik twierdzil wiec "TAK, sa polaczone" z deklarowana
pewnoscia 100% dla pary encji, ktorej nigdy nie sprawdzil -- niezaleznie od
zawartosci grafu. To bylo fabrykowanie twierdzenia, nie nieaktualny docstring.
"""
import numpy as np
import pytest

from src.qa_engine.hyper_symbolic_qa import HyperSymbolicQAEngine

# Odpowiedzi silnika sa po polsku; asercje musza uzywac dokladnie tych form.
ODMOWA = "Nie mogę rozstrzygnąć"
POWOD_ZAKRES = "poza zakresem"
POWOD_BRAK = "brak zbudowanego domknięcia"


def _pytanie(u: int, v: int) -> str:
    return f"Czy encja {u} jest połączona z encją {v}?"


# Maly, jawny graf skierowany: 0 -> 1 osiagalne, encja 2 izolowana.
CLOSURE = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]], dtype=np.int32)


def _engine(matrix=CLOSURE):
    return HyperSymbolicQAEngine(clifford_closure_matrix=matrix)


@pytest.mark.parametrize("u,v", [(5, 1), (0, 9), (3, 3), (100, 200)])
def test_out_of_range_query_is_not_answered_affirmatively(u: int, v: int):
    """Indeks poza ksztaltem macierzy -> odmowa rozstrzygniecia, nigdy "TAK"."""
    answer = _engine().ask(_pytanie(u, v))

    assert "TAK" not in answer, "Silnik potwierdza relacje, ktorej nie sprawdzil: " + answer
    assert "Gwarancja" not in answer, "Silnik deklaruje pewnosc bez sprawdzenia: " + answer
    assert ODMOWA in answer, "Brak odmowy rozstrzygniecia: " + answer
    assert POWOD_ZAKRES in answer, "Brak podanego powodu: " + answer


def test_missing_closure_matrix_is_not_answered_affirmatively():
    """Brak macierzy domkniecia -> odmowa rozstrzygniecia z podanym powodem."""
    answer = HyperSymbolicQAEngine(clifford_closure_matrix=None).ask(_pytanie(0, 1))

    assert "TAK" not in answer, "Silnik potwierdza relacje bez macierzy domkniecia: " + answer
    assert ODMOWA in answer, answer
    assert POWOD_BRAK in answer, answer


@pytest.mark.parametrize("u,v", [(i, j) for i in range(3) for j in range(3)])
def test_in_range_answer_agrees_with_closure_matrix(u: int, v: int):
    """W zakresie: odpowiedz zgadza sie z macierza dla kazdej pary, w obie strony."""
    answer = _engine().ask(_pytanie(u, v))
    connected = bool(CLOSURE[u, v] > 0)

    if connected:
        assert "TAK" in answer, f"Para ({u},{v}) polaczona, a odpowiedz to: {answer}"
        assert "NIE" not in answer
    else:
        assert "NIE" in answer, f"Para ({u},{v}) niepolaczona, a odpowiedz to: {answer}"
        assert "TAK" not in answer


def test_asymmetry_is_preserved():
    """Domkniecie jest skierowane: 0->1 tak, 1->0 nie. Odpowiedzi musza sie roznic."""
    assert "TAK" in _engine().ask(_pytanie(0, 1))
    assert "NIE" in _engine().ask(_pytanie(1, 0))
