import numpy as np
from typing import TypeVar, Generic, Callable, List, Tuple, Dict, Optional
from src.mct_nse.monadic_engine import State, KleisliArrow

S = TypeVar('S') # State Type
A = TypeVar('A') # Action / Proposed Transition Type

class CategoryInvariant(Generic[S]):
    r"""
    Formalny Inwariant Kategoryczny P: S -> Bool z przypisanym morfizmem naprawczym M: S -> S.
    """
    def __init__(self, name: str, predicate: Callable[[S], bool], correction_morphism: Callable[[S], S]):
        self.name = name
        self.predicate = predicate
        self.correction_morphism = correction_morphism

    def evaluate(self, state: S) -> bool:
        return self.predicate(state)

    def enforce(self, state: S) -> S:
        if not self.predicate(state):
            return self.correction_morphism(state)
        return state

class CategoryFilter(Generic[S, A]):
    r"""
    Kategoryczny Filtr Reguł Formalnych (Formal Category Guard).
    
    Opakowuje przejścia stanowe w obostrzenie Kleisliego (Guarded Kleisli Arrow)
    i iteruje morfizmy naprawcze do punktu stałego.

    Zakres gwarancji: `filter_state` wykonuje najwyżej `max_iters` przebiegów i zwraca
    stan taki, jaki osiągnie — bez sygnalizacji, że punkt stały nie został osiągnięty.
    Spełnienie inwariantów jest więc zagwarantowane tylko dla morfizmów naprawczych,
    które są idempotentne i wzajemnie nieinterferujące. Weryfikuje to
    `tests/test_safety_and_nary.py::test_concurrent_category_filter_zero_violations`
    dla pojedynczego inwariantu box-bounds. Brak testu dla inwariantów interferujących.
    """
    def __init__(self):
        self.invariants: List[CategoryInvariant[S]] = []

    def add_invariant(self, name: str, predicate: Callable[[S], bool], correction_morphism: Callable[[S], S]):
        self.invariants.append(CategoryInvariant(name, predicate, correction_morphism))

    def filter_state(self, state: S, max_iters: int = 5) -> Tuple[S, List[str]]:
        """
        Sprawdza wszystkie inwarianty kategoryczne i stosuje morfizmy naprawcze do osadzenia
        stanu w punkcie stałym (Fixpoint) kategorii stanów dopuszczalnych.
        Zwraca deterministycznie bezpieczny stan oraz listę zmodyfikowanych inwariantów.
        """
        violations = []
        current_s = state
        for _ in range(max_iters):
            changed = False
            for inv in self.invariants:
                if not inv.evaluate(current_s):
                    if inv.name not in violations:
                        violations.append(inv.name)
                    current_s = inv.enforce(current_s)
                    changed = True
            if not changed:
                break
        return current_s, violations

    def guard_arrow(self, arrow: KleisliArrow[S, A, S]) -> KleisliArrow[S, A, S]:
        """
        Przekształca surową strzałkę Kleisliego A -> State[S, S] w strzałkę chronioną (Guarded Arrow).
        """
        def guarded_fn(a: A) -> State[S, S]:
            def run_guarded(s: S) -> Tuple[S, S]:
                # Execution of raw monadic step
                _, raw_next_state = arrow(a).run(s)
                # Enforce formal category invariants
                safe_state, _ = self.filter_state(raw_next_state)
                return safe_state, safe_state
            return State(run_guarded)
        return KleisliArrow(guarded_fn)
