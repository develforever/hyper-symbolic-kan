import numpy as np
from typing import List, Tuple, Dict, Callable
from src.mct_nse.concurrent_monadic_engine import VectorState, VectorKleisliArrow

class VectorCategoryInvariant:
    r"""
    Wektorowy Inwariant Kategoryczny dla Floty N Agentów.
    Predicate: S \in \mathbb{R}^{N \times d} -> Mask \in \{True, False\}^N
    Correction Morphism: S \in \mathbb{R}^{N \times d} -> S' \in \mathbb{R}^{N \times d}
    """
    def __init__(
        self,
        name: str,
        predicate: Callable[[np.ndarray], np.ndarray],
        correction_morphism: Callable[[np.ndarray], np.ndarray]
    ):
        self.name = name
        self.predicate = predicate
        self.correction_morphism = correction_morphism

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        return self.predicate(state)

    def enforce(self, state: np.ndarray) -> np.ndarray:
        valid_mask = self.predicate(state)
        if np.all(valid_mask):
            return state
        return self.correction_morphism(state)


class ConcurrentCategoryFilter:
    r"""
    Współbieżny i Wektorowy Filtr Reguł Kategorycznych (MCT-NSE v2 Formal Category Guard).
    
    Opakowuje wektorowe przejścia stanowe całej floty w chronioną strzałkę Kleisliego
    i iteruje morfizmy naprawcze do punktu stałego.

    Zakres gwarancji: `filter_state` wykonuje najwyżej `max_iters` przebiegów i zwraca
    stan taki, jaki osiągnie — bez sygnalizacji, że punkt stały nie został osiągnięty.
    Spełnienie inwariantów jest zagwarantowane tylko dla morfizmów idempotentnych
    i wzajemnie nieinterferujących; weryfikuje to
    `tests/test_safety_and_nary.py::test_concurrent_category_filter_zero_violations`
    dla pojedynczego inwariantu box-bounds przy N=500. Brak testu dla inwariantów
    interferujących; latencja na agenta nie jest mierzona w testach.
    """
    def __init__(self):
        self.invariants: List[VectorCategoryInvariant] = []

    def add_invariant(
        self,
        name: str,
        predicate: Callable[[np.ndarray], np.ndarray],
        correction_morphism: Callable[[np.ndarray], np.ndarray]
    ):
        self.invariants.append(VectorCategoryInvariant(name, predicate, correction_morphism))

    def filter_state(self, state: np.ndarray, max_iters: int = 5) -> Tuple[np.ndarray, Dict[str, int]]:
        r"""
        Wektorowa pętla punktu stałego Fixpoint S* = Fix(\prod M_i) dla całej floty N agentów.
        """
        current_s = state.copy()
        violation_counts: Dict[str, int] = {inv.name: 0 for inv in self.invariants}
        
        for _ in range(max_iters):
            changed = False
            for inv in self.invariants:
                valid_mask = inv.evaluate(current_s)
                num_invalid = int(np.sum(~valid_mask))
                if num_invalid > 0:
                    violation_counts[inv.name] += num_invalid
                    current_s = inv.enforce(current_s)
                    changed = True
            if not changed:
                break
                
        return current_s, violation_counts

    def guard_arrow(self, arrow: VectorKleisliArrow) -> VectorKleisliArrow:
        """
        Przekształca surową wektorową strzałkę Kleisliego w strzałkę chronioną (Guarded Vector Arrow).
        """
        def guarded_fn(a: np.ndarray) -> VectorState:
            def run_guarded(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                _, raw_next_state = arrow(a).run(s)
                safe_state, _ = self.filter_state(raw_next_state)
                return safe_state, safe_state
            return VectorState(run_guarded)
        return VectorKleisliArrow(guarded_fn)
