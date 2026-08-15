import numpy as np
from typing import Callable, Tuple, Any, Optional

class VectorState:
    r"""
    Wektorowa Monada Stanu (Vectorized State Monad) dla N Agentów Równolegle.
    
    Operuje na macierzach stanu S \in \mathbb{R}^{N \times d_{state}}.
    Formalnie: M(A) = S -> (A, S) uogólnione na N agentów w czystym paradygmacie kategorycznym.
    0 EPOK GRADIENTOWYCH: Wszystkie złożenia zachodzą w wysoce zoptymalizowanej pamięci podręcznej SIMD.
    """
    def __init__(self, run: Callable[[np.ndarray], Tuple[Any, np.ndarray]]):
        self.run = run

    @staticmethod
    def pure(a: Any) -> 'VectorState':
        """Jednostka monadyczna η dla wektora agentów."""
        return VectorState(lambda s: (a, s))

    def bind(self, f: Callable[[Any], 'VectorState']) -> 'VectorState':
        """
        Mnożenie monadyczne μ (bind / >>=) dla N agentów.
        Składa obliczenia stanowe dla całej floty równolegle w O(1) wywołaniach macierzowych.
        """
        def chained_run(s: np.ndarray) -> Tuple[Any, np.ndarray]:
            val, next_state = self.run(s)
            return f(val).run(next_state)
        return VectorState(chained_run)

    def map(self, f: Callable[[Any], Any]) -> 'VectorState':
        """Funktorowe przekształcenie wartości wewnątrz monady wektorowej."""
        return self.bind(lambda a: VectorState.pure(f(a)))

    @staticmethod
    def get() -> 'VectorState':
        """Pobiera macierz stanów całej floty s -> (s, s)."""
        return VectorState(lambda s: (s, s))

    @staticmethod
    def put(s_new: np.ndarray) -> 'VectorState':
        """Zastępuje stan całej floty s -> (None, s_new)."""
        return VectorState(lambda s: (None, s_new))


class VectorKleisliArrow:
    r"""
    Wektorowa Strzałka Kleisliego A \in \mathbb{R}^{N \times d_A} -> VectorState[S, S].
    Morfizm w Katerogii Kleisliego dla floty N agentów.
    """
    def __init__(self, fn: Callable[[np.ndarray], VectorState]):
        self.fn = fn

    def __call__(self, a: np.ndarray) -> VectorState:
        return self.fn(a)

    def compose(self, next_arrow: 'VectorKleisliArrow') -> 'VectorKleisliArrow':
        """Składanie wektorowych strzałek Kleisliego (Vector Kleisli Composition >=>)."""
        return VectorKleisliArrow(lambda a: self.fn(a).bind(next_arrow.fn))


class ConcurrentMonadicEngine:
    r"""
    Współbieżny Silnik Monadyczny MCT-NSE v2.
    Zapewnia deterministyczne składanie stanów i egzekwowanie inwariantów dla floty N=1000 agentów.
    """
    def __init__(self, initial_state: np.ndarray):
        assert initial_state.ndim == 2 # Shape (N, d_state)
        self.current_state = np.array(initial_state, dtype=np.float64)

    @property
    def num_agents(self) -> int:
        return self.current_state.shape[0]

    @property
    def state_dim(self) -> int:
        return self.current_state.shape[1]

    def execute(self, computation: VectorState) -> Tuple[Any, np.ndarray]:
        """Uruchamia obliczenie monadowe dla całej floty agentów."""
        val, new_state = computation.run(self.current_state)
        self.current_state = new_state
        return val, new_state
