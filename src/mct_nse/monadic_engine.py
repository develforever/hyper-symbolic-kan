import numpy as np
from typing import TypeVar, Generic, Callable, Tuple, List, Any

S = TypeVar('S') # State Type
A = TypeVar('A') # Input/Intermediate Value Type
B = TypeVar('B') # Output Value Type

class State(Generic[S, A]):
    r"""
    Monada Stanu (State Monad) w paradygmacie kategorycznym (Category-Theoretic Monad).
    
    Formalnie: Endofunktor M: C -> C wyposażony w transformacje naturalne η: Id -> M (pure)
    oraz μ: M^2 -> M (join/bind), gdzie M(A) = S -> (A, S).
    """
    def __init__(self, run: Callable[[S], Tuple[A, S]]):
        self.run = run

    @staticmethod
    def pure(a: A) -> 'State[S, A]':
        """Jednostka monadyczna η (unit / return)."""
        return State(lambda s: (a, s))

    def bind(self, f: Callable[[A], 'State[S, B]']) -> 'State[S, B]':
        """
        Mnożenie monadyczne μ (bind / >>=).
        Składa obliczenie stanowe w sposób czysty (Kleisli Composition).
        """
        def chained_run(s: S) -> Tuple[B, S]:
            val, next_state = self.run(s)
            return f(val).run(next_state)
        return State(chained_run)

    def map(self, f: Callable[[A], B]) -> 'State[S, B]':
        """Funktorowe przekształcenie wartości wewnątrz monady."""
        return self.bind(lambda a: State.pure(f(a)))

    @staticmethod
    def get() -> 'State[S, S]':
        """Pobiera aktualny stan s -> (s, s)."""
        return State(lambda s: (s, s))

    @staticmethod
    def put(s_new: S) -> 'State[S, None]':
        """Zastępuje aktualny stan nową wartością s -> (None, s_new)."""
        return State(lambda s: (None, s_new))

    @staticmethod
    def modify(f: Callable[[S], S]) -> 'State[S, None]':
        """Modyfikuje stan za pomocą funkcji f: s -> (None, f(s))."""
        return State(lambda s: (None, f(s)))

class KleisliArrow(Generic[S, A, B]):
    r"""
    Strzałka Kleisliego A -> State[S, B] stanowiąca morfizm w Katerogii Kleisliego C_M.
    """
    def __init__(self, fn: Callable[[A], State[S, B]]):
        self.fn = fn

    def __call__(self, a: A) -> State[S, B]:
        return self.fn(a)

    def compose(self, next_arrow: 'KleisliArrow[S, B, Any]') -> 'KleisliArrow[S, A, Any]':
        """Składanie strzałek Kleisliego (Kleisli Composition >=>)."""
        return KleisliArrow(lambda a: self.fn(a).bind(next_arrow.fn))

class MonadicEngine(Generic[S]):
    r"""
    Monadyczny Silnik Obliczeniowy MCT-NSE.
    
    Deterministyczne, bezgradientowe składanie obliczeń stanowych w monadzie stanu.
    Determinizm wynika z braku RNG w ścieżce wykonania — nie jest weryfikowany testem.
    """
    def __init__(self, initial_state: S):
        self.current_state = initial_state

    def execute(self, computation: State[S, A]) -> Tuple[A, S]:
        """Uruchamia obliczenie monadowe od aktualnego stanu."""
        val, new_state = computation.run(self.current_state)
        self.current_state = new_state
        return val, new_state
