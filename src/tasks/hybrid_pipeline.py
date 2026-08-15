import time
import numpy as np
from typing import Dict, List, Tuple

from src.hs_ckan.clifford_algebra import CliffordAlgebraEngine
from src.hs_ckan.nary_spatiotemporal import NarySpatioTemporalEngine
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.mct_nse.monadic_engine import State, KleisliArrow, MonadicEngine
from src.mct_nse.category_filter import CategoryFilter

class HybridPipelineTask:
    r"""
    Zintegrowane Zadanie Benchmarkowe Hybrydowego Pipeline'u End-to-End (Faza 4).
    
    Architektura (0 Epok Gradientowych):
    1. HS-CKAN Engine: Relacyjne wnioskowanie $N$-argumentowe wyznaczające docelową strefę bezpieczną.
    2. TDFF-Net Geometry Field: Odczyt ciagłego pola przeszkód $f(\mathbf{x})$ oraz analitycznego gradientu $\nabla f(\mathbf{x})$.
    3. MCT-NSE Monadic Guard: Zapewnia 100% deterministyczną filtrację stanów w monadzie stanu $S^* = \text{Fix}(\prod_i M_i)$.
    """
    def __init__(self, num_agents: int = 5, num_zones: int = 4, num_steps: int = 100):
        self.num_agents = num_agents
        self.num_zones = num_zones
        self.num_steps = num_steps
        
    def run_benchmark(self) -> Dict[str, float]:
        np.random.seed(42)
        
        # 1. Inicjalizacja HS-CKAN (Wnioskowanie Relacyjne)
        hs_engine = NarySpatioTemporalEngine(num_entities=max(self.num_agents, self.num_zones), num_predicates=2, kan_degree=4)
        
        # 2. Inicjalizacja TDFF-Net (Ciągłe Pole Geometrii & Gradient Analityczny)
        tdff_net = TDFFNet(spatial_dim=2, rank=16, degree=5)
        als_solver = ClosedFormALSSolver(alpha=1e-3, max_als_iters=5)
        
        # Geometria testowa: Pole odległości do przeszkody w centrum (0.0, 0.0)
        X_train = np.random.uniform(-1.0, 1.0, size=(1000, 2))
        Y_train = (np.linalg.norm(X_train, axis=1) - 0.4) # SDF sfery r=0.4
        
        t0_fit = time.perf_counter()
        als_solver.fit(tdff_net, X_train, Y_train)
        t_fit_ms = (time.perf_counter() - t0_fit) * 1000.0
        
        # 3. Inicjalizacja MCT-NSE (Monadowy Guard Bezpieczeństwa)
        filter_engine = CategoryFilter[np.ndarray, np.ndarray]()
        
        # Inwariant 1: Strefa Zakazana SDF >= 0.0 (poza sferą r=0.4)
        def sdf_safety_predicate(s: np.ndarray) -> bool:
            val = tdff_net.evaluate(s[:2].reshape(1, 2))
            return bool(val[0] >= -1e-7)

        def sdf_safety_morphism(s: np.ndarray) -> np.ndarray:
            val = tdff_net.evaluate(s[:2].reshape(1, 2))[0]
            grad = tdff_net.gradient(s[:2].reshape(1, 2))[0]
            s_corr = s.copy()
            if val < 0:
                g = grad
                g_norm = np.linalg.norm(g)
                if g_norm < 1e-6:
                    g = s[:2] / (np.linalg.norm(s[:2]) + 1e-6)
                    g_norm = 1.0
                # Odpychanie analitycznym gradientem pola tensorowego
                s_corr[:2] = s[:2] + (g / g_norm) * (abs(val) + 1e-3)
            return s_corr

        filter_engine.add_invariant("TDFF-SDF-Safety", sdf_safety_predicate, sdf_safety_morphism)
        
        # Inwariant 2: Limit Prędkości
        filter_engine.add_invariant(
            "Speed-Limit", 
            lambda s: bool(np.linalg.norm(s[2:]) <= 0.8 + 1e-7),
            lambda s: np.concatenate([s[:2], (s[2:] / (np.linalg.norm(s[2:]) + 1e-8)) * 0.7999]) if np.linalg.norm(s[2:]) > 0.8 else s
        )
        
        # Krok dynamiczny monady stanu
        def transition_step(action: np.ndarray) -> State[np.ndarray, np.ndarray]:
            def run_step(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                dt = 0.1
                new_v = s[2:] + action * dt
                new_p = s[:2] + new_v * dt
                next_s = np.concatenate([new_p, new_v])
                return next_s, next_s
            return State(run_step)

        raw_arrow = KleisliArrow(transition_step)
        guarded_arrow = filter_engine.guard_arrow(raw_arrow)
        
        # 4. Wykonanie Symulacji End-to-End
        start_state = np.array([-0.9, -0.9, 0.5, 0.5])
        current_state = start_state.copy()
        
        unfiltered_violations = 0
        filtered_violations = 0
        
        t0_exec = time.perf_counter()
        
        for step in range(self.num_steps):
            # Odczyt pola tensorowego i gradientu w punkcie agenta
            sdf_val = tdff_net.evaluate(current_state[:2].reshape(1, 2))[0]
            sdf_grad = tdff_net.gradient(current_state[:2].reshape(1, 2))[0]
            
            # Wektor sterowania: przyciąganie do celu (0.8, 0.8) + analityczne odpychanie gradientowe
            goal_dir = np.array([0.8, 0.8]) - current_state[:2]
            goal_dir /= (np.linalg.norm(goal_dir) + 1e-6)
            
            action = goal_dir * 2.0
            if sdf_val < 0.2:
                action += sdf_grad * 5.0 # Dodatkowa siła odpychająca z pola TDFF-Net
                
            # Weryfikacja bez kategorycznego guarded arrow
            _, raw_s = raw_arrow(action).run(current_state)
            _, raw_viols = filter_engine.filter_state(raw_s)
            if len(raw_viols) > 0:
                unfiltered_violations += 1
                
            # Wykonanie monadowe pod osłoną guarded arrow MCT-NSE
            _, current_state = guarded_arrow(action).run(current_state)
            _, post_viols = filter_engine.filter_state(current_state)
            if len(post_viols) > 0:
                filtered_violations += 1

        t_exec_elapsed = time.perf_counter() - t0_exec
        latency_per_step_ms = (t_exec_elapsed / self.num_steps) * 1000.0

        return {
            "num_steps": self.num_steps,
            "als_fit_time_ms": t_fit_ms,
            "pipeline_step_latency_ms": latency_per_step_ms,
            "unfiltered_violation_rate": (unfiltered_violations / self.num_steps) * 100.0,
            "mct_nse_violation_rate": (filtered_violations / self.num_steps) * 100.0
        }
