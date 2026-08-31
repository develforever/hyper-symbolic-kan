import time
import numpy as np
from typing import Dict, List, Tuple
from src.mct_nse.monadic_engine import State, KleisliArrow, MonadicEngine
from src.mct_nse.category_filter import CategoryFilter

class FormalVerificationTask:
    r"""
    Benchmark Weryfikacji Formalnej Zasad Bezpieczeństwa dla Architektury MCT-NSE.
    
    Testuje sterowanie neuro-symboliczne w środowisku z krytycznymi ograniczeniami
    przestrzennymi (Strefa Zakazana / No-Fly Zone), limitem prędkości oraz granicami przestrzeni stanów.
    
    Porównuje:
    - Surowy KAN / Neural Predictor (brak filtracji): generuje naruszenia zasad.
    - MCT-NSE Monadic Category Filter: deterministyczna filtracja stanów; zmierzony
      wskaźnik naruszeń po filtracji raportowany jako `mct_nse_violation_rate`
      (nie jest gwarancją a priori — patrz `CategoryFilter.filter_state`, max_iters).
    """
    def __init__(self, num_episodes: int = 20, steps_per_episode: int = 50):
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        
        # Definicja Strefy Zakazanej (No-Fly Zone)
        self.obstacle_center = np.array([0.3, 0.3])
        self.obstacle_radius = 0.35
        self.max_velocity = 0.8
        
    def _create_category_filter(self) -> CategoryFilter[np.ndarray, np.ndarray]:
        filter_engine = CategoryFilter[np.ndarray, np.ndarray]()
        
        # 1. Inwariant Strefy Zakazanej (Spatial Safety Boundary)
        def obstacle_free_predicate(s: np.ndarray) -> bool:
            dist = np.linalg.norm(s[:2] - self.obstacle_center)
            return dist >= (self.obstacle_radius - 1e-7)
            
        def obstacle_projection_morphism(s: np.ndarray) -> np.ndarray:
            s_corr = s.copy()
            diff = s[:2] - self.obstacle_center
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.array([1.0, 0.0])
                dist = 1.0
            # Rzutowanie na zewnętrzną granicę sfery bezpieczeństwa
            s_corr[:2] = self.obstacle_center + (diff / dist) * (self.obstacle_radius + 1e-4)
            # Wyzerowanie składowej prędkości wchodzącej w przeszkodę
            normal = diff / dist
            v = s_corr[2:]
            v_normal = np.dot(v, normal)
            if v_normal < 0:
                s_corr[2:] = v - v_normal * normal
            return s_corr

        filter_engine.add_invariant("No-Fly-Zone", obstacle_free_predicate, obstacle_projection_morphism)
        
        # 2. Inwariant Maksymalnej Prędkości (Speed Limit Boundary)
        def speed_limit_predicate(s: np.ndarray) -> bool:
            v_mag = np.linalg.norm(s[2:])
            return v_mag <= (self.max_velocity + 1e-7)
            
        def speed_limit_morphism(s: np.ndarray) -> np.ndarray:
            s_corr = s.copy()
            v_mag = np.linalg.norm(s[2:])
            if v_mag > self.max_velocity:
                s_corr[2:] = (s[2:] / v_mag) * (self.max_velocity - 1e-7)
            return s_corr

        filter_engine.add_invariant("Max-Velocity", speed_limit_predicate, speed_limit_morphism)

        # 3. Inwariant Granicy Dziedziny Przestrzennej (Bounding Box [-1, 1])
        def domain_predicate(s: np.ndarray) -> bool:
            return bool(np.all(np.abs(s[:2]) <= 1.0 + 1e-7))

        def domain_morphism(s: np.ndarray) -> np.ndarray:
            s_corr = s.copy()
            s_corr[:2] = np.clip(s_corr[:2], -1.0, 1.0)
            return s_corr

        filter_engine.add_invariant("Domain-Boundary", domain_predicate, domain_morphism)

        return filter_engine

    def run_benchmark(self) -> Dict[str, float]:
        np.random.seed(42)
        category_filter = self._create_category_filter()
        
        # Surowa strzałka Kleisliego realizująca prosty krok dynamiczny na podstawie akcji a
        def raw_transition_step(action: np.ndarray) -> State[np.ndarray, np.ndarray]:
            def step_fn(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                dt = 0.1
                # Akcja reprezentuje przyspieszenie (dvx, dvy)
                new_v = s[2:] + action * dt
                new_pos = s[:2] + new_v * dt
                next_s = np.concatenate([new_pos, new_v])
                return next_s, next_s
            return State(step_fn)

        raw_arrow = KleisliArrow(raw_transition_step)
        guarded_arrow = category_filter.guard_arrow(raw_arrow)
        
        total_steps = self.num_episodes * self.steps_per_episode
        unfiltered_violations = 0
        filtered_violations = 0
        
        t_start = time.perf_counter()
        
        for ep in range(self.num_episodes):
            # Stan początkowy: x, y, vx, vy
            start_state = np.array([-0.8, -0.8, 0.5, 0.5])
            
            s_raw = start_state.copy()
            s_guarded = start_state.copy()
            
            for step in range(self.steps_per_episode):
                # Generowanie akcji zmierzającej w stronę celu prnącej przez przeszkodę (0.3, 0.3)
                goal = np.array([0.8, 0.8])
                desired_dir = (goal - s_raw[:2])
                desired_dir /= (np.linalg.norm(desired_dir) + 1e-6)
                
                # Dodanie szumu do sterowania dla symulacji niepewności modelu neuro-symbolicznego
                action = desired_dir * 3.0 + np.random.normal(0, 0.5, size=2)
                
                # 1. Wykonanie kroków bez filtracji
                _, s_raw = raw_arrow(action).run(s_raw)
                _, raw_violations = category_filter.filter_state(s_raw)
                if len(raw_violations) > 0:
                    unfiltered_violations += 1

                # 2. Wykonanie kroków z monadowym guarded arrow MCT-NSE
                _, s_guarded = guarded_arrow(action).run(s_guarded)
                _, post_violations = category_filter.filter_state(s_guarded)
                if len(post_violations) > 0:
                    filtered_violations += 1

        t_elapsed = time.perf_counter() - t_start
        latency_per_step_ms = (t_elapsed / total_steps) * 1000.0
        
        unfiltered_violation_rate = (unfiltered_violations / total_steps) * 100.0
        filtered_violation_rate = (filtered_violations / total_steps) * 100.0
        
        return {
            "total_steps": total_steps,
            "unfiltered_violation_rate": unfiltered_violation_rate,
            "mct_nse_violation_rate": filtered_violation_rate,
            "latency_per_step_ms": latency_per_step_ms
        }
