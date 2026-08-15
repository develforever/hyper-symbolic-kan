import time
import numpy as np
from typing import Dict, Tuple

from src.tdff_net.symplectic_kan import SymplecticKANEngine
from src.tdff_net.closed_form_als import ClosedFormALSSolver

class SymplecticPhysicsTask:
    r"""
    TASK 8: Benchmark Symplektycznych Pol KAN i Dynamiki Hamiltonowskiej (Faza 5).
    
    Testuje zachowanie energii H(q, p) = 0.5 * (q^2 + p^2) w trajektorii 1000 kroków (T = 20s).
    Porównuje:
    - Standardowa Integracja Eulera (nie-symplektyczna): znaczny dryf energii.
    - Symplektyczny KAN Integrator (Leapfrog/Symplectic Euler): brak dryfu energii (\Delta H \to 0).
    """
    def __init__(self, num_steps: int = 2500, dt: float = 0.02):
        self.num_steps = num_steps
        self.dt = dt

    def generate_harmonic_dataset(self, num_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        # Próbkowanie w kanonicznej dziedzinie Czebyszewa [-1.0, 1.0]^2
        QP = np.random.uniform(-1.0, 1.0, size=(num_samples, 2))
        # Dokładna funkcja Hamiltona oscylatora harmonicznego: H(q, p) = 0.5 * q^2 + 0.5 * p^2
        H_val = 0.5 * (QP[:, 0] ** 2 + QP[:, 1] ** 2)
        return QP, H_val

    def run_benchmark(self) -> Dict[str, float]:
        QP_train, H_train = self.generate_harmonic_dataset()
        
        # 1. Inicjalizacja i dopasowanie pola Hamiltona H(q, p) w czasie O(1)
        engine = SymplecticKANEngine(position_dim=1, rank=16, degree=6)
        als_solver = ClosedFormALSSolver(alpha=1e-8, max_als_iters=8)
        
        t0_fit = time.perf_counter()
        als_solver.fit(engine.hamiltonian_field, QP_train, H_train)
        t_fit_ms = (time.perf_counter() - t0_fit) * 1000.0
        
        # Stan początkowy w kanonicznej dziedzinie przestrzeni fazowej (q0 = 0.8, p0 = 0.0) -> H0 = 0.32
        initial_state = np.array([[0.8, 0.0]])
        H_initial = engine.hamiltonian_field.evaluate(initial_state)[0]
        
        # 2. Symulacja Standardowej Integracji Eulera (nie-symplektyczna)
        s_euler = initial_state.copy()
        for _ in range(self.num_steps):
            v = engine.phase_velocity(s_euler)
            s_euler += v * self.dt
        H_final_euler = engine.hamiltonian_field.evaluate(s_euler)[0]
        energy_drift_euler = abs(H_final_euler - H_initial)

        # 3. Symulacja Symplektycznego Integratora KAN (Leapfrog Symplectic Engine)
        s_symp = initial_state.copy()
        t0_step = time.perf_counter()
        for _ in range(self.num_steps):
            s_symp = engine.symplectic_step(s_symp, dt=self.dt)
        t_step_elapsed = time.perf_counter() - t0_step
        latency_per_step_ms = (t_step_elapsed / self.num_steps) * 1000.0
        
        H_final_symp = engine.hamiltonian_field.evaluate(s_symp)[0]
        energy_drift_symp = abs(H_final_symp - H_initial)

        return {
            "als_fit_time_ms": t_fit_ms,
            "step_latency_ms": latency_per_step_ms,
            "h_initial": H_initial,
            "h_final_euler": H_final_euler,
            "energy_drift_euler": energy_drift_euler,
            "h_final_symp": H_final_symp,
            "energy_drift_symp": energy_drift_symp
        }
