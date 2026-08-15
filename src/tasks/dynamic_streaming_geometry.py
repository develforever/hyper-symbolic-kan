import time
import numpy as np
from typing import Dict, Tuple

from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.tdff_net.streaming_als import StreamingALSSolver

class DynamicStreamingGeometryTask:
    r"""
    TASK 10: Benchmark Strumieniowej Adaptacji Online RLS-ALS (Concept Drift) w KAN (Faza 7).
    
    Testuje bezgradientowe śledzenie w czasie rzeczywistym ruchomej i deformującej się przeszkody
    f(x, t) = ||x - c(t)||_2 - R(t) bez konieczności ponownego uczenia i bez buforowania danych.
    """
    def __init__(self, num_initial_samples: int = 2000, streaming_steps: int = 500):
        self.num_initial_samples = num_initial_samples
        self.streaming_steps = streaming_steps

    def obstacle_center_and_radius(self, t_step: int) -> Tuple[np.ndarray, float]:
        t = t_step * 0.02
        c = np.array([0.3 * np.sin(t), 0.3 * np.cos(t) - 0.3, 0.0])
        r = 0.4 + 0.1 * np.sin(2.0 * t)
        return c, r

    def run_benchmark(self) -> Dict[str, float]:
        np.random.seed(42)
        # 1. Początkowe dopasowanie pola KAN dla t=0
        c0, r0 = self.obstacle_center_and_radius(0)
        X_init = np.random.uniform(-0.8, 0.8, size=(self.num_initial_samples, 3))
        Y_init = np.linalg.norm(X_init - c0, axis=1) - r0
        
        model = TDFFNet(spatial_dim=3, rank=20, degree=5)
        batch_solver = ClosedFormALSSolver(alpha=1e-4, max_als_iters=8)
        
        t0_batch = time.perf_counter()
        batch_solver.fit(model, X_init, Y_init)
        t_batch_fit_ms = (time.perf_counter() - t0_batch) * 1000.0
        
        # 2. Inicjalizacja Strumieniowego Solvera RLS-ALS
        streaming_solver = StreamingALSSolver(model, forget_factor=0.992, initial_covariance=50.0)
        
        # Sety walidacyjne
        X_val = np.random.uniform(-0.8, 0.8, size=(1000, 3))
        
        # Błąd a priori przed adaptacją online dla t=streaming_steps
        c_final, r_final = self.obstacle_center_and_radius(self.streaming_steps)
        Y_val_final = np.linalg.norm(X_val - c_final, axis=1) - r_final
        rmse_before_streaming = float(np.sqrt(np.mean((Y_val_final - model.evaluate(X_val)) ** 2)))
        
        # 3. Pętla Strumieniowej Adaptacji Online (Concept Drift)
        t0_stream = time.perf_counter()
        streaming_latencies = []
        
        for step in range(1, self.streaming_steps + 1):
            c_t, r_t = self.obstacle_center_and_radius(step)
            # Pomiar strumieniowy z czujnika (1 punkt)
            x_stream = np.random.uniform(-0.8, 0.8, size=(1, 3))
            y_stream = float(np.linalg.norm(x_stream - c_t) - r_t)
            
            t0_single = time.perf_counter()
            streaming_solver.update_online(x_stream[0], y_stream)
            streaming_latencies.append((time.perf_counter() - t0_single) * 1000.0)
            
        t_total_stream_ms = (time.perf_counter() - t0_stream) * 1000.0
        avg_step_latency_ms = float(np.mean(streaming_latencies))
        
        # Błąd walidacyjny po adapacji strumieniowej RLS-ALS
        rmse_after_streaming = float(np.sqrt(np.mean((Y_val_final - model.evaluate(X_val)) ** 2)))
        
        return {
            "batch_fit_time_ms": t_batch_fit_ms,
            "total_stream_time_ms": t_total_stream_ms,
            "avg_step_latency_ms": avg_step_latency_ms,
            "rmse_before_streaming": rmse_before_streaming,
            "rmse_after_streaming": rmse_after_streaming,
            "rmse_improvement_pct": ((rmse_before_streaming - rmse_after_streaming) / rmse_before_streaming) * 100.0
        }
