import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.tdff_net.serializer import KANSerializer

def export_showcase_field():
    print("[+] Trening demonstracyjnego pola 3D KAN (0 epok gradientowych - analityczny ALS)...")
    
    # 3D pole: kombinacja sferycznych przeszkód i nieliniowych fal potencjału
    model = TDFFNet(spatial_dim=3, rank=8, degree=5)
    solver = ClosedFormALSSolver(alpha=1e-4, max_als_iters=8)
    
    # Generowanie siatki próbek
    N = 3000
    X = np.random.uniform(-0.95, 0.95, (N, 3))
    
    # Kształt celu: dwa centra odpychające (przeszkody) + falowa struktura KAN
    obs1 = np.array([-0.35, 0.2, 0.0])
    obs2 = np.array([0.4, -0.25, 0.1])
    
    d1 = np.linalg.norm(X - obs1, axis=1)
    d2 = np.linalg.norm(X - obs2, axis=1)
    
    # SDF / Potential target: wysoki potencjał w pobliżu przeszkód, niski na obrzeżach
    Y = np.exp(-d1**2 / 0.12) + 0.8 * np.exp(-d2**2 / 0.15) + 0.25 * np.cos(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    
    mse = solver.fit(model, X, Y)
    print(f"[+] Dopasowano model z błędem MSE = {mse:.6f}")
    
    output_path = os.path.join(os.path.dirname(__file__), "web_showcase", "src", "data", "initial_kan_weights.json")
    KANSerializer.save_json(model, output_path)
    print(f"[+] Zapisano wagi WebGPU do: {output_path}")

if __name__ == "__main__":
    export_showcase_field()
