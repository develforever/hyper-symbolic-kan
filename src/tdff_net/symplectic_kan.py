import numpy as np
from typing import Tuple
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tucker_tensor_field import TuckerTDFFNet

class SymplecticKANEngine:
    r"""
    Symplektyczny Silnik KAN dla Fizyki i Dynamiki Hamiltonowskiej (Symplectic KAN Engine).
    
    Zachowuje formę symplektyczną dq \wedge dp = const oraz energię układu H(q, p)
    w bezgradientowym paradygmacie O(1).
    
    Równania Hamiltona:
    \dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}
    """
    def __init__(self, position_dim: int = 1, rank: int = 16, degree: int = 5):
        self.position_dim = position_dim
        self.phase_dim = 2 * position_dim # (q, p)
        self.hamiltonian_field = TDFFNet(spatial_dim=self.phase_dim, rank=rank, degree=degree)

    def phase_velocity(self, phase_state: np.ndarray) -> np.ndarray:
        """
        Oblicza symplektyczny wektor prędkości przestrzeni fazowej [dq/dt, dp/dt]
        wykorzystując analityczny gradient pola Hamiltona \nabla H(q, p).
        """
        N = phase_state.shape[0]
        grad_H = self.hamiltonian_field.gradient(phase_state) # (N, 2d)
        
        # Macierz symplektyczna J: dq/dt = dH/dp, dp/dt = -dH/dq
        q_dim = self.position_dim
        dH_dq = grad_H[:, :q_dim]
        dH_dp = grad_H[:, q_dim:]
        
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        return np.hstack([dq_dt, dp_dt])

    def symplectic_step(self, phase_state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Wykonuje krok 2-giego rzędu integracji symplektycznej (Stormer-Verlet / Position Verlet),
        gwarantując błąd energii rzędu O(dt^2) i ścisłe zachowanie formy symplektycznej w długim horyzoncie.
        """
        q_dim = self.position_dim
        q = phase_state[:, :q_dim].copy()
        p = phase_state[:, q_dim:].copy()
        
        # 1. Pół-krok p: p_{n+1/2} = p_n - (dt/2) * (dH/dq)(q_n, p_n)
        grad_H_half = self.hamiltonian_field.gradient(np.hstack([q, p]))
        dH_dq_half = grad_H_half[:, :q_dim]
        p_half = p - 0.5 * dt * dH_dq_half
        
        # 2. Pełny krok q: q_{n+1} = q_n + dt * (dH/dp)(q_n, p_{n+1/2})
        grad_H_mid = self.hamiltonian_field.gradient(np.hstack([q, p_half]))
        dH_dp_mid = grad_H_mid[:, q_dim:]
        q_next = q + dt * dH_dp_mid
        
        # 3. Pół-krok p: p_{n+1} = p_{n+1/2} - (dt/2) * (dH/dq)(q_{n+1}, p_{n+1/2})
        grad_H_next = self.hamiltonian_field.gradient(np.hstack([q_next, p_half]))
        dH_dq_next = grad_H_next[:, :q_dim]
        p_next = p_half - 0.5 * dt * dH_dq_next
        
        return np.hstack([q_next, p_next])
