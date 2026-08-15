"""
Applications & Industrial Deployment Modules for Hyper-Symbolic KAN.

Modules:
- robotics_cbf_planner: Control Barrier Functions (CBF), HOCBF, and collision-free trajectory planning.
- pde_poisson_solver: Mesh-free PDE Poisson/Laplace solver with analytical 2nd Chebyshev derivatives in 0 epochs.
"""

from src.applications.robotics_cbf_planner import (
    CBFConfig,
    ContinuousKANObstacleField,
    CBFPlanner,
    InterAgentCBF,
    DomainBoxCBF
)

from src.applications.pde_poisson_solver import (
    chebyshev_derivatives_2nd,
    SpectralKANPoissonSolver,
    TTPoissonSolver,
    PoissonAnalyticalSolution
)

__all__ = [
    "CBFConfig",
    "ContinuousKANObstacleField",
    "CBFPlanner",
    "InterAgentCBF",
    "DomainBoxCBF",
    "chebyshev_derivatives_2nd",
    "SpectralKANPoissonSolver",
    "TTPoissonSolver",
    "PoissonAnalyticalSolution"
]
