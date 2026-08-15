"""
Hyper-Symbolic Kolmogorov-Arnold Networks (Hyper-Symbolic KAN / TDFF-Net).

High-level facade API:
    import hyper_kan as hk
    # or
    import src as hk

    # Continuous Polyadic KAN Field
    field = hk.TensorField(spatial_dim=3, rank=16, degree=5)
    field.fit(X, y)
    y_pred = field(X)
    grads = field.gradient(X)

    # Tensor Train KAN Field for High-Dimensional Spaces (D >= 10)
    tt_field = hk.TensorTrainField(spatial_dim=20, degree=5)
    tt_field.fit_cross(target_fn)

    # Mesh-Free Spectral PDE Poisson Solver
    solver = hk.PoissonSolver(dim=3, degree=8)
    solver.solve(source_fn, boundary_fn)

    # Certified Robotics Control Barrier Function (CBF) Planner
    planner = hk.CBFPlanner()
    planner.add_obstacle_field(field)
    traj = planner.plan_trajectory(start, goal)
"""

import sys

from src.facade import (
    TensorField,
    TensorTrainField,
    PoissonSolver,
    CBFPlanner
)

# Register hyper_kan alias in sys.modules so `import hyper_kan as hk` works everywhere
sys.modules["hyper_kan"] = sys.modules[__name__]

__version__ = "0.1.0"

__all__ = [
    "TensorField",
    "TensorTrainField",
    "PoissonSolver",
    "CBFPlanner",
    "__version__"
]
