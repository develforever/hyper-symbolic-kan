r"""
High-Level Ergonomic Facade API for Hyper-Symbolic KAN.

Exposes user-friendly classes:
- hk.TensorField: Continuous Polyadic KAN Field with 0-epoch Closed-Form ALS.
- hk.TensorTrainField: High-dimensional continuous field (TT-KAN) with TT-Cross.
- hk.PoissonSolver: Mesh-free spectral PDE solver for Poisson & Laplace equations.
- hk.CBFPlanner: Robotics Control Barrier Function (CBF) collision-free planner.
"""

from typing import Optional, List, Tuple, Union, Callable, Dict, Any, Sequence
import numpy as np
import pathlib
import json

from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tt_kan import TensorTrainKAN
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.tdff_net.tt_cross import TTCrossSolver
from src.applications.pde_poisson_solver import SpectralKANPoissonSolver, TTPoissonSolver
from src.applications.robotics_cbf_planner import (
    CBFPlanner as _CBFPlannerBase,
    CBFConfig,
    ContinuousKANObstacleField
)

# Optional PyTorch and JAX ecosystem bridges
try:
    import torch
    from src.torch_kan.layers import ContinuousKANLayer, TensorTrainKANLayer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    from src.jax_kan.layers import ContinuousKANJAXLayer, TensorTrainKANJAXLayer
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


class TensorField:
    r"""
    High-Level Ergonomic Interface for Continuous Polyadic KAN Fields (TDFF-Net).
    
    Represents an implicit continuous functional field:
    f(x_1, ..., x_D) = \sum_{r=1}^R \lambda_r \prod_{d=1}^D \left( \sum_{k=0}^K W_{r, k}^{(d)} T_k(x_d) \right)
    
    Features:
    - 0-epoch instantaneous fitting via Closed-Form ALS (`.fit()`).
    - Vectorized NumPy inference (`.predict()`, `__call__()`). This facade does NOT
      use `FastCPPKANEngine`: `predict()` calls `TDFFNet.evaluate()`, i.e. plain
      NumPy. Measured C++ engine throughput lives separately in
      `benchmarks/test_kernel_benchmarks.py` (pytest-benchmark, not collected by
      `pytest tests/`).
    - Exact analytical spatial gradients $\nabla f(X)$ (`.gradient()`).
    - Weight export to PyTorch (`.to_torch()`) and JAX (`.to_jax()`) layers. The
      conversion copies the weights: `torch.from_numpy(...).to(dtype, device)`
      followed by `copy_()`.
    """
    def __init__(
        self,
        spatial_dim: int = 3,
        rank: int = 16,
        degree: int = 5,
        model: Optional[TDFFNet] = None
    ):
        self.spatial_dim = spatial_dim
        self.rank = rank
        self.degree = degree
        if model is not None:
            self._model = model
            self.spatial_dim = model.spatial_dim
            self.rank = model.rank
            self.degree = model.degree
        else:
            self._model = TDFFNet(spatial_dim=spatial_dim, rank=rank, degree=degree)

    @property
    def lambdas(self) -> np.ndarray:
        return self._model.lambdas

    @property
    def factors(self) -> List[np.ndarray]:
        return self._model.factors

    def fit(
        self,
        X: Union[np.ndarray, "torch.Tensor", "jnp.ndarray"],
        y: Union[np.ndarray, "torch.Tensor", "jnp.ndarray"],
        alpha: float = 1e-4,
        max_iters: int = 10
    ) -> "TensorField":
        """
        Fits field parameters instantaneously in 0 epochs using Closed-Form ALS.
        X: (N, spatial_dim)
        y: (N,) or (N, 1)
        """
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        elif hasattr(X, "__array__"):
            X = np.asarray(X)
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        elif hasattr(y, "__array__"):
            y = np.asarray(y)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        solver = ClosedFormALSSolver(alpha=alpha, max_als_iters=max_iters)
        solver.fit(self._model, X, y)
        return self

    def fit_als(self, X: np.ndarray, y: np.ndarray, alpha: float = 1e-4, max_iters: int = 10) -> "TensorField":
        return self.fit(X, y, alpha=alpha, max_iters=max_iters)

    def predict(self, X: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Evaluates field predictions at points X."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1 and len(X_arr) == self.spatial_dim:
            X_arr = X_arr[None, :]
            return self._model.evaluate(X_arr)[0]
        return self._model.evaluate(X_arr)

    def __call__(self, X: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        return self.predict(X)

    def gradient(self, X: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Computes exact analytical gradient vectors \nabla f(X)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1 and len(X_arr) == self.spatial_dim:
            X_arr = X_arr[None, :]
            return self._model.gradient(X_arr)[0]
        return self._model.gradient(X_arr)

    def save(self, file_path: Union[str, pathlib.Path]) -> None:
        """Saves model parameters to JSON format."""
        data = {
            "type": "TDFFNet_CP_KAN",
            "spatial_dim": self.spatial_dim,
            "rank": self.rank,
            "degree": self.degree,
            "lambdas": self.lambdas.tolist(),
            "factors": [f.tolist() for f in self.factors]
        }
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, pathlib.Path]) -> "TensorField":
        """Loads model parameters from JSON format."""
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        field = cls(
            spatial_dim=data["spatial_dim"],
            rank=data["rank"],
            degree=data["degree"]
        )
        field._model.lambdas = np.array(data["lambdas"], dtype=np.float64)
        field._model.factors = [np.array(f, dtype=np.float64) for f in data["factors"]]
        return field

    def to_torch(self, dtype=None, device=None) -> "ContinuousKANLayer":
        """Copies the current weights into a new PyTorch ContinuousKANLayer."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for `.to_torch()`")
        if dtype is None:
            dtype = torch.float64
        layer = ContinuousKANLayer(
            in_features=self.spatial_dim,
            out_features=1,
            rank=self.rank,
            degree=self.degree,
            dtype=dtype,
            device=device
        )
        with torch.no_grad():
            layer.lambdas.copy_(torch.from_numpy(self.lambdas).to(dtype=dtype, device=device))
            for d in range(self.spatial_dim):
                layer.factors[d].copy_(torch.from_numpy(self.factors[d]).to(dtype=dtype, device=device))
        return layer

    def to_jax(self, dtype=None) -> Tuple[Dict[str, Any], "ContinuousKANJAXLayer"]:
        """Converts to JAX ContinuousKANJAXLayer with initialized parameters."""
        if not _HAS_JAX:
            raise ImportError("JAX is required for `.to_jax()`")
        if dtype is None:
            dtype = jnp.float32
        layer = ContinuousKANJAXLayer(
            in_features=self.spatial_dim,
            out_features=1,
            rank=self.rank,
            degree=self.degree,
            dtype=dtype
        )
        params = {
            "lambdas": jnp.asarray(self.lambdas, dtype=dtype),
            "factors": jnp.asarray(np.stack(self.factors, axis=0), dtype=dtype)
        }
        return params, layer


class TensorTrainField:
    r"""
    High-Level Ergonomic Interface for Tensor Train KAN (TT-KAN) Fields.
    
    Represents high-dimensional continuous mapping f: \mathbb{R}^D \to \mathbb{R} (D >= 10):
    f(x) = G^{(0)}(x_1) G^{(1)}(x_2) \dots G^{(D-1)}(x_D)
    
    Overcomes the curse of dimensionality with $O(D \cdot R^2 \cdot K)$ memory complexity.
    """
    def __init__(
        self,
        spatial_dim: int = 10,
        ranks: Optional[List[int]] = None,
        degree: int = 5,
        model: Optional[TensorTrainKAN] = None
    ):
        self.spatial_dim = spatial_dim
        self.degree = degree
        if model is not None:
            self._model = model
            self.spatial_dim = model.spatial_dim
            self.degree = model.degree
            self.ranks = model.ranks
        else:
            self._model = TensorTrainKAN(spatial_dim=spatial_dim, ranks=ranks, degree=degree)
            self.ranks = self._model.ranks

    @property
    def cores(self) -> List[np.ndarray]:
        return self._model.cores

    def fit_cross(
        self,
        target_fn: Callable[[np.ndarray], np.ndarray],
        max_rank: int = 16,
        eps: float = 1e-3
    ) -> "TensorTrainField":
        """
        Fits continuous TT-KAN field from black-box function using TT-Cross with O(D R^2 K) evaluations.
        """
        solver = TTCrossSolver(
            max_rank=max_rank,
            eps=eps
        )
        fitted_tt = solver.fit_function(
            func=target_fn,
            spatial_dim=self.spatial_dim,
            degree=self.degree
        )
        self._model = fitted_tt
        self.ranks = fitted_tt.ranks
        return self

    def predict(self, X: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Evaluates high-dimensional TT-KAN field at points X."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1 and len(X_arr) == self.spatial_dim:
            X_arr = X_arr[None, :]
            return self._model.evaluate(X_arr)[0]
        return self._model.evaluate(X_arr)

    def __call__(self, X: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        return self.predict(X)

    def gradient(self, X: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Computes exact analytical gradient vectors \nabla f(X)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1 and len(X_arr) == self.spatial_dim:
            X_arr = X_arr[None, :]
            return self._model.gradient(X_arr)[0]
        return self._model.gradient(X_arr)

    def to_torch(self, dtype=None, device=None) -> "TensorTrainKANLayer":
        """Copies the current TT cores into a new PyTorch TensorTrainKANLayer."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for `.to_torch()`")
        if dtype is None:
            dtype = torch.float64
        layer = TensorTrainKANLayer(
            in_features=self.spatial_dim,
            ranks=self.ranks,
            degree=self.degree,
            dtype=dtype,
            device=device
        )
        with torch.no_grad():
            for d in range(self.spatial_dim):
                layer.cores[d].copy_(torch.from_numpy(self.cores[d]).to(dtype=dtype, device=device))
        return layer

    def to_jax(self, dtype=None) -> Tuple[Dict[str, Any], "TensorTrainKANJAXLayer"]:
        """Converts to JAX TensorTrainKANJAXLayer."""
        if not _HAS_JAX:
            raise ImportError("JAX is required for `.to_jax()`")
        if dtype is None:
            dtype = jnp.float32
        layer = TensorTrainKANJAXLayer(
            in_features=self.spatial_dim,
            ranks=self.ranks,
            degree=self.degree,
            dtype=dtype
        )
        params = {"cores": [jnp.asarray(c, dtype=dtype) for c in self.cores]}
        return params, layer


class PoissonSolver:
    r"""
    High-Level Mesh-Free Physics-Informed KAN Poisson & Laplace PDE Solver.
    
    Solves \nabla^2 u(\mathbf{x}) = f(\mathbf{x}) with Dirichlet boundary conditions in 0 epochs algebraically.
    """
    def __init__(self, dim: int = 2, degree: int = 10, rank: int = 16):
        self.dim = dim
        self.degree = degree
        self.rank = rank
        if dim <= 3:
            self._solver = SpectralKANPoissonSolver(spatial_dim=dim, degree=degree)
        else:
            self._solver = TTPoissonSolver(dim=dim, degree=degree, rank=rank)
        self._fitted = False

    def solve(
        self,
        source_fn: Callable[..., np.ndarray],
        boundary_fn: Optional[Callable[..., np.ndarray]] = None,
        num_collocation: int = 400
    ) -> Dict[str, Any]:
        """
        Solves PDE algebraically in 0 gradient epochs.
        """
        if self.dim <= 3:
            # Check source_fn signature: if takes (X), wrap or pass
            def wrapped_f(X: np.ndarray) -> np.ndarray:
                if self.dim == 2:
                    try:
                        return source_fn(X[:, 0], X[:, 1])
                    except TypeError:
                        return source_fn(X)
                elif self.dim == 3:
                    try:
                        return source_fn(X[:, 0], X[:, 1], X[:, 2])
                    except TypeError:
                        return source_fn(X)
                return source_fn(X)

            if boundary_fn is None:
                def wrapped_g(X: np.ndarray) -> np.ndarray:
                    return np.zeros(len(X))
            else:
                def wrapped_g(X: np.ndarray) -> np.ndarray:
                    if self.dim == 2:
                        try:
                            return boundary_fn(X[:, 0], X[:, 1])
                        except TypeError:
                            return boundary_fn(X)
                    elif self.dim == 3:
                        try:
                            return boundary_fn(X[:, 0], X[:, 1], X[:, 2])
                        except TypeError:
                            return boundary_fn(X)
                    return boundary_fn(X)

            result = self._solver.fit(wrapped_f, wrapped_g)
        else:
            result = self._solver.solve(source_fn, boundary_fn, num_samples=num_collocation)
        self._fitted = True
        return result

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluates PDE solution at arbitrary coordinates X."""
        if not self._fitted:
            raise RuntimeError("Must call `.solve()` before `.evaluate()`.")
        return self._solver.evaluate(X)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X)


class CBFPlanner:
    r"""
    High-Level Control Barrier Function (CBF) Robotics Trajectory Planner.
    
    Filters control inputs through a CBF constraint solved as a QP (SLSQP).

    Scope of the guarantee: collision-freedom is checked on single scenarios in
    `tests/test_applications.py`. It is NOT a guarantee in the dynamic regime --
    the HOCBF condition drops the Hessian term (audit M1, open), and the fallback
    path after QP failure does not preserve CBF satisfaction (audit M4, open).
    """
    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        safety_margin: float = 0.05,
        alpha_cbf: float = 3.0
    ):
        if config is None:
            self.config = CBFConfig(d_safe=safety_margin, alpha=alpha_cbf)
        else:
            self.config = config
        self._planner = _CBFPlannerBase(self.config)
        self.obstacles: List[Any] = []

    def add_obstacle_field(self, field: Union[ContinuousKANObstacleField, TensorField, Any]) -> "CBFPlanner":
        """Adds a continuous KAN obstacle field to the planner."""
        if isinstance(field, TensorField):
            kan_field = ContinuousKANObstacleField(
                kan_model=field._model,
                threshold=0.0,
                invert=False
            )
            self.obstacles.append(kan_field)
        elif hasattr(field, "evaluate_h"):
            self.obstacles.append(field)
        else:
            kan_field = ContinuousKANObstacleField(
                kan_model=field,
                threshold=0.0,
                invert=False
            )
            self.obstacles.append(kan_field)
        return self

    def plan_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        dt: float = 0.01,
        max_steps: int = 500,
        goal_tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Plans a collision-free trajectory from start to goal.
        """
        return self._planner.simulate_kinematic_trajectory(
            start=start,
            goal=goal,
            obstacles=self.obstacles,
            dt=dt,
            max_steps=max_steps,
            goal_tolerance=goal_tolerance
        )
