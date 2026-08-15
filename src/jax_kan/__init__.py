"""
JAX Backend Package for Hyper-Symbolic KAN.

Exposes:
- `cp_kan_forward`, `tt_kan_forward` (Analytical Custom VJP autograd functions)
- `ContinuousKANJAXLayer`
- `TensorTrainKANJAXLayer`
"""

from src.jax_kan.autograd_ops import (
    cp_kan_forward,
    tt_kan_forward,
    compute_chebyshev_jax,
    compute_chebyshev_and_deriv_jax
)
from src.jax_kan.layers import (
    ContinuousKANJAXLayer,
    TensorTrainKANJAXLayer
)

__all__ = [
    "cp_kan_forward",
    "tt_kan_forward",
    "compute_chebyshev_jax",
    "compute_chebyshev_and_deriv_jax",
    "ContinuousKANJAXLayer",
    "TensorTrainKANJAXLayer"
]
