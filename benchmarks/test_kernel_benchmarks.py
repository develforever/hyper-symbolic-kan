r"""Performance benchmarks for the native kernels (audit V2).

`tests/test_cpp_kernels.py` used to assert absolute, hardware-dependent
thresholds::

    assert throughput >= 1_500_000     # points / s
    assert latency_us <= 0.67          # us / point

Those numbers describe one machine, not the code, so they fail on any slower
CPU and pass vacuously on a faster one. They are replaced by pytest-benchmark
runs with **no absolute assertion**: the regression criterion is a comparison
against a baseline recorded on the *same* machine (median regression > 20%),
which is a property of the change rather than of the runner.

See `benchmarks/README.md` for how to record a baseline and run the gate.
These benchmarks are not part of the `tests/` suite and are not collected by
`pytest tests/`.
"""

import numpy as np
import pytest

from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tt_kan import TensorTrainKAN
from tests._native import requires_native

pytestmark = requires_native

# Sizes are fixed so that a saved baseline stays comparable across runs.
TT_DIM = 10
TT_DEGREE = 5
TT_RANK = 8
TT_BATCH = 100_000

CP_DIM = 4
CP_RANK = 12
CP_DEGREE = 5
CP_BATCH = 100_000


@pytest.fixture(scope="module")
def tt_model():
    rng = np.random.default_rng(42)
    model = TensorTrainKAN(
        spatial_dim=TT_DIM,
        ranks=[1] + [TT_RANK] * (TT_DIM - 1) + [1],
        degree=TT_DEGREE,
    )
    X = rng.uniform(-0.9, 0.9, (TT_BATCH, TT_DIM))
    engine = FastCPPKANEngine(spatial_dim=TT_DIM, degree=TT_DEGREE)
    return model, engine, np.ascontiguousarray(X)


@pytest.fixture(scope="module")
def cp_model():
    rng = np.random.default_rng(42)
    model = TDFFNet(spatial_dim=CP_DIM, rank=CP_RANK, degree=CP_DEGREE)
    X = rng.uniform(-0.9, 0.9, (CP_BATCH, CP_DIM))
    engine = FastCPPKANEngine(spatial_dim=CP_DIM, degree=CP_DEGREE)
    return model, engine, np.ascontiguousarray(X)


def test_bench_tt_kan_forward_batch(benchmark, tt_model):
    model, engine, X = tt_model
    engine.evaluate_batch(X[:1000], model.cores, model.ranks)  # warm-up

    result = benchmark(engine.evaluate_batch, X, model.cores, model.ranks)

    assert result.shape == (TT_BATCH,)
    benchmark.extra_info["points"] = TT_BATCH
    benchmark.extra_info["config"] = f"D={TT_DIM} rank={TT_RANK} degree={TT_DEGREE}"


def test_bench_tt_kan_gradient_batch(benchmark, tt_model):
    model, engine, X = tt_model
    X_small = np.ascontiguousarray(X[:20_000])
    engine.gradient_batch(X_small[:1000], model.cores, model.ranks)  # warm-up

    result = benchmark(engine.gradient_batch, X_small, model.cores, model.ranks)

    assert result.shape == (20_000, TT_DIM)
    benchmark.extra_info["points"] = 20_000
    benchmark.extra_info["config"] = f"D={TT_DIM} rank={TT_RANK} degree={TT_DEGREE}"


def test_bench_cp_kan_forward_batch(benchmark, cp_model):
    model, engine, X = cp_model
    engine.evaluate_cp_batch(X[:1000], model.factors, model.lambdas)  # warm-up

    result = benchmark(engine.evaluate_cp_batch, X, model.factors, model.lambdas)

    assert result.shape == (CP_BATCH,)
    benchmark.extra_info["points"] = CP_BATCH
    benchmark.extra_info["config"] = f"D={CP_DIM} rank={CP_RANK} degree={CP_DEGREE}"
