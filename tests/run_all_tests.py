import sys
import os
import traceback
import tempfile
import pathlib

# System path patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_tensor_field import (
    test_tdff_net_analytical_gradient,
    test_closed_form_als_convergence_and_stability,
    test_tt_kan_gradient_exactness
)
from tests.test_serializer import (
    test_serializer_roundtrip_json,
    test_webgpu_buffers_format
)
from tests.test_safety_and_nary import (
    test_concurrent_category_filter_zero_violations,
    test_isotropic_spatiotemporal_encoding
)
from tests.test_cpp_kernels import (
    test_tt_kan_cpp_forward_precision,
    test_tt_kan_cpp_gradient_precision,
    test_cp_kan_cpp_forward_precision,
    test_cp_kan_cpp_gradient_precision,
    test_cpp_throughput_and_latency_benchmark,
    test_gil_release_and_concurrency
)

def main():
    print("=" * 75)
    print("HYPER-SYMBOLIC KAN FULL ARCHITECTURAL & C++ NUMERICAL TEST SUITE")
    print("=" * 75)
    
    tests = [
        ("TDFFNet Analytical Gradient vs Finite Differences", test_tdff_net_analytical_gradient),
        ("Closed-Form ALS Convergence & Normalized Stability", test_closed_form_als_convergence_and_stability),
        ("Tensor Train KAN (TT-KAN) Gradient Exactness", test_tt_kan_gradient_exactness),
        ("WebGPU Buffer Layout & Serialization Format", test_webgpu_buffers_format),
        ("Concurrent Category Guard Safety (0% Violations)", test_concurrent_category_filter_zero_violations),
        ("Isotropic Multi-Dimensional Spatio-Temporal Encoding", test_isotropic_spatiotemporal_encoding),
        ("TT-KAN Native C++ (nanobind) Forward Precision (< 1e-11)", test_tt_kan_cpp_forward_precision),
        ("TT-KAN Native C++ (nanobind) Gradient Precision (< 1e-10)", test_tt_kan_cpp_gradient_precision),
        ("CP-KAN Native C++ (nanobind) Forward Precision (< 1e-11)", test_cp_kan_cpp_forward_precision),
        ("CP-KAN Native C++ (nanobind) Gradient Precision (< 1e-10)", test_cp_kan_cpp_gradient_precision),
        ("Native C++ SIMD Throughput & Latency (> 1,500,000 pts/s)", test_cpp_throughput_and_latency_benchmark),
        ("Native C++ GIL Release & Concurrency (Multi-Threading)", test_gil_release_and_concurrency),
    ]
    
    passed = 0
    for name, func in tests:
        try:
            func()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")
            traceback.print_exc()
            
    # Test JSON roundtrip with tmp directory
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_serializer_roundtrip_json(pathlib.Path(tmp_dir))
            print(" [PASS] KAN JSON Serialization Roundtrip")
            passed += 1
    except Exception as e:
        print(f" [FAIL] KAN JSON Serialization Roundtrip: {e}")
        traceback.print_exc()

    total = len(tests) + 1
    print("=" * 75)
    print(f"TOTAL: {passed} / {total} PASSED")
    print("=" * 75)
    
    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
