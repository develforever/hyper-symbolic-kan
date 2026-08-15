import sys
import os
import traceback

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

import tempfile
import pathlib

def main():
    print("=" * 70)
    print("HYPER-SYMBOLIC KAN NUMERICAL & ARCHITECTURAL TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("TDFFNet Analytical Gradient vs Finite Differences", test_tdff_net_analytical_gradient),
        ("Closed-Form ALS Convergence & Normalized Stability", test_closed_form_als_convergence_and_stability),
        ("Tensor Train KAN (TT-KAN) Gradient Exactness", test_tt_kan_gradient_exactness),
        ("WebGPU Buffer Layout & Serialization Format", test_webgpu_buffers_format),
        ("Concurrent Category Guard Safety (0% Violations)", test_concurrent_category_filter_zero_violations),
        ("Isotropic Multi-Dimensional Spatio-Temporal Encoding", test_isotropic_spatiotemporal_encoding),
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

    print("=" * 70)
    print(f"TOTAL: {passed} / {len(tests) + 1} PASSED")
    print("=" * 70)
    
    if passed == len(tests) + 1:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
