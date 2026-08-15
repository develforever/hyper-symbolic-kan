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
from tests.test_tt_cross import (
    test_maxvol_submatrix_selection,
    test_tt_cross_reconstruction_20d,
    test_tt_cross_high_dimensional_50d,
    test_dmrg_2site_rank_adaptation,
    test_cpp_kernels_stage_b_precision
)
from tests.test_torch_kan import (
    test_continuous_kan_autograd_gradcheck,
    test_tensor_train_kan_autograd_gradcheck,
    test_continuous_kan_layer_forward_and_shapes,
    test_tensor_train_kan_layer_forward_and_shapes,
    test_torch_kan_optimization_loop,
    test_tensor_train_kan_optimization_loop,
    test_hybrid_als_gradient_fine_tuning,
    test_hybrid_tt_cross_gradient_fine_tuning,
    test_safetensors_serialization_roundtrip
)
from tests.test_applications import (
    test_chebyshev_second_derivative_analytical_exactness,
    test_cbf_kinematic_single_agent_3d,
    test_cbf_dynamic_hocbf_drone_flight,
    test_cbf_multi_agent_swarm_avoidance,
    test_poisson_solver_2d_analytical_benchmark,
    test_poisson_solver_3d_zero_epochs,
    test_poisson_high_dim_tensor_train
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
        # Stage B: Advanced Tensor Solvers (TT-Cross, MaxVol & 2-Site DMRG)
        ("MaxVol Submatrix Selection Accuracy & Volume Optimality", test_maxvol_submatrix_selection),
        ("TT-Cross Continuous 20D Field Reconstruction (O(D R^2 K))", test_tt_cross_reconstruction_20d),
        ("TT-Cross High-Dimensional 50D Scaling (< 1.0s)", test_tt_cross_high_dimensional_50d),
        ("2-Site DMRG Dynamic Rank Adaptation & SVD Truncation", test_dmrg_2site_rank_adaptation),
        ("Native C++ Stage B Kernels (Modal Projection & DMRG Normal)", test_cpp_kernels_stage_b_precision),
        # Stage D: PyTorch Ecosystem & SafeTensors Integration
        ("PyTorch ContinuousKANAutograd Gradcheck (< 1e-4)", test_continuous_kan_autograd_gradcheck),
        ("PyTorch TensorTrainKANAutograd Gradcheck (< 1e-4)", test_tensor_train_kan_autograd_gradcheck),
        ("PyTorch ContinuousKANLayer Forward & Multi-Batching", test_continuous_kan_layer_forward_and_shapes),
        ("PyTorch TensorTrainKANLayer Forward & Multi-Batching", test_tensor_train_kan_layer_forward_and_shapes),
        ("PyTorch ContinuousKANLayer Adam Optimizer Convergence", test_torch_kan_optimization_loop),
        ("PyTorch TensorTrainKANLayer Adam Optimizer Convergence", test_tensor_train_kan_optimization_loop),
        ("PyTorch Hybrid ALS -> Adam Gradient Fine-Tuning", test_hybrid_als_gradient_fine_tuning),
        ("PyTorch Hybrid TT-Cross -> TensorTrainKANLayer Bridge", test_hybrid_tt_cross_gradient_fine_tuning),
        ("SafeTensors Serialization & Metadata Header Roundtrip", test_safetensors_serialization_roundtrip),
        # Stage E: Industrial & Real-World Applications (Robotics CBF & Mesh-Free PDE)
        ("Chebyshev 2nd Derivative Recurrence Exactness (< 1e-11)", test_chebyshev_second_derivative_analytical_exactness),
        ("Robotics Kinematic 3D CBF Trajectory Planner (0% Collisions)", test_cbf_kinematic_single_agent_3d),
        ("Robotics Dynamic HOCBF Drone Flight (Relative Degree 2)", test_cbf_dynamic_hocbf_drone_flight),
        ("Robotics Multi-Agent Swarm Decentralized CBF (0% Violations)", test_cbf_multi_agent_swarm_avoidance),
        ("Mesh-Free 2D Poisson Analytical Benchmark (< 1e-4 in 0 Epochs)", test_poisson_solver_2d_analytical_benchmark),
        ("Mesh-Free 3D Poisson Spectral Solver (< 1e-4 in 0 Epochs)", test_poisson_solver_3d_zero_epochs),
        ("High-Dimensional 4D TT-KAN Poisson Solver (ALS 0 Epochs)", test_poisson_high_dim_tensor_train),
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
