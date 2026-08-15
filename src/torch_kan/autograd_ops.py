import torch
import numpy as np
from typing import List, Tuple, Optional

# Import native C++ kernels if available
try:
    from src.cpp_kernels import _cpp_kernels as _native_kernels
    _HAS_CPP = True
except ImportError:
    try:
        import _cpp_kernels as _native_kernels
        _HAS_CPP = True
    except ImportError:
        _native_kernels = None
        _HAS_CPP = False


def _compute_chebyshev_torch(x_d: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Computes Chebyshev polynomials T_0(x) ... T_K(x) for 1D input tensor.
    x_d: (N,)
    returns: (N, degree + 1)
    """
    N = x_d.shape[0]
    T = torch.empty((N, degree + 1), dtype=x_d.dtype, device=x_d.device)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x_d
    for k in range(1, degree):
        T[:, k + 1] = 2.0 * x_d * T[:, k] - T[:, k - 1]
    return T


def _compute_chebyshev_and_deriv_torch(
    x_d: torch.Tensor, degree: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Chebyshev polynomials T_k(x) and analytical derivatives dT_k/dx.
    x_d: (N,)
    returns: (T, dT) each of shape (N, degree + 1)
    """
    N = x_d.shape[0]
    T = torch.empty((N, degree + 1), dtype=x_d.dtype, device=x_d.device)
    dT = torch.empty((N, degree + 1), dtype=x_d.dtype, device=x_d.device)
    
    T[:, 0] = 1.0
    dT[:, 0] = 0.0
    
    if degree >= 1:
        T[:, 1] = x_d
        dT[:, 1] = 1.0
        
    for k in range(1, degree):
        T[:, k + 1] = 2.0 * x_d * T[:, k] - T[:, k - 1]
        dT[:, k + 1] = 2.0 * T[:, k] + 2.0 * x_d * dT[:, k] - dT[:, k - 1]
        
    return T, dT


class ContinuousKANAutograd(torch.autograd.Function):
    r"""
    Dedykowany Custom Autograd Function dla CP-KAN (TDFF-Net).
    
    Forward:
      Wywołuje natywny kernel C++ (nanobind / AVX2 / OpenMP) lub wektoryzowany PyTorch/NumPy.
    Backward:
      - Gradient po wejściu X: \nabla_X f(X) analityczny z tożsamości Czebyszewa (O(1) alokacji grafu).
      - Gradient po lambdas: iloczyn składowych bazowych tensorów.
      - Gradient po macierzach czynników: akumulacja prefiksowo-sufiksowa bez dzielenia przez zero.
    """
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        lambdas: torch.Tensor,
        degree: int,
        *factors: torch.Tensor
    ) -> torch.Tensor:
        D = X.shape[1]
        N = X.shape[0]
        rank = lambdas.shape[0]

        assert len(factors) == D, f"Expected {D} factors, got {len(factors)}"

        ctx.degree = degree
        ctx.save_for_backward(X, lambdas, *factors)

        # Fast C++ evaluation if on CPU and float64
        if _HAS_CPP and _native_kernels is not None and X.device.type == "cpu" and X.dtype == torch.float64:
            X_np = X.detach().contiguous().numpy()
            lambdas_np = lambdas.detach().contiguous().numpy()
            factors_flat = np.concatenate([
                f.detach().contiguous().numpy().ravel() for f in factors
            ]).astype(np.float64, copy=False)
            
            Y_np = np.empty(N, dtype=np.float64)
            _native_kernels.evaluate_cp_kan_batch(
                X_np, factors_flat, lambdas_np, rank, degree, Y_np
            )
            return torch.from_numpy(Y_np).to(device=X.device, dtype=X.dtype)

        # Pure PyTorch tensor implementation (differentiable / GPU compatible)
        cp_product = torch.ones((N, rank), dtype=X.dtype, device=X.device)
        for d in range(D):
            T_d = _compute_chebyshev_torch(X[:, d], degree)
            phi_d = T_d @ factors[d].t()  # (N, R)
            cp_product = cp_product * phi_d
            
        return cp_product @ lambdas

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        X, lambdas, *factors = ctx.saved_tensors
        degree = ctx.degree
        N, D = X.shape
        rank = lambdas.shape[0]
        g = grad_output.reshape(N)

        T_list = []
        dT_list = []
        phi_list = []
        dphi_list = []

        for d in range(D):
            T_d, dT_d = _compute_chebyshev_and_deriv_torch(X[:, d], degree)
            phi_d = T_d @ factors[d].t()   # (N, R)
            dphi_d = dT_d @ factors[d].t() # (N, R)
            T_list.append(T_d)
            dT_list.append(dT_d)
            phi_list.append(phi_d)
            dphi_list.append(dphi_d)

        # Prefix products: pref[d] = prod_{j=0}^{d-1} phi_j
        pref = [None] * D
        curr_pref = torch.ones((N, rank), dtype=X.dtype, device=X.device)
        for d in range(D):
            pref[d] = curr_pref
            curr_pref = curr_pref * phi_list[d]

        # Suffix products: suff[d] = prod_{j=d+1}^{D-1} phi_j
        suff = [None] * D
        curr_suff = torch.ones((N, rank), dtype=X.dtype, device=X.device)
        for d in range(D - 1, -1, -1):
            suff[d] = curr_suff
            curr_suff = curr_suff * phi_list[d]

        # Total CP product per rank: (N, R)
        total_P = curr_pref

        # 1. Gradient w.r.t. X
        grad_X = None
        if ctx.needs_input_grad[0]:
            grad_X = torch.zeros((N, D), dtype=X.dtype, device=X.device)
            if _HAS_CPP and _native_kernels is not None and X.device.type == "cpu" and X.dtype == torch.float64:
                X_np = X.detach().contiguous().numpy()
                lambdas_np = lambdas.detach().contiguous().numpy()
                factors_flat = np.concatenate([
                    f.detach().contiguous().numpy().ravel() for f in factors
                ]).astype(np.float64, copy=False)
                
                grad_np = np.empty((N, D), dtype=np.float64)
                _native_kernels.evaluate_cp_kan_gradient_batch(
                    X_np, factors_flat, lambdas_np, rank, degree, grad_np
                )
                grad_X = torch.from_numpy(grad_np).to(device=X.device, dtype=X.dtype) * g[:, None]
            else:
                for d in range(D):
                    Q_d = pref[d] * suff[d]
                    df_d = (dphi_list[d] * Q_d) @ lambdas
                    grad_X[:, d] = g * df_d

        # 2. Gradient w.r.t. lambdas
        grad_lambdas = None
        if ctx.needs_input_grad[1]:
            grad_lambdas = total_P.t() @ g

        # 3. Gradient w.r.t. factors
        grad_factors = []
        for d in range(D):
            if ctx.needs_input_grad[3 + d]:
                Q_d = pref[d] * suff[d]  # (N, R)
                # H_d_{n, r} = g_n * lambda_r * Q_d_{n, r}
                H_d = g[:, None] * lambdas[None, :] * Q_d  # (N, R)
                # grad_factors[d] = H_d^T @ T_d -> (R, K1)
                g_fact_d = H_d.t() @ T_list[d]
                grad_factors.append(g_fact_d)
            else:
                grad_factors.append(None)

        return (grad_X, grad_lambdas, None, *grad_factors)


class TensorTrainKANAutograd(torch.autograd.Function):
    r"""
    Dedykowany Custom Autograd Function dla Tensor Train KAN (TT-KAN).
    
    Forward:
      Wywołuje natywny kernel C++ (nanobind / AVX2 / OpenMP) lub wektoryzowaną kontrakcję TT.
    Backward:
      - Gradient po wejściu X: \nabla_X f(X) analityczny (O(1) pamięci grafu).
      - Gradient po rdzeniach TT G^(d): kontrakcja lewych prefiksów L^(d-1), baz T_k i prawych sufiksów R^(d+1).
    """
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        degree: int,
        ranks: Tuple[int, ...],
        *cores: torch.Tensor
    ) -> torch.Tensor:
        D = X.shape[1]
        N = X.shape[0]
        K1 = degree + 1

        assert len(cores) == D, f"Expected {D} cores, got {len(cores)}"
        
        ctx.degree = degree
        ctx.ranks = tuple(int(r) for r in ranks)
        ctx.save_for_backward(X, *cores)

        # Fast C++ evaluation if on CPU and float64
        if _HAS_CPP and _native_kernels is not None and X.device.type == "cpu" and X.dtype == torch.float64:
            X_np = X.detach().contiguous().numpy()
            cores_flat_list = []
            core_offsets = []
            offset = 0
            for c in cores:
                c_flat = c.detach().contiguous().numpy().ravel()
                cores_flat_list.append(c_flat)
                core_offsets.append(offset)
                offset += c_flat.size
                
            all_cores_flat = np.concatenate(cores_flat_list).astype(np.float64, copy=False)
            core_offsets_arr = np.array(core_offsets, dtype=np.int32)
            ranks_arr = np.array(ranks, dtype=np.int32)
            
            Y_np = np.empty(N, dtype=np.float64)
            _native_kernels.evaluate_tt_kan_batch(
                X_np,
                all_cores_flat,
                core_offsets_arr,
                ranks_arr,
                degree,
                Y_np
            )
            return torch.from_numpy(Y_np).to(device=X.device, dtype=X.dtype)

        # PyTorch fallback
        curr = torch.ones((N, 1), dtype=X.dtype, device=X.device)
        for d in range(D):
            T_d = _compute_chebyshev_torch(X[:, d], degree)
            core_d = cores[d]
            r_prev, _, r_next = core_d.shape
            core_flat = core_d.transpose(0, 1).reshape(K1, r_prev * r_next)
            # (N, K1) @ (K1, r_prev * r_next) -> (N, r_prev, r_next)
            M_d = (T_d @ core_flat).reshape(N, r_prev, r_next)
            curr = torch.bmm(curr.unsqueeze(1), M_d).squeeze(1)

        return curr.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        X, *cores = ctx.saved_tensors
        degree = ctx.degree
        ranks = ctx.ranks
        N, D = X.shape
        K1 = degree + 1
        g = grad_output.reshape(N)

        T_list = []
        dT_list = []
        M_list = []
        dM_list = []

        for d in range(D):
            T_d, dT_d = _compute_chebyshev_and_deriv_torch(X[:, d], degree)
            core_d = cores[d]
            r_prev, _, r_next = core_d.shape
            
            core_flat = core_d.transpose(0, 1).reshape(K1, r_prev * r_next)
            M_d = (T_d @ core_flat).reshape(N, r_prev, r_next)
            dM_d = (dT_d @ core_flat).reshape(N, r_prev, r_next)
            
            T_list.append(T_d)
            dT_list.append(dT_d)
            M_list.append(M_d)
            dM_list.append(dM_d)

        # Left prefixes L[d]: shape (N, ranks[d])
        L = [None] * (D + 1)
        L[0] = torch.ones((N, 1), dtype=X.dtype, device=X.device)
        for d in range(D):
            L[d + 1] = torch.bmm(L[d].unsqueeze(1), M_list[d]).squeeze(1)

        # Right suffixes R[d]: shape (N, ranks[d])
        R = [None] * (D + 1)
        R[D] = torch.ones((N, 1), dtype=X.dtype, device=X.device)
        for d in range(D - 1, -1, -1):
            R[d] = torch.bmm(M_list[d], R[d + 1].unsqueeze(-1)).squeeze(-1)

        # 1. Gradient w.r.t. X
        grad_X = None
        if ctx.needs_input_grad[0]:
            grad_X = torch.zeros((N, D), dtype=X.dtype, device=X.device)
            if _HAS_CPP and _native_kernels is not None and X.device.type == "cpu" and X.dtype == torch.float64:
                X_np = X.detach().contiguous().numpy()
                cores_flat_list = []
                core_offsets = []
                offset = 0
                for c in cores:
                    c_flat = c.detach().contiguous().numpy().ravel()
                    cores_flat_list.append(c_flat)
                    core_offsets.append(offset)
                    offset += c_flat.size
                    
                all_cores_flat = np.concatenate(cores_flat_list).astype(np.float64, copy=False)
                core_offsets_arr = np.array(core_offsets, dtype=np.int32)
                ranks_arr = np.array(ranks, dtype=np.int32)
                
                grad_np = np.empty((N, D), dtype=np.float64)
                _native_kernels.evaluate_tt_kan_gradient_batch(
                    X_np,
                    all_cores_flat,
                    core_offsets_arr,
                    ranks_arr,
                    degree,
                    grad_np
                )
                grad_X = torch.from_numpy(grad_np).to(device=X.device, dtype=X.dtype) * g[:, None]
            else:
                for d in range(D):
                    mid = torch.bmm(L[d].unsqueeze(1), dM_list[d]).squeeze(1) # (N, r_next)
                    df_d = (mid * R[d + 1]).sum(dim=1)
                    grad_X[:, d] = g * df_d

        # 2. Gradient w.r.t. cores
        grad_cores = []
        for d in range(D):
            if ctx.needs_input_grad[3 + d]:
                L_weighted = g[:, None] * L[d] # (N, r_prev)
                grad_c = torch.einsum('nr, nk, ns -> rks', L_weighted, T_list[d], R[d + 1])
                grad_cores.append(grad_c)
            else:
                grad_cores.append(None)

        return (grad_X, None, None, *grad_cores)
