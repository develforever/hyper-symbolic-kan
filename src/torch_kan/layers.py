import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union, Callable

from src.torch_kan.autograd_ops import ContinuousKANAutograd, TensorTrainKANAutograd
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tt_kan import TensorTrainKAN, TTALSSolver
from src.tdff_net.closed_form_als import ClosedFormALSSolver
from src.tdff_net.tt_cross import TTCrossSolver


class ContinuousKANLayer(nn.Module):
    r"""
    Warstwa PyTorch nn.Module oparta na rozkładzie tensorowym CP-KAN (TDFF-Net).
    
    Ciągła reprezentacja nieliniowych mapowań f: \mathbb{R}^{D} \to \mathbb{R}^{out\_features}
    oparta na bazach wielomianów Czebyszewa:
    f(x_1, ..., x_D) = \sum_{r=1}^R \lambda_r \prod_{d=1}^D \left( \sum_{k=0}^K W_{r, k}^{(d)} T_k(x_d) \right)
    
    Zapewnia:
    - O(1) analityczny backward pass przez natywny kernel C++ (nanobind).
    - Możliwość natychmiastowej analitycznej inicjalizacji przez Closed-Form ALS (0 epok).
    - Pełną kompatybilność z torch.optim (Adam, SGD, LBFGS) oraz autogradem.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        rank: int = 16,
        degree: int = 5,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.degree = degree
        self.dtype = dtype

        factory_kwargs = {"device": device, "dtype": dtype}

        # Inicjalizacja parametrów
        if self.out_features == 1:
            self.lambdas = nn.Parameter(torch.ones(self.rank, **factory_kwargs))
            self.factors = nn.ParameterList([
                nn.Parameter(
                    torch.randn(self.rank, self.degree + 1, **factory_kwargs) / np.sqrt(self.degree + 1)
                )
                for _ in range(self.in_features)
            ])
        else:
            self.lambdas = nn.Parameter(torch.ones(self.out_features, self.rank, **factory_kwargs))
            self.factors = nn.ParameterList([
                nn.Parameter(
                    torch.randn(self.out_features, self.rank, self.degree + 1, **factory_kwargs) / np.sqrt(self.degree + 1)
                )
                for _ in range(self.in_features)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if orig_shape[-1] != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {orig_shape[-1]}")

        x_2d = x.reshape(-1, self.in_features)
        if x_2d.dtype != self.dtype:
            x_2d = x_2d.to(dtype=self.dtype)

        if self.out_features == 1:
            out_2d = ContinuousKANAutograd.apply(
                x_2d, self.lambdas, self.degree, *self.factors
            ).unsqueeze(-1)
        else:
            outs = []
            for out_idx in range(self.out_features):
                factors_c = [self.factors[d][out_idx] for d in range(self.in_features)]
                out_c = ContinuousKANAutograd.apply(
                    x_2d, self.lambdas[out_idx], self.degree, *factors_c
                )
                outs.append(out_c)
            out_2d = torch.stack(outs, dim=-1)

        out_shape = (*orig_shape[:-1], self.out_features)
        return out_2d.reshape(out_shape)

    def fit_als(
        self,
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray],
        alpha: float = 1e-4,
        max_als_iters: int = 10
    ) -> float:
        r"""
        Beziteracyjna optymalizacja wag warstwy metodą Closed-Form ALS w 0 epokach.
        X: (N, in_features)
        Y: (N, out_features) lub (N,)
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if isinstance(Y, torch.Tensor):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = np.asarray(Y)

        if Y_np.ndim == 1:
            Y_np = Y_np[:, None]

        solver = ClosedFormALSSolver(alpha=alpha, max_als_iters=max_als_iters)
        total_mse = 0.0

        for out_idx in range(self.out_features):
            tdff = TDFFNet(spatial_dim=self.in_features, rank=self.rank, degree=self.degree)
            mse_c = solver.fit(tdff, X_np, Y_np[:, out_idx])
            total_mse += mse_c

            with torch.no_grad():
                if self.out_features == 1:
                    self.lambdas.copy_(torch.from_numpy(tdff.lambdas).to(device=self.lambdas.device, dtype=self.dtype))
                    for d in range(self.in_features):
                        self.factors[d].copy_(torch.from_numpy(tdff.factors[d]).to(device=self.factors[d].device, dtype=self.dtype))
                else:
                    self.lambdas[out_idx].copy_(torch.from_numpy(tdff.lambdas).to(device=self.lambdas.device, dtype=self.dtype))
                    for d in range(self.in_features):
                        self.factors[d][out_idx].copy_(torch.from_numpy(tdff.factors[d]).to(device=self.factors[d].device, dtype=self.dtype))

        return total_mse / self.out_features

    def freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = True

    def to_tdff_net(self) -> Union[TDFFNet, List[TDFFNet]]:
        if self.out_features == 1:
            model = TDFFNet(spatial_dim=self.in_features, rank=self.rank, degree=self.degree)
            model.lambdas = self.lambdas.detach().cpu().numpy().copy()
            model.factors = [f.detach().cpu().numpy().copy() for f in self.factors]
            return model
        else:
            models = []
            for out_idx in range(self.out_features):
                model = TDFFNet(spatial_dim=self.in_features, rank=self.rank, degree=self.degree)
                model.lambdas = self.lambdas[out_idx].detach().cpu().numpy().copy()
                model.factors = [self.factors[d][out_idx].detach().cpu().numpy().copy() for d in range(self.in_features)]
                models.append(model)
            return models

    def from_tdff_net(self, tdff_net: Union[TDFFNet, List[TDFFNet]]):
        with torch.no_grad():
            if isinstance(tdff_net, TDFFNet):
                assert self.out_features == 1
                self.lambdas.copy_(torch.from_numpy(tdff_net.lambdas).to(device=self.lambdas.device, dtype=self.dtype))
                for d in range(self.in_features):
                    self.factors[d].copy_(torch.from_numpy(tdff_net.factors[d]).to(device=self.factors[d].device, dtype=self.dtype))
            elif isinstance(tdff_net, list):
                assert len(tdff_net) == self.out_features
                for out_idx, m in enumerate(tdff_net):
                    self.lambdas[out_idx].copy_(torch.from_numpy(m.lambdas).to(device=self.lambdas.device, dtype=self.dtype))
                    for d in range(self.in_features):
                        self.factors[d][out_idx].copy_(torch.from_numpy(m.factors[d]).to(device=self.factors[d].device, dtype=self.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, degree={self.degree}, dtype={self.dtype}"


class TensorTrainKANLayer(nn.Module):
    r"""
    Warstwa PyTorch nn.Module oparta na formacie Tensor Train (TT-KAN).
    
    Ciągła reprezentacja nieliniowych pól f: \mathbb{R}^{D} \to \mathbb{R}^{out\_features}
    w łańcuchu rdzeni tensora:
    f(x) = G^(0)(x_1) G^(1)(x_2) ... G^(D-1)(x_D)
    
    Zapewnia:
    - Skalowanie do wysokich wymiarów D >= 20 bez klątwy wymiarowości.
    - Analityczny gradient \nabla_X f(X) oraz analityczny backward wag dG^(d).
    - Bezgradientową inicjalizację przez TT-Cross (O(D R^2 K) ewaluacji) oraz TT-ALS.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        ranks: Optional[List[int]] = None,
        degree: int = 5,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.dtype = dtype

        if ranks is None:
            R = 8
            self.ranks = [1] + [R] * (self.in_features - 1) + [1]
        else:
            assert len(ranks) == self.in_features + 1 and ranks[0] == 1 and ranks[-1] == 1
            self.ranks = list(ranks)

        factory_kwargs = {"device": device, "dtype": dtype}
        K1 = self.degree + 1

        if self.out_features == 1:
            self.cores = nn.ParameterList([
                nn.Parameter(
                    torch.randn(self.ranks[d], K1, self.ranks[d + 1], **factory_kwargs)
                    / np.sqrt(self.ranks[d] * self.ranks[d + 1] * K1)
                )
                for d in range(self.in_features)
            ])
        else:
            self.cores = nn.ParameterList([
                nn.Parameter(
                    torch.randn(self.out_features, self.ranks[d], K1, self.ranks[d + 1], **factory_kwargs)
                    / np.sqrt(self.ranks[d] * self.ranks[d + 1] * K1)
                )
                for d in range(self.in_features)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if orig_shape[-1] != self.in_features:
            raise ValueError(f"Expected in_features={self.in_features}, got {orig_shape[-1]}")

        x_2d = x.reshape(-1, self.in_features)
        if x_2d.dtype != self.dtype:
            x_2d = x_2d.to(dtype=self.dtype)

        if self.out_features == 1:
            out_2d = TensorTrainKANAutograd.apply(
                x_2d, self.degree, tuple(self.ranks), *self.cores
            ).unsqueeze(-1)
        else:
            outs = []
            for out_idx in range(self.out_features):
                cores_c = [self.cores[d][out_idx] for d in range(self.in_features)]
                out_c = TensorTrainKANAutograd.apply(
                    x_2d, self.degree, tuple(self.ranks), *cores_c
                )
                outs.append(out_c)
            out_2d = torch.stack(outs, dim=-1)

        out_shape = (*orig_shape[:-1], self.out_features)
        return out_2d.reshape(out_shape)

    def fit_tt_cross(
        self,
        func_or_samples: Union[Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]],
        max_rank: int = 8,
        eps: float = 1e-5,
        max_sweeps: int = 4
    ):
        r"""
        Dopasowanie rdzeni TT-Cross dla czarnej skrzynki func(X) w złożoności O(D * R^2 * K).
        """
        solver = TTCrossSolver(max_rank=max_rank, eps=eps, max_sweeps=max_sweeps)

        if self.out_features == 1:
            fn = func_or_samples if callable(func_or_samples) else func_or_samples[0]
            tt_kan = solver.fit_function(
                fn, spatial_dim=self.in_features, degree=self.degree, target_ranks=self.ranks
            )
            self.from_tt_kan(tt_kan)
        else:
            fns = func_or_samples if isinstance(func_or_samples, list) else [func_or_samples]
            models = []
            for out_idx in range(self.out_features):
                tt_kan = solver.fit_function(
                    fns[out_idx], spatial_dim=self.in_features, degree=self.degree, target_ranks=self.ranks
                )
                models.append(tt_kan)
            self.from_tt_kan(models)

    def fit_als(
        self,
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray],
        alpha: float = 1e-4,
        max_sweeps: int = 4
    ) -> float:
        r"""
        Dopasowanie rdzeni TT-KAN za pomocą solvera TT-ALS w 0 epokach gradientowych.
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if isinstance(Y, torch.Tensor):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = np.asarray(Y)

        if Y_np.ndim == 1:
            Y_np = Y_np[:, None]

        solver = TTALSSolver(alpha=alpha, max_sweeps=max_sweeps)
        total_mse = 0.0

        for out_idx in range(self.out_features):
            tt_model = TensorTrainKAN(spatial_dim=self.in_features, ranks=self.ranks, degree=self.degree)
            rmse_c = solver.fit(tt_model, X_np, Y_np[:, out_idx])
            total_mse += (rmse_c ** 2)

            with torch.no_grad():
                if self.out_features == 1:
                    for d in range(self.in_features):
                        self.cores[d].copy_(torch.from_numpy(tt_model.cores[d]).to(device=self.cores[d].device, dtype=self.dtype))
                else:
                    for d in range(self.in_features):
                        self.cores[d][out_idx].copy_(torch.from_numpy(tt_model.cores[d]).to(device=self.cores[d].device, dtype=self.dtype))

        return total_mse / self.out_features

    def freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = True

    def to_tt_kan(self) -> Union[TensorTrainKAN, List[TensorTrainKAN]]:
        if self.out_features == 1:
            model = TensorTrainKAN(spatial_dim=self.in_features, ranks=self.ranks, degree=self.degree)
            model.cores = [c.detach().cpu().numpy().copy() for c in self.cores]
            return model
        else:
            models = []
            for out_idx in range(self.out_features):
                model = TensorTrainKAN(spatial_dim=self.in_features, ranks=self.ranks, degree=self.degree)
                model.cores = [self.cores[d][out_idx].detach().cpu().numpy().copy() for d in range(self.in_features)]
                models.append(model)
            return models

    def from_tt_kan(self, tt_kan: Union[TensorTrainKAN, List[TensorTrainKAN]]):
        with torch.no_grad():
            if isinstance(tt_kan, TensorTrainKAN):
                assert self.out_features == 1
                self.ranks = list(tt_kan.ranks)
                for d in range(self.in_features):
                    # Check if shape matches, else re-create parameter
                    if self.cores[d].shape != tt_kan.cores[d].shape:
                        self.cores[d] = nn.Parameter(
                            torch.from_numpy(tt_kan.cores[d]).to(device=self.cores[d].device, dtype=self.dtype)
                        )
                    else:
                        self.cores[d].copy_(torch.from_numpy(tt_kan.cores[d]).to(device=self.cores[d].device, dtype=self.dtype))
            elif isinstance(tt_kan, list):
                assert len(tt_kan) == self.out_features
                self.ranks = list(tt_kan[0].ranks)
                for out_idx, m in enumerate(tt_kan):
                    for d in range(self.in_features):
                        self.cores[d][out_idx].copy_(torch.from_numpy(m.cores[d]).to(device=self.cores[d].device, dtype=self.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, ranks={self.ranks}, degree={self.degree}, dtype={self.dtype}"
