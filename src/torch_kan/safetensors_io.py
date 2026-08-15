import json
import os
import torch
import numpy as np
from typing import Dict, Any, Union, Optional
from safetensors.torch import save_file, load_file
import safetensors

from src.torch_kan.layers import ContinuousKANLayer, TensorTrainKANLayer
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tt_kan import TensorTrainKAN


def save_kan_safetensors(
    model: Union[ContinuousKANLayer, TensorTrainKANLayer, TDFFNet, TensorTrainKAN],
    filepath: str,
    metadata: Optional[Dict[str, str]] = None
):
    r"""
    Bezpieczny zapis wag modelu KAN do formatu .safetensors z nagłówkiem metadanych topologii tensorowej.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    meta = metadata.copy() if metadata is not None else {}

    tensors: Dict[str, torch.Tensor] = {}

    if isinstance(model, ContinuousKANLayer):
        meta["model_type"] = "ContinuousKANLayer"
        meta["in_features"] = str(model.in_features)
        meta["out_features"] = str(model.out_features)
        meta["rank"] = str(model.rank)
        meta["degree"] = str(model.degree)
        meta["dtype"] = str(model.dtype)

        tensors["lambdas"] = model.lambdas.detach().contiguous()
        for d in range(model.in_features):
            tensors[f"factor_{d}"] = model.factors[d].detach().contiguous()

    elif isinstance(model, TensorTrainKANLayer):
        meta["model_type"] = "TensorTrainKANLayer"
        meta["in_features"] = str(model.in_features)
        meta["out_features"] = str(model.out_features)
        meta["ranks"] = json.dumps(model.ranks)
        meta["degree"] = str(model.degree)
        meta["dtype"] = str(model.dtype)

        for d in range(model.in_features):
            tensors[f"core_{d}"] = model.cores[d].detach().contiguous()

    elif isinstance(model, TDFFNet):
        meta["model_type"] = "TDFFNet_CP"
        meta["spatial_dim"] = str(model.spatial_dim)
        meta["rank"] = str(model.rank)
        meta["degree"] = str(model.degree)

        tensors["lambdas"] = torch.from_numpy(model.lambdas).contiguous()
        for d in range(model.spatial_dim):
            tensors[f"factor_{d}"] = torch.from_numpy(model.factors[d]).contiguous()

    elif isinstance(model, TensorTrainKAN):
        meta["model_type"] = "TensorTrainKAN"
        meta["spatial_dim"] = str(model.spatial_dim)
        meta["ranks"] = json.dumps([int(r) for r in model.ranks])
        meta["degree"] = str(model.degree)

        for d in range(model.spatial_dim):
            tensors[f"core_{d}"] = torch.from_numpy(model.cores[d]).contiguous()
    else:
        raise TypeError(f"Unsupported model type for safetensors export: {type(model)}")

    # Ensure all metadata values are strings for SafeTensors spec
    meta_str = {k: str(v) for k, v in meta.items()}
    save_file(tensors, filepath, metadata=meta_str)


def load_kan_safetensors(
    filepath: str,
    device: str = "cpu",
    as_torch: bool = True
) -> Union[ContinuousKANLayer, TensorTrainKANLayer, TDFFNet, TensorTrainKAN]:
    r"""
    Wczytuje model KAN z formatu .safetensors rekonstruując strukturę tensorową z metadanych.
    """
    with safetensors.safe_open(filepath, framework="pt", device=device) as f:
        meta = f.metadata() or {}
        model_type = meta.get("model_type")

        if model_type in ("ContinuousKANLayer", "TDFFNet_CP"):
            in_features = int(meta.get("in_features", meta.get("spatial_dim", "0")))
            out_features = int(meta.get("out_features", "1"))
            rank = int(meta.get("rank", "16"))
            degree = int(meta.get("degree", "5"))
            
            lambdas = f.get_tensor("lambdas")
            factors = [f.get_tensor(f"factor_{d}") for d in range(in_features)]

            if as_torch:
                dtype = lambdas.dtype
                layer = ContinuousKANLayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    degree=degree,
                    dtype=dtype,
                    device=torch.device(device)
                )
                with torch.no_grad():
                    layer.lambdas.copy_(lambdas)
                    for d in range(in_features):
                        layer.factors[d].copy_(factors[d])
                return layer
            else:
                assert out_features == 1, "NumPy TDFFNet only supports single output channel"
                tdff = TDFFNet(spatial_dim=in_features, rank=rank, degree=degree)
                tdff.lambdas = lambdas.cpu().numpy()
                tdff.factors = [fac.cpu().numpy() for fac in factors]
                return tdff

        elif model_type in ("TensorTrainKANLayer", "TensorTrainKAN"):
            in_features = int(meta.get("in_features", meta.get("spatial_dim", "0")))
            out_features = int(meta.get("out_features", "1"))
            degree = int(meta.get("degree", "5"))
            ranks = json.loads(meta.get("ranks", "[]"))

            cores = [f.get_tensor(f"core_{d}") for d in range(in_features)]

            if as_torch:
                dtype = cores[0].dtype
                layer = TensorTrainKANLayer(
                    in_features=in_features,
                    out_features=out_features,
                    ranks=ranks,
                    degree=degree,
                    dtype=dtype,
                    device=torch.device(device)
                )
                with torch.no_grad():
                    for d in range(in_features):
                        layer.cores[d].copy_(cores[d])
                return layer
            else:
                assert out_features == 1, "NumPy TensorTrainKAN only supports single output channel"
                tt_kan = TensorTrainKAN(spatial_dim=in_features, ranks=ranks, degree=degree)
                tt_kan.cores = [c.cpu().numpy() for c in cores]
                return tt_kan

        else:
            raise ValueError(f"Unknown KAN model type in safetensors metadata: {model_type}")
