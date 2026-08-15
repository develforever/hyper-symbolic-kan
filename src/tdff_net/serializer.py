import json
import os
import numpy as np
from typing import Dict, Any, Union
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.tt_kan import TensorTrainKAN

class KANSerializer:
    """
    Standardowy serializator modeli KAN do formatów JSON oraz binarnych buforów WebGPU / WGSL.
    """
    @staticmethod
    def cp_to_dict(model: TDFFNet) -> Dict[str, Any]:
        return {
            "type": "TDFFNet_CP",
            "spatial_dim": int(model.spatial_dim),
            "rank": int(model.rank),
            "degree": int(model.degree),
            "lambdas": [float(x) for x in model.lambdas],
            "factors": [f.tolist() for f in model.factors] # (D, R, K+1)
        }

    @staticmethod
    def cp_from_dict(data: Dict[str, Any]) -> TDFFNet:
        model = TDFFNet(
            spatial_dim=data["spatial_dim"],
            rank=data["rank"],
            degree=data["degree"]
        )
        model.lambdas = np.array(data["lambdas"], dtype=np.float64)
        model.factors = [np.array(f, dtype=np.float64) for f in data["factors"]]
        return model

    @staticmethod
    def tt_to_dict(model: TensorTrainKAN) -> Dict[str, Any]:
        return {
            "type": "TensorTrainKAN",
            "spatial_dim": int(model.spatial_dim),
            "ranks": [int(r) for r in model.ranks],
            "degree": int(model.degree),
            "cores": [c.tolist() for c in model.cores] # List of 3D arrays (r_{d-1}, K+1, r_d)
        }

    @staticmethod
    def to_webgpu_buffers(model: TDFFNet) -> Dict[str, Any]:
        """
        Eksportuje wagi modelu do formatu zoptymalizowanego pod WebGPU Uniform/Storage Buffer.
        Wymiary są wyrównane do 16 bajtów (vec4 alignment w WGSL).
        """
        D = model.spatial_dim
        R = model.rank
        K1 = model.degree + 1
        
        factors_flat = np.zeros((D, R, K1), dtype=np.float32)
        for d in range(D):
            factors_flat[d] = model.factors[d].astype(np.float32)
            
        lambdas_f32 = model.lambdas.astype(np.float32)
        
        return {
            "meta": {
                "spatial_dim": D,
                "rank": R,
                "degree": model.degree,
                "k1": K1
            },
            "lambdas": lambdas_f32.tolist(),
            "factors": factors_flat.ravel().tolist() # Flat float array for GPU storage
        }

    @staticmethod
    def save_json(model: Union[TDFFNet, TensorTrainKAN], filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if isinstance(model, TDFFNet):
            data = KANSerializer.cp_to_dict(model)
        elif isinstance(model, TensorTrainKAN):
            data = KANSerializer.tt_to_dict(model)
        else:
            raise ValueError("Nieobsługiwany typ modelu KAN")
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(filepath: str) -> Union[TDFFNet, TensorTrainKAN]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        mtype = data.get("type")
        if mtype == "TDFFNet_CP":
            return KANSerializer.cp_from_dict(data)
        elif mtype == "TensorTrainKAN":
            model = TensorTrainKAN(
                spatial_dim=data["spatial_dim"],
                ranks=data["ranks"],
                degree=data["degree"]
            )
            model.cores = [np.array(c, dtype=np.float64) for c in data["cores"]]
            return model
        else:
            raise ValueError(f"Nieznany format: {mtype}")
