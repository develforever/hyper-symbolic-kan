import os
import json
import numpy as np
from src.tdff_net.tensor_field import TDFFNet
from src.tdff_net.serializer import KANSerializer

def test_serializer_roundtrip_json(tmp_path):
    model = TDFFNet(spatial_dim=3, rank=6, degree=3)
    X = np.random.uniform(-0.9, 0.9, (10, 3))
    y_orig = model.evaluate(X)
    
    json_file = str(tmp_path / "test_kan.json")
    KANSerializer.save_json(model, json_file)
    
    loaded_model = KANSerializer.load_json(json_file)
    y_loaded = loaded_model.evaluate(X)
    
    np.testing.assert_allclose(y_orig, y_loaded, atol=1e-12)

def test_webgpu_buffers_format():
    model = TDFFNet(spatial_dim=3, rank=8, degree=4)
    gpu_data = KANSerializer.to_webgpu_buffers(model)
    
    assert "meta" in gpu_data
    assert gpu_data["meta"]["spatial_dim"] == 3
    assert gpu_data["meta"]["rank"] == 8
    assert gpu_data["meta"]["degree"] == 4
    assert gpu_data["meta"]["k1"] == 5
    
    assert len(gpu_data["lambdas"]) == 8
    assert len(gpu_data["factors"]) == 3 * 8 * 5
    
    for val in gpu_data["factors"]:
        assert isinstance(val, float)
        assert np.isfinite(val)
