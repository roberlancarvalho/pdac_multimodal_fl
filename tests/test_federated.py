import numpy as np
import torch

from federated.server import build_strategy, make_central_eval_fn, weighted_average
from models.multimodal_pdac import MultimodalPDACModel
from utils.common import batchnorm_state_keys, get_parameters, set_parameters


def test_parameter_round_trip(model_cfg):
    a = MultimodalPDACModel(**model_cfg)
    b = MultimodalPDACModel(**model_cfg)
    set_parameters(b, get_parameters(a))
    for pa, pb in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        assert torch.equal(pa, torch.as_tensor(pb))


def test_fedbn_skip_preserves_batchnorm(model_cfg):
    model = MultimodalPDACModel(**model_cfg)
    bn = batchnorm_state_keys(model)
    assert bn, "o modelo deveria ter camadas BatchNorm (DenseNet3D)"
    key = sorted(bn)[0]
    before = model.state_dict()[key].clone()
    zeros = [np.zeros_like(v.numpy()) for v in model.state_dict().values()]
    set_parameters(model, zeros, skip_keys=bn)
    assert torch.equal(model.state_dict()[key], before)
    non_bn = next(k for k, v in model.state_dict().items() if k not in bn and v.numel() > 1)
    assert float(model.state_dict()[non_bn].abs().sum()) == 0.0


def test_weighted_average_per_key_denominator():
    metrics = [(10, {"c_index": 0.8, "auc_dx": 0.6}), (10, {"c_index": 0.6})]
    agg = weighted_average(metrics)
    assert abs(agg["c_index"] - 0.7) < 1e-6      # (0.8*10 + 0.6*10) / 20
    assert abs(agg["auc_dx"] - 0.6) < 1e-6       # só um cliente -> denominador = 10


def test_dp_wrapper_selected(model_cfg):
    cfg = {
        "model": model_cfg,
        "federated": {
            "server_address": "x", "num_rounds": 1,
            "fraction_fit": 1.0, "fraction_evaluate": 1.0,
            "min_fit_clients": 2, "min_evaluate_clients": 2, "min_available_clients": 2,
            "strategy": "FedAvg",
            "dp": {"enabled": True, "clip_norm": 1.0, "noise_multiplier": 1.0},
        },
        "data": {"manifest_csv": "", "modality_dropout": 0.0},
        "train": {"batch_size": 4},
    }
    s = build_strategy(cfg)
    assert type(s).__name__ == "DifferentialPrivacyServerSideFixedClipping"


def test_central_eval_fn(model_cfg):
    cfg = {
        "model": model_cfg,
        "federated": {"central_eval": {"enabled": True, "samples": 12, "seed": 5}},
        "data": {"manifest_csv": "", "modality_dropout": 0.1},
        "train": {"batch_size": 4},
    }
    fn = make_central_eval_fn(cfg)
    loss, metrics = fn(1, get_parameters(MultimodalPDACModel(**model_cfg)), {})
    assert np.isfinite(loss)
    assert "central_c_index" in metrics
