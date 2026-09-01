import torch
from torch.utils.data import DataLoader

from data.dataset import ModalityShapes, SyntheticPDACDataset, collate_multimodal
from federated.engine import evaluate, train_one_epoch
from models.multimodal_pdac import MultimodalPDACModel


def _loader(seed=0):
    shapes = ModalityShapes(ct_shape=(32, 32, 32), n_patches=6, patch_feat_dim=32, ct_channels=2)
    ds = SyntheticPDACDataset(n_samples=16, shapes=shapes, modality_dropout=0.1, seed=seed)
    return DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_multimodal)


def test_train_one_epoch_returns_components(model_cfg):
    model = MultimodalPDACModel(**model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = train_one_epoch(
        model, _loader(), opt, torch.device("cpu"),
        lambda_aux=0.3, lambda_balance=0.1, w_diagnosis=0.5, w_subtype=0.3,
    )
    assert "loss_total" in out and "loss_dx" in out and "loss_subtype" in out
    assert all(v == v for v in out.values())  # sem NaN


def test_evaluate_metrics_keys(model_cfg):
    model = MultimodalPDACModel(**model_cfg)
    m = evaluate(model, _loader(seed=1), torch.device("cpu"))
    assert "c_index" in m and "loss" in m
    assert any(k.startswith("gate_") for k in m)


def test_training_reduces_loss(model_cfg):
    torch.manual_seed(0)
    model = MultimodalPDACModel(**model_cfg)
    loader = _loader()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = train_one_epoch(model, loader, opt, torch.device("cpu"))["loss"]
    for _ in range(4):
        last = train_one_epoch(model, loader, opt, torch.device("cpu"))["loss"]
    assert last < first * 1.5  # otimização não diverge
