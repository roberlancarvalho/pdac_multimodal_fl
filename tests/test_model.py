import pytest
import torch

from models.multimodal_pdac import MultimodalPDACModel


@pytest.mark.parametrize("mode", ["coattention", "transformer"])
def test_forward_both_fusion_modes(model_cfg, batch, mode):
    model = MultimodalPDACModel(**{**model_cfg, "fusion_mode": mode})
    out = model(batch)
    assert out["risk"].shape == (4, 1)
    assert out["dx_logit"].shape == (4, 1)
    assert out["subtype_logits"].shape == (4, 2)
    for key in ("risk", "fused", "dx_logit", "subtype_logits"):
        assert torch.isfinite(out[key]).all(), key


def test_gradients_flow_and_no_nan(model_cfg, batch):
    model = MultimodalPDACModel(**model_cfg)
    out = model(batch)
    loss = out["risk"].sum() + out["dx_logit"].sum() + out["subtype_logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "nenhum gradiente propagado"
    assert all(torch.isfinite(g).all() for g in grads)


def test_missing_modalities_still_predict(model_cfg, batch):
    model = MultimodalPDACModel(**model_cfg)
    only_genomics = {k: batch[k] for k in ("mutation_status", "variant_type", "vaf", "time", "event")}
    out = model(only_genomics)
    assert out["risk"].shape == (4, 1)
    assert torch.isfinite(out["risk"]).all()


def test_raises_without_any_modality(model_cfg):
    model = MultimodalPDACModel(**model_cfg)
    with pytest.raises(ValueError):
        model({"time": torch.zeros(2), "event": torch.zeros(2)})
