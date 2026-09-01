import math

import torch

from utils.losses import (
    concordance_index,
    cox_ph_loss,
    multimodal_cox_loss,
    multitask_loss,
)


def test_cox_loss_zero_without_events():
    risk = torch.randn(8)
    time = torch.rand(8) * 10
    event = torch.zeros(8)
    assert float(cox_ph_loss(risk, time, event)) == 0.0


def test_cox_loss_finite_and_lower_for_correct_ordering():
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)
    good = torch.tensor([3.0, 2.0, 1.0, 0.0])   # maior risco morre antes
    bad = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert cox_ph_loss(good, time, event) < cox_ph_loss(bad, time, event)


def test_concordance_index_perfect_and_random():
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.ones(4)
    perfect = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert concordance_index(perfect, time, event) == 1.0
    inverted = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert concordance_index(inverted, time, event) == 0.0


def test_multimodal_cox_loss_components():
    out = {
        "risk": torch.randn(4, 1),
        "aux_risk": {m: torch.randn(4, 1) for m in ("radiomics", "histology", "genomics")},
    }
    time, event = torch.rand(4) * 10, torch.tensor([1.0, 0.0, 1.0, 1.0])
    total, parts = multimodal_cox_loss(out, time, event, lambda_aux=0.3, lambda_balance=0.1)
    assert {"loss_fused", "loss_aux", "loss_balance", "loss_total"} <= set(parts)
    assert math.isfinite(float(total))
    # sem regularização -> degenera para a Cox da fusão
    bare, _ = multimodal_cox_loss(out, time, event)
    assert torch.allclose(bare, cox_ph_loss(out["risk"], time, event))


def test_multitask_loss_masks_missing_labels():
    out = {
        "risk": torch.randn(4, 1),
        "dx_logit": torch.randn(4, 1),
        "subtype_logits": torch.randn(4, 2),
    }
    batch = {
        "time": torch.rand(4) * 10,
        "event": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "dx": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "subtype": torch.tensor([0, 1, -1, 1]),  # -1 mascarado
    }
    total, parts = multitask_loss(out, batch, w_diagnosis=0.5, w_subtype=0.5)
    assert "loss_dx" in parts and "loss_subtype" in parts
    assert math.isfinite(float(total))

    batch_all_missing = dict(batch, subtype=torch.tensor([-1, -1, -1, -1]))
    _, parts2 = multitask_loss(out, batch_all_missing, w_subtype=0.5)
    assert "loss_subtype" not in parts2
