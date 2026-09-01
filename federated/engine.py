"""
Laços locais de treino/avaliação executados em cada cliente federado.

Mantido separado de `client.py` para que a lógica de otimização seja testável
isoladamente e reutilizável fora do Flower.

Treino: perda **multitarefa** (Figura 4 do artigo) -- prognóstico de sobrevida
(Cox + regularização de balanceamento entre modalidades), diagnóstico (BCE) e
subtipagem molecular (cross-entropy). Ver `utils.losses.multitask_loss`.

Logging opcional: passe um `SummaryWriter` do TensorBoard em `writer` (+ `step`)
para as curvas locais (`local/loss_*`, `local/eval_loss`, `local/c_index`,
`local/gate_*`, `local/auc_dx`, `local/acc_subtype`). O servidor registra as
métricas *por rodada* e histogramas em `federated/reporting.py`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from models.fusion_attention import MODALITIES
from models.multimodal_pdac import MultimodalPDACModel
from utils.losses import concordance_index, cox_ph_loss, multitask_loss


def train_one_epoch(
    model: MultimodalPDACModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_aux: float = 0.0,
    lambda_balance: float = 0.0,
    w_diagnosis: float = 0.0,
    w_subtype: float = 0.0,
    writer: Any | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Uma época de treino local. Retorna as componentes de perda médias."""
    model.train()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss, parts = multitask_loss(
            out, {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()},
            lambda_aux=lambda_aux, lambda_balance=lambda_balance,
            w_diagnosis=w_diagnosis, w_subtype=w_subtype,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        parts.setdefault("loss_total", float(loss.detach()))
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1

    means = {k: v / max(n_batches, 1) for k, v in sums.items()}
    if writer is not None and step is not None:
        for k, v in means.items():
            writer.add_scalar(f"local/{k}", v, step)
        writer.flush()
    means["loss"] = means.get("loss_total", means.get("loss_fused", 0.0))
    return means


@torch.no_grad()
def evaluate(
    model: MultimodalPDACModel,
    loader: DataLoader,
    device: torch.device,
    *,
    writer: Any | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Avaliação local: C-index (prognóstico), AUC (diagnóstico), acurácia (subtipo),
    perda de Cox e contribuição média por modalidade."""
    model.eval()
    risks, times, events = [], [], []
    dx_logits, dx_true = [], []
    st_pred, st_true = [], []
    gate_sums: dict[str, float] = {}
    gate_counts: dict[str, int] = {}
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        out = model(batch)
        loss = cox_ph_loss(out["risk"], batch["time"].to(device), batch["event"].to(device))
        total_loss += float(loss)
        n_batches += 1
        risks.append(out["risk"].view(-1).cpu())
        times.append(batch["time"].view(-1).cpu())
        events.append(batch["event"].view(-1).cpu())
        if "dx_logit" in out and batch.get("dx") is not None:
            dx_logits.append(out["dx_logit"].view(-1).cpu())
            dx_true.append(batch["dx"].view(-1).cpu())
        if "subtype_logits" in out and batch.get("subtype") is not None:
            st_pred.append(out["subtype_logits"].argmax(dim=1).cpu())
            st_true.append(batch["subtype"].view(-1).cpu())
        for m, g in out.get("modality_gate", {}).items():
            gate_sums[m] = gate_sums.get(m, 0.0) + float(g.sum())
            gate_counts[m] = gate_counts.get(m, 0) + g.numel()

    risk = torch.cat(risks)
    time = torch.cat(times)
    event = torch.cat(events)
    c_index = float(concordance_index(risk, time, event))
    metrics = {"loss": total_loss / max(n_batches, 1), "c_index": c_index}

    if dx_logits:
        y = torch.cat(dx_true).numpy()
        p = torch.sigmoid(torch.cat(dx_logits)).numpy()
        if 0 < y.sum() < len(y):  # AUC precisa das duas classes
            from sklearn.metrics import roc_auc_score

            metrics["auc_dx"] = float(roc_auc_score(y, p))
    if st_pred:
        yp, yt = torch.cat(st_pred), torch.cat(st_true)
        valid = yt >= 0
        if valid.any():
            metrics["acc_subtype"] = float((yp[valid] == yt[valid]).float().mean())

    for m in MODALITIES:
        if gate_counts.get(m):
            metrics[f"gate_{m}"] = gate_sums[m] / gate_counts[m]

    if writer is not None and step is not None:
        for key in ("loss", "c_index", "auc_dx", "acc_subtype"):
            if key in metrics and metrics[key] == metrics[key]:
                writer.add_scalar(f"local/eval_{key}" if key == "loss" else f"local/{key}", metrics[key], step)
        for m in MODALITIES:
            if f"gate_{m}" in metrics:
                writer.add_scalar(f"local/gate_{m}", metrics[f"gate_{m}"], step)
        writer.flush()

    return metrics
