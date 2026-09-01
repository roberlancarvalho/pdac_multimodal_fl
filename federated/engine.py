"""
Laços locais de treino/avaliação executados em cada cliente federado.

Mantido separado de `client.py` para que a lógica de otimização seja testável
isoladamente e reutilizável fora do Flower.

Logging opcional: passe um `SummaryWriter` do TensorBoard em `writer` (+ `step`)
para registrar as curvas locais (`local/train_loss`, `local/eval_loss`,
`local/c_index`). O servidor registra as métricas *por rodada* e os histogramas
de pesos/atenção em `federated/reporting.py`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from models.multimodal_pdac import MultimodalPDACModel
from utils.losses import concordance_index, cox_ph_loss


def train_one_epoch(
    model: MultimodalPDACModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    writer: Any | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Uma época de treino local com a perda de Cox. Retorna métricas médias."""
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = cox_ph_loss(
            out["risk"], batch["time"].to(device), batch["event"].to(device)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    if writer is not None and step is not None:
        writer.add_scalar("local/train_loss", mean_loss, step)
        writer.flush()
    return {"loss": mean_loss}


@torch.no_grad()
def evaluate(
    model: MultimodalPDACModel,
    loader: DataLoader,
    device: torch.device,
    *,
    writer: Any | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Avaliação local: perda de Cox e C-index agregados sobre todo o loader."""
    model.eval()
    risks, times, events = [], [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        out = model(batch)
        loss = cox_ph_loss(
            out["risk"], batch["time"].to(device), batch["event"].to(device)
        )
        total_loss += float(loss)
        n_batches += 1
        risks.append(out["risk"].view(-1).cpu())
        times.append(batch["time"].view(-1).cpu())
        events.append(batch["event"].view(-1).cpu())

    risk = torch.cat(risks)
    time = torch.cat(times)
    event = torch.cat(events)
    c_index = float(concordance_index(risk, time, event))
    mean_loss = total_loss / max(n_batches, 1)

    if writer is not None and step is not None:
        writer.add_scalar("local/eval_loss", mean_loss, step)
        if c_index == c_index:  # não-NaN
            writer.add_scalar("local/c_index", c_index, step)
        writer.flush()

    return {"loss": mean_loss, "c_index": c_index}
