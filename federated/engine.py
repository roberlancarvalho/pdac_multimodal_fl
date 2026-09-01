"""
Laços locais de treino/avaliação executados em cada cliente federado.

Mantido separado de `client.py` para que a lógica de otimização seja testável
isoladamente e reutilizável fora do Flower.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models.multimodal_pdac import MultimodalPDACModel
from utils.losses import concordance_index, cox_ph_loss


def train_one_epoch(
    model: MultimodalPDACModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    model: MultimodalPDACModel,
    loader: DataLoader,
    device: torch.device,
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
    c_index = concordance_index(risk, time, event)
    return {"loss": total_loss / max(n_batches, 1), "c_index": float(c_index)}
