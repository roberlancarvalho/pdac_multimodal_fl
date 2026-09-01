"""
Funções de perda e métricas para análise de sobrevida em PDAC.

Contém:
  - `cox_ph_loss`: negative partial log-likelihood do modelo de Cox (Breslow),
    usada para treinar a cabeça de risco da fusão.
  - `concordance_index`: C-index de Harrell (métrica de avaliação, sem gradiente).
"""

from __future__ import annotations

import torch


def cox_ph_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative partial log-likelihood de Cox (aproximação de Breslow).

    Args:
        risk: Tensor (B,) ou (B, 1) -- log-hazard predito pelo modelo.
        time: Tensor (B,) -- tempo de follow-up / evento (mesma unidade para todos).
        event: Tensor (B,) -- 1 se o evento (óbito/progressão) ocorreu, 0 se censurado.

    Returns:
        Escalar -- perda a ser minimizada. Retorna 0 se não houver eventos no lote.
    """
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1).float()

    if event.sum() == 0:
        return risk.sum() * 0.0

    # Ordena por tempo decrescente: o "risk set" de i são os índices <= i.
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)  # log sum_{j in risk set} exp(risk_j)
    per_event = (risk - log_cumsum) * event
    return -per_event.sum() / (event.sum() + eps)


@torch.no_grad()
def concordance_index(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> float:
    """C-index de Harrell.

    Fração de pares comparáveis em que o paciente com maior risco predito
    teve o evento antes. 0.5 = aleatório, 1.0 = ordenação perfeita.
    """
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1).bool()

    n = time.size(0)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if time[j] <= time[i]:
                continue
            permissible += 1.0
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5

    return concordant / permissible if permissible > 0 else float("nan")
