"""
Funções de perda e métricas para análise de sobrevida em PDAC.

Contém:
  - `cox_ph_loss`: negative partial log-likelihood do modelo de Cox (Breslow),
    usada para treinar a cabeça de risco da fusão.
  - `multimodal_cox_loss`: perda de Cox da fusão + regularização de balanceamento
    entre modalidades (cabeças unimodais), mecanismo (iii) da Seção 6.1.
  - `concordance_index`: C-index de Harrell (métrica de avaliação, sem gradiente).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


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


def multimodal_cox_loss(
    out: dict[str, Any],
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    lambda_aux: float = 0.0,
    lambda_balance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perda de Cox da fusão + regularização de balanceamento entre modalidades.

    L = L_cox(risco fundido)
        + lambda_aux     · média_m L_cox(risco unimodal_m)      -- cada modalidade deve ser preditiva
        + lambda_balance · var_m  L_cox(risco unimodal_m)       -- e de forma equilibrada

    Precisa de `out["aux_risk"]` (fusão com `aux_heads=True`). Sem ele, ou com os
    dois pesos em zero, degenera para `cox_ph_loss(out["risk"], ...)`.

    Args:
        out: saída de `MultimodalPDACModel.forward` (`risk` e, opcional, `aux_risk`).
        time, event: rótulos de sobrevida (B,).
        lambda_aux, lambda_balance: pesos dos dois termos de regularização.

    Returns:
        (loss escalar, dict com as componentes escalares para logging).

    TODO: mascarar do risk set as amostras sem a modalidade `m` na perda unimodal
    (hoje elas entram com risco ~constante -> ruído limitado).
    """
    main = cox_ph_loss(out["risk"], time, event)
    parts: dict[str, float] = {"loss_fused": float(main.detach())}

    aux = out.get("aux_risk")
    if not aux or (lambda_aux == 0.0 and lambda_balance == 0.0):
        return main, parts

    per_mod = []
    for name, risk_m in aux.items():
        l_m = cox_ph_loss(risk_m, time, event)
        per_mod.append(l_m)
        parts[f"loss_aux_{name}"] = float(l_m.detach())

    losses = torch.stack(per_mod)
    l_aux = losses.mean()
    l_bal = losses.var(unbiased=False) if losses.numel() > 1 else losses.sum() * 0.0
    parts["loss_aux"] = float(l_aux.detach())
    parts["loss_balance"] = float(l_bal.detach())

    total = main + lambda_aux * l_aux + lambda_balance * l_bal
    parts["loss_total"] = float(total.detach())
    return total, parts


def multitask_loss(
    out: dict[str, Any],
    batch: dict[str, Any],
    *,
    lambda_aux: float = 0.0,
    lambda_balance: float = 0.0,
    w_diagnosis: float = 0.5,
    w_subtype: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perda multitarefa da Figura 4: prognóstico (Cox + balanceamento) + diagnóstico
    (BCE) + subtipagem molecular (cross-entropy). Rótulos ausentes (< 0) são
    mascarados por amostra.
    """
    total, parts = multimodal_cox_loss(
        out, batch["time"], batch["event"],
        lambda_aux=lambda_aux, lambda_balance=lambda_balance,
    )

    if "dx_logit" in out and batch.get("dx") is not None and w_diagnosis > 0:
        dx = batch["dx"].float().view(-1)
        m = dx >= 0
        if m.any():
            l_dx = F.binary_cross_entropy_with_logits(out["dx_logit"].view(-1)[m], dx[m])
            total = total + w_diagnosis * l_dx
            parts["loss_dx"] = float(l_dx.detach())

    if "subtype_logits" in out and batch.get("subtype") is not None and w_subtype > 0:
        st = batch["subtype"].long().view(-1)
        m = st >= 0
        if m.any():
            l_st = F.cross_entropy(out["subtype_logits"][m], st[m])
            total = total + w_subtype * l_st
            parts["loss_subtype"] = float(l_st.detach())

    parts["loss_total"] = float(total.detach())
    return total, parts


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
