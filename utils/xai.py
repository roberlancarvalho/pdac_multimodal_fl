"""
Explicabilidade (XAI) do Pipeline Multimodal Federado para PDAC.

  - `GradCAM3D`      : mapa de saliência 3D para o Ramo A (radiômico), retropropagando
                       o risco fundido até um mapa de features convolucional do DenseNet3D.
  - `genomics_shap`  : valores SHAP por gene driver (KRAS/TP53/SMAD4/CDKN2A) sobre a
                       cabeça de risco unimodal genômica da fusão (`fusion_aux_heads=True`).

Ambos são demonstrados no painel (aba "Explicabilidade"). Com dados sintéticos os
resultados não têm significado clínico -- validam o mecanismo.

Referências: Selvaraju et al. 2017 (Grad-CAM); Lundberg & Lee 2017 (SHAP).
Cf. Seção 6 do artigo ("explicabilidade via SHAP e Grad-CAM").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.branch_c_genomics import PDAC_DRIVER_GENES
from models.multimodal_pdac import MultimodalPDACModel


# --------------------------------------------------------------------------- #
# Grad-CAM 3D -- Ramo A                                                        #
# --------------------------------------------------------------------------- #
class GradCAM3D:
    """Grad-CAM para um `nn.Module` convolucional 3D dentro do modelo multimodal.

    Uso:
        cam = GradCAM3D(model)
        heat = cam(batch)          # (B, D, H, W) em [0, 1], no tamanho do volume
        cam.remove()
    """

    def __init__(self, model: MultimodalPDACModel, target_layer: torch.nn.Module | None = None) -> None:
        self.model = model
        # Camada alvo padrão: BN final do bloco de features do DenseNet3D (1024 canais).
        if target_layer is None:
            target_layer = model.branch_a.backbone.features.norm5
        self.target = target_layer
        self._activations: list[torch.Tensor] = []
        self._gradients: list[torch.Tensor] = []
        self._h_fwd = target_layer.register_forward_hook(self._save_activations)
        self._h_bwd = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output) -> None:
        self._activations.append(output.detach())

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        self._gradients.insert(0, grad_out[0].detach())  # backward: ordem inversa

    def __call__(self, batch: dict, out_index: int = 0) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._activations.clear()
        self._gradients.clear()

        with torch.enable_grad():
            out = self.model(batch)
            score = out["risk"][:, out_index].sum()
            score.backward()

        # Média entre chamadas do backbone (ex.: 1 por fase AP/VP).
        act = torch.stack(self._activations, 0).mean(0)   # (B, C, d, h, w)
        grad = torch.stack(self._gradients, 0).mean(0)
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)           # (B, C, 1, 1, 1)
        raw = (weights * act).sum(dim=1)                           # (B, d, h, w)
        cam = torch.relu(raw)
        # Fallback por amostra: se a ReLU zerou tudo (comum em modelos pouco
        # treinados), usa a magnitude |raw| para não devolver um mapa vazio.
        empty = cam.flatten(1).amax(1) < 1e-8
        cam = torch.where(empty.view(-1, 1, 1, 1), raw.abs(), cam)

        ref_vol = batch["ct_volume"]
        cam = F.interpolate(
            cam.unsqueeze(1), size=ref_vol.shape[-3:], mode="trilinear", align_corners=False
        ).squeeze(1)                                               # (B, D, H, W)
        flat = cam.flatten(1)
        cam = (cam - flat.min(1).values.view(-1, 1, 1, 1)) / (
            flat.max(1).values.view(-1, 1, 1, 1) - flat.min(1).values.view(-1, 1, 1, 1) + 1e-8
        )
        if was_training:
            self.model.train()
        return cam.detach()

    def remove(self) -> None:
        self._h_fwd.remove()
        self._h_bwd.remove()


def radiomics_gradcam(
    model: MultimodalPDACModel, batch: dict, out_index: int = 0
) -> torch.Tensor:
    """Atalho: retorna o mapa Grad-CAM (B, D, H, W) para o `ct_volume` de `batch`."""
    cam = GradCAM3D(model)
    try:
        return cam(batch, out_index=out_index)
    finally:
        cam.remove()


# --------------------------------------------------------------------------- #
# SHAP genômico -- Ramo C                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _genomics_only_risk(model: MultimodalPDACModel, mutation_status: torch.Tensor) -> torch.Tensor:
    """Log-hazard usando SÓ a genômica (cabeça unimodal da fusão de co-atenção)."""
    fu = model.fusion
    if not getattr(fu, "aux_heads", False):
        raise RuntimeError(
            "SHAP genômico requer fusion_mode='coattention' e model.fusion_aux_heads=True."
        )
    tok = model.branch_c(mutation_status.long(), return_tokens=True)  # (B, n_genes, D)
    h = fu.in_norm["genomics"](torch.nan_to_num(tok)) + fu.type_embed["genomics"]
    q = fu.pool_query["genomics"].expand(h.size(0), -1, -1)
    vec, _ = fu.pool_attn["genomics"](q, h, h, need_weights=False)
    return fu.aux_risk["genomics"](vec.squeeze(1))  # (B, n_outputs)


def genomics_shap(
    model: MultimodalPDACModel,
    mutation_status: torch.Tensor,
    n_states: int = 3,
    nsamples: int = 200,
) -> dict:
    """Valores SHAP por gene driver para a cabeça de risco unimodal genômica.

    Args:
        model: modelo treinado (co-atenção com `fusion_aux_heads=True`).
        mutation_status: Tensor long (1, n_genes) -- o paciente a explicar.
        n_states: nº de estados por gene (para o `background`).
        nsamples: amostras do KernelExplainer.

    Returns:
        dict com `genes`, `shap` (por gene), `base_value`, `prediction`.
    """
    import numpy as np
    import shap

    model.eval()
    n_genes = mutation_status.shape[1]

    def f(rows: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.rint(rows), dtype=torch.long).clamp_(0, n_states - 1)
        return _genomics_only_risk(model, t)[:, 0].cpu().numpy()

    background = np.zeros((1, n_genes), dtype=float)  # tudo wild-type
    explainer = shap.KernelExplainer(f, background)
    x = mutation_status.cpu().numpy().astype(float)
    values = explainer.shap_values(x, nsamples=nsamples, silent=True)
    values = np.asarray(values).reshape(-1)[:n_genes]

    return {
        "genes": list(PDAC_DRIVER_GENES[:n_genes]),
        "shap": values.tolist(),
        "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        "prediction": float(f(x)[0]),
    }


@torch.no_grad()
def _clinical_only_risk(
    model: MultimodalPDACModel, clinical_num: torch.Tensor, clinical_cat: torch.Tensor
) -> torch.Tensor:
    """Log-hazard usando SÓ o ramo clínico (cabeça unimodal da fusão de co-atenção)."""
    fu = model.fusion
    if not getattr(fu, "aux_heads", False) or not getattr(model, "enable_clinical", False):
        raise RuntimeError("SHAP clínico requer fusion_aux_heads=True e enable_clinical=True.")
    tok = model.branch_d(clinical_num.float(), clinical_cat.long(), return_tokens=True)
    h = fu.in_norm["clinical"](torch.nan_to_num(tok)) + fu.type_embed["clinical"]
    q = fu.pool_query["clinical"].expand(h.size(0), -1, -1)
    vec, _ = fu.pool_attn["clinical"](q, h, h, need_weights=False)
    return fu.aux_risk["clinical"](vec.squeeze(1))


def clinical_shap(
    model: MultimodalPDACModel,
    clinical_num: torch.Tensor,
    clinical_cat: torch.Tensor,
    continuous_names: list[str] | None = None,
    categorical_names: list[str] | None = None,
    nsamples: int = 200,
) -> dict:
    """Valores SHAP por campo clínico para a cabeça de risco unimodal do Ramo D.

    `background` = contínuas na média (0 após z-score) e categóricas na moda (0).
    """
    import numpy as np
    import shap

    model.eval()
    n_cont = clinical_num.shape[1]
    n_cat = clinical_cat.shape[1]
    names = (continuous_names or [f"cont_{i}" for i in range(n_cont)]) + (
        categorical_names or [f"cat_{i}" for i in range(n_cat)]
    )

    def f(rows: np.ndarray) -> np.ndarray:
        num = torch.as_tensor(rows[:, :n_cont], dtype=torch.float32)
        cat = torch.as_tensor(np.rint(rows[:, n_cont:]), dtype=torch.long).clamp_min(0)
        return _clinical_only_risk(model, num, cat)[:, 0].cpu().numpy()

    background = np.zeros((1, n_cont + n_cat), dtype=float)
    explainer = shap.KernelExplainer(f, background)
    x = np.concatenate(
        [clinical_num.cpu().numpy().astype(float), clinical_cat.cpu().numpy().astype(float)], axis=1
    )
    values = np.asarray(explainer.shap_values(x, nsamples=nsamples, silent=True)).reshape(-1)

    return {
        "fields": names,
        "shap": values[: len(names)].tolist(),
        "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        "prediction": float(f(x)[0]),
    }
