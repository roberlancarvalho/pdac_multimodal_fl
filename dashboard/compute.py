"""Cálculos pesados (carregam o modelo salvo) — cacheados por caminho + mtime."""

from __future__ import annotations

import streamlit as st


def _load_checkpoint(model_path: str):
    import torch

    return torch.load(model_path, map_location="cpu", weights_only=False)


def load_xai_model(model_path: str):
    """Carrega o modelo global salvo. `strict=False`: execuções anteriores à
    co-atenção têm um state_dict de arquitetura diferente."""
    from models.multimodal_pdac import MultimodalPDACModel

    ckpt = _load_checkpoint(model_path)
    model = MultimodalPDACModel(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, ckpt["model_cfg"]


@st.cache_data(show_spinner=False)
def load_model_cfg(model_path: str, mtime: float) -> dict:
    return _load_checkpoint(model_path)["model_cfg"]


@st.cache_data(show_spinner=False)
def compute_attention(model_path: str, mtime: float, n_patches: int, seed: int):
    import torch

    model, cfg = load_xai_model(model_path)
    gen = torch.Generator().manual_seed(seed)
    bag = torch.randn(1, n_patches, cfg["patch_feat_dim"], generator=gen)
    mask = torch.ones(1, n_patches, dtype=torch.bool)
    with torch.no_grad():
        _, attn = model.branch_b(bag, mask)
    return attn.squeeze(0).numpy()


@st.cache_data(show_spinner=False)
def compute_gradcam(model_path: str, mtime: float, seed: int, dz: int, dy: int, dx: int):
    import torch

    from utils.xai import radiomics_gradcam

    model, cfg = load_xai_model(model_path)
    gen = torch.Generator().manual_seed(seed)
    ct = torch.randn(1, cfg.get("ct_in_channels", 1), dz, dy, dx, generator=gen)
    batch = {"ct_volume": ct, "mutation_status": torch.tensor([[1, 1, 0, 1]])}
    cam = radiomics_gradcam(model, batch)[0].numpy()
    return ct[0, 0].numpy(), cam


@st.cache_data(show_spinner=False)
def compute_genomic_shap(model_path: str, mtime: float, mutations: tuple[int, ...], nsamples: int):
    import torch

    from utils.xai import genomics_shap

    model, _ = load_xai_model(model_path)
    return genomics_shap(model, torch.tensor([list(mutations)]), nsamples=nsamples)


@st.cache_data(show_spinner=False)
def compute_clinical_shap(
    model_path: str, mtime: float, num: tuple[float, ...], cat: tuple[int, ...], nsamples: int
):
    import torch

    from utils.xai import clinical_shap

    model, _ = load_xai_model(model_path)
    return clinical_shap(
        model, torch.tensor([list(num)]), torch.tensor([list(cat)]), nsamples=nsamples
    )
