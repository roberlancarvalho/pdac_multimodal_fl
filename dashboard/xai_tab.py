"""Aba "Explicabilidade" -- Grad-CAM 3D (Ramo A) e SHAP unimodal (Ramos C e D)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.charts import signed_shap_chart
from dashboard.compute import (
    compute_clinical_shap,
    compute_genomic_shap,
    compute_gradcam,
    load_model_cfg,
)

GENES = ["KRAS", "TP53", "SMAD4", "CDKN2A"]


def _gradcam_tab(model_path: Path) -> None:
    import numpy as np

    ctrl = st.container(horizontal=True)
    seed = int(ctrl.number_input("Seed do volume sintético", value=0, step=1, key="gc_seed"))
    dim = ctrl.slider("Lado do volume (D=H=W)", 32, 96, 48, step=16, key="gc_dim")

    if st.button("Gerar Grad-CAM", type="primary", key="gc_btn"):
        with st.spinner("Retropropagando o risco pelo DenseNet3D…"):
            st.session_state["gc_result"] = compute_gradcam(
                str(model_path), model_path.stat().st_mtime, seed, dim, dim, dim
            )

    if "gc_result" not in st.session_state:
        return

    vol, cam = st.session_state["gc_result"]
    z = st.slider("Fatia axial", 0, vol.shape[0] - 1, vol.shape[0] // 2, key="gc_z")

    metrics = st.container(horizontal=True)
    metrics.metric("Volume", "×".join(map(str, vol.shape)), border=True)
    metrics.metric(
        "Fração saliente (CAM > 0,5)", f"{float((cam > 0.5).mean()):.1%}", border=True,
        help="Proporção de voxels que mais influenciaram o risco predito.",
    )

    slice_ = vol[z]
    slice_ = (slice_ - slice_.min()) / (float(np.ptp(slice_)) + 1e-8)
    overlay = np.zeros((*slice_.shape, 4))
    overlay[..., 0] = 1.0
    overlay[..., 3] = np.clip(cam[z], 0, 1) * 0.6

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**TC sintética (fatia)**")
        st.caption("Entrada original do Ramo A, sem realce.")
        st.image(slice_, clamp=True, width="stretch")
    with c2, st.container(border=True):
        st.markdown("**Grad-CAM sobreposto**")
        st.caption("Vermelho = região que mais empurrou o risco predito para cima.")
        st.image(np.stack([slice_] * 3, axis=-1), clamp=True, width="stretch")
        st.image(overlay, clamp=True, width="stretch")


def _genomic_shap_tab(model_path: Path) -> None:
    st.markdown("**Status mutacional do paciente sintético**")
    cols = st.columns(4)
    mutations = []
    for gene, col in zip(GENES, cols, strict=True):
        choice = col.segmented_control(
            gene, ["wt", "mut"], default="mut" if gene in ("KRAS", "TP53") else "wt", key=f"shap_{gene}"
        )
        mutations.append(1 if choice == "mut" else 0)
    nsamples = st.slider("Amostras do KernelExplainer", 32, 512, 200, step=32, key="shap_ns")

    if not st.button("Calcular SHAP", type="primary", key="shap_btn"):
        return

    with st.spinner("Estimando valores SHAP…"):
        res = compute_genomic_shap(str(model_path), model_path.stat().st_mtime, tuple(mutations), nsamples)

    metrics = st.container(horizontal=True)
    metrics.metric("Risco base (tudo wt)", f"{res['base_value']:+.3f}", border=True)
    metrics.metric("Risco predito", f"{res['prediction']:+.3f}", border=True)

    df = pd.DataFrame({"gene": res["genes"], "shap": res["shap"]})
    with st.container(border=True):
        st.subheader("Contribuição de cada gene driver", divider=False)
        st.caption(
            "Quanto a mutação de cada gene desloca o risco em relação ao paciente todo "
            "*wild-type*. 🔴 empurra o risco pra cima, 🔵 empurra pra baixo. "
            "Soma dos SHAP + risco base = risco predito."
        )
        st.altair_chart(signed_shap_chart(df, "gene"), width="stretch")


def _clinical_shap_tab(model_path: Path, cfg: dict) -> None:
    n_cont = int(cfg.get("clinical_n_continuous", 5))
    cardinalities = list(cfg.get("clinical_cat_cardinalities", [2, 4, 5, 3]))

    st.markdown("**Paciente sintético — variáveis clínicas** (contínuas em z-score)")
    num_cols = st.columns(min(n_cont, 5))
    num = [
        num_cols[i % len(num_cols)].number_input(f"cont_{i}", value=0.0, step=0.5, key=f"cn_{i}")
        for i in range(n_cont)
    ]
    cat_cols = st.columns(min(len(cardinalities), 5))
    cat = [
        cat_cols[i % len(cat_cols)].number_input(
            f"cat_{i} (0..{card - 1})", min_value=0, max_value=card - 1, value=0, key=f"cc_{i}"
        )
        for i, card in enumerate(cardinalities)
    ]
    nsamples = st.slider("Amostras do KernelExplainer", 32, 512, 200, step=32, key="cshap_ns")

    if not st.button("Calcular SHAP clínico", type="primary", key="cshap_btn"):
        return

    with st.spinner("Estimando valores SHAP…"):
        res = compute_clinical_shap(
            str(model_path), model_path.stat().st_mtime,
            tuple(float(x) for x in num), tuple(int(x) for x in cat), nsamples,
        )

    metrics = st.container(horizontal=True)
    metrics.metric("Risco base", f"{res['base_value']:+.3f}", border=True)
    metrics.metric("Risco predito", f"{res['prediction']:+.3f}", border=True)

    df = pd.DataFrame({"campo": res["fields"], "shap": res["shap"]})
    with st.container(border=True):
        st.subheader("Contribuição de cada variável clínica", divider=False)
        st.caption(
            "Deslocamento do risco vs. um paciente de referência (contínuas na média, "
            "categóricas na categoria 0). 🔴 empurra o risco pra cima, 🔵 pra baixo."
        )
        st.altair_chart(signed_shap_chart(df, "campo"), width="stretch")


def render(run_dir: Path) -> None:
    model_path = run_dir / "global_model.pt"
    st.caption(
        "Grad-CAM 3D no Ramo A e SHAP por gene no Ramo C — cf. Seção 6 do artigo "
        "(*explicabilidade via SHAP e Grad-CAM*). Com dados sintéticos os mapas não "
        "têm significado clínico; validam o mecanismo."
    )

    if not model_path.exists():
        st.info("O modelo global desta execução ainda não foi salvo.")
        return

    cfg = load_model_cfg(str(model_path), model_path.stat().st_mtime)
    coattention = cfg.get("fusion_mode", "coattention") == "coattention"

    gc_tab, shap_tab, clinical_tab = st.tabs(
        ["Grad-CAM 3D (Ramo A)", "SHAP genômico (Ramo C)", "SHAP clínico (Ramo D)"]
    )

    with gc_tab:
        _gradcam_tab(model_path)

    with shap_tab:
        if coattention:
            _genomic_shap_tab(model_path)
        else:
            st.warning("SHAP genômico requer `fusion_mode: coattention` (esta execução usa o modo legado).")

    with clinical_tab:
        if coattention and cfg.get("enable_clinical", True):
            _clinical_shap_tab(model_path, cfg)
        else:
            st.warning("SHAP clínico requer `fusion_mode: coattention` e `enable_clinical: true`.")
