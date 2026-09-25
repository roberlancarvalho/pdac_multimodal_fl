"""Aba "Atenção — histopatologia" -- pesos do Ramo B (attention-MIL) sobre uma bag sintética."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.charts import attention_heatmap
from dashboard.compute import compute_attention


def render(run_dir: Path) -> None:
    model_path = run_dir / "global_model.pt"
    st.caption(
        "O Ramo B agrega embeddings de patches de WSI por *attention-MIL*. "
        "Aqui os patches são aleatórios (sintéticos), então a atenção fica ~uniforme "
        "— a aba demonstra o encanamento de interpretabilidade."
    )

    if not model_path.exists():
        st.info("O modelo global desta execução ainda não foi salvo (aguarde a 1ª rodada).")
        return

    ctrl = st.container(horizontal=True)
    n_patches = ctrl.slider(
        "Patches na *bag*", 32, 512, 200, step=32,
        help="Quantos patches de WSI compõem a lâmina do paciente sintético.",
    )
    patient_seed = int(ctrl.number_input(
        "Seed do paciente sintético", value=0, step=1,
        help="Muda o paciente sintético gerado (embeddings de patches aleatórios).",
    ))
    top_k = ctrl.slider("Top-k patches", 5, 50, 15, help="Quantos patches de maior atenção listar na tabela.")

    if not st.button(
        "Gerar visualização de atenção",
        type="primary",
        help="Roda o Ramo B (attention-MIL) do modelo global sobre a lâmina "
        "sintética e mostra o peso de atenção de cada patch.",
    ):
        return

    with st.spinner("Rodando o Ramo B…"):
        attn = compute_attention(str(model_path), model_path.stat().st_mtime, n_patches, patient_seed)
    df_attn = pd.DataFrame({"patch": range(len(attn)), "atencao": attn})

    metrics = st.container(horizontal=True)
    metrics.metric("Patches", len(attn), border=True, help="Tamanho da bag processada.")
    metrics.metric(
        "Atenção máx.", f"{attn.max():.4f}", border=True,
        help="Maior peso atribuído a um único patch (os pesos somam 1).",
    )
    metrics.metric(
        "Concentração (máx/média)", f"{attn.max() / attn.mean():.2f}×", border=True,
        help="1× = atenção uniforme; quanto maior, mais o modelo se apoia em poucos patches.",
    )

    c1, c2 = st.columns([2, 3])
    with c1, st.container(border=True):
        st.subheader(f"Top-{top_k} patches por atenção", divider=False)
        st.caption("Os patches que mais pesaram no embedding final da lâmina.")
        st.dataframe(
            df_attn.nlargest(top_k, "atencao").reset_index(drop=True),
            hide_index=True,
            width="stretch",
            column_config={
                "atencao": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0.0, max_value=float(attn.max())
                ),
            },
        )
    with c2, st.container(border=True):
        st.subheader("Mapa de atenção (grade pseudo-espacial)", divider=False)
        st.caption(
            "Mais claro = maior peso de atenção. Os patches são dispostos numa grade só "
            "para visualização — não corresponde à posição real na lâmina."
        )
        st.altair_chart(attention_heatmap(attn), width="stretch")
