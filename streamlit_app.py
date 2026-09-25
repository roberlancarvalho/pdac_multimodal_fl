"""
Painel do Pipeline Multimodal Federado para PDAC.

Dispara a simulação federada (`federated/simulation.py`) como subprocesso e
acompanha em tela: status/progresso, C-index e perdas por rodada, métricas por
cliente, atenção da histopatologia e explicabilidade (Grad-CAM, SHAP).

Executar:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_sortables import sort_items

from dashboard import attention_tab, data_tab, tour, training_tab, xai_tab
from dashboard.io import list_runs, read_json
from dashboard.process import launch_run, proc_alive, stop_run
from dashboard.training_tab import HIDDEN_HEADER, VISIBLE_HEADER, default_kpi_layout

st.set_page_config(page_title="PDAC FL — painel", page_icon=":material/hub:", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, .18));
        border-top: 3px solid var(--primary-color);
        border-radius: 10px;
        padding: .9rem 1.1rem .6rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, .15);
    }
    div[data-testid="stMetricLabel"] { opacity: .8; letter-spacing: .02em; }
    div[data-testid="stMetricValue"] { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

_SORTABLE_CSS = """
.sortable-container {
    background-color: var(--secondary-background-color, rgba(128, 128, 128, .06));
    border: 1px solid var(--border-color, rgba(128, 128, 128, .25));
    border-radius: 10px;
    padding: 8px;
    min-height: 44px;
}
.sortable-container-header {
    font-weight: 600;
    font-size: .85rem;
    margin-bottom: 6px;
}
.sortable-item {
    background-color: var(--primary-color, #3B7DD8);
    color: white;
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 6px;
    font-size: .85rem;
    cursor: grab;
}
.sortable-item:active { cursor: grabbing; }
"""

if st.session_state.pop("_reset_kpi_layout", False):
    # Precisa rodar antes do componente key="kpi_sortable" ser criado nesta
    # execução -- Streamlit proíbe reatribuir session_state de um widget já instanciado.
    st.session_state.kpi_layout = default_kpi_layout()

with st.sidebar:
    if st.button("Tour do painel", icon=":material/help:", width="stretch",
                  help="Reabre a introdução passo a passo do painel."):
        tour.open_now()
        st.rerun()

    with st.popover("🎛️ Personalizar cards", width="stretch"):
        st.caption(
            "Arraste os cards entre as colunas para reordenar ou esconder "
            "(aba **Treino federado**)."
        )
        layout = st.session_state.get("kpi_layout") or default_kpi_layout()
        result = sort_items(
            [
                {"header": VISIBLE_HEADER, "items": layout[VISIBLE_HEADER]},
                {"header": HIDDEN_HEADER, "items": layout[HIDDEN_HEADER]},
            ],
            multi_containers=True,
            direction="vertical",
            key="kpi_sortable",
            custom_style=_SORTABLE_CSS,
        )
        st.session_state.kpi_layout = {d["header"]: d["items"] for d in result}
        if st.button("Restaurar padrão", width="stretch"):
            st.session_state._reset_kpi_layout = True
            st.rerun()

    st.subheader("Configuração da simulação")

    with st.form("config"):
        num_clients = st.slider(
            "Clientes (instituições)", 2, 6, 2,
            help="Quantos nós federados virtuais participam. Cada um treina só nos "
            "seus dados; o servidor agrega os pesos.",
        )
        num_rounds = st.slider(
            "Rodadas federadas", 1, 20, 3,
            help="Ciclos de treino-local → agregação. Cada rodada leva ~1 min em CPU.",
        )
        local_epochs = st.slider(
            "Épocas locais por rodada", 1, 5, 1,
            help="Passagens completas pelos dados locais de cada cliente antes de "
            "enviar os pesos ao servidor.",
        )
        lr = st.select_slider(
            "Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=3e-4,
            format_func=lambda v: f"{v:.0e}",
            help="Taxa de aprendizado do otimizador AdamW nos clientes.",
        )
        strategy = st.segmented_control(
            "Estratégia de agregação", ["FedAvg", "FedProx", "FedAdam"], default="FedAvg",
            help="Como o servidor combina os pesos dos clientes. FedProx penaliza "
            "desvio do modelo global (bom p/ dados não-IID); FedAdam usa momento no servidor.",
        )
        fedbn = st.toggle(
            "FedBN (BatchNorm local)", value=False,
            help="Mantém as camadas BatchNorm de cada cliente locais — mitiga "
            "*shift* de distribuição entre instituições (não-IID; Li et al. 2021).",
        )
        dp_enabled = st.toggle(
            "DP-FedAvg (privacidade)", value=False,
            help="Clipping da atualização de cada cliente + ruído gaussiano no "
            "agregado (servidor) — treinamento com privacidade / LGPD.",
        )
        dp_noise = st.slider(
            "σ (noise multiplier)", 0.0, 4.0, 1.0, step=0.25, disabled=not dp_enabled,
            help="Mais ruído = mais privacidade, menos utilidade.",
        )
        synthetic_samples = st.slider(
            "Amostras sintéticas (pool)", 16, 256, 64, step=16,
            help="Tamanho do dataset sintético repartido entre os clientes "
            "(80% treino / 20% validação por cliente).",
        )
        modality_dropout = st.slider(
            "Dropout de modalidade", 0.0, 0.5, 0.1, step=0.05,
            help="Fração de modalidades ausentes por paciente — simula coortes "
            "incompletas. A fusão lida com modalidade faltante via máscara.",
        )
        lambda_aux = st.slider(
            "λ balanceamento — cabeças unimodais", 0.0, 1.0, 0.3, step=0.1,
            help="Peso da média das perdas de Cox unimodais. Força cada modalidade "
            "(radiômica, histologia, genômica) a ser individualmente preditiva "
            "(mecanismo iii da Seção 6.1).",
        )
        lambda_balance = st.slider(
            "λ balanceamento — variância", 0.0, 1.0, 0.1, step=0.1,
            help="Peso da variância entre as perdas unimodais. Penaliza uma "
            "modalidade carregar o modelo sozinha.",
        )
        seed = int(st.number_input(
            "Seed", value=42, step=1, help="Semente de aleatoriedade (reprodutibilidade)."
        ))
        submitted = st.form_submit_button(
            "Iniciar simulação", type="primary", width="stretch", disabled=proc_alive()
        )

    if proc_alive():
        st.caption(":material/sync: Simulação em andamento…")
        if st.button("Parar simulação", width="stretch"):
            stop_run()
            st.rerun()

    manifest_path = st.session_state.get("data_manifest_path", "")
    st.caption(
        "A 1ª rodada leva ~1 min (subida do Ray + DenseNet3D em CPU). "
        + (
            f"Usando o manifesto em **{manifest_path}** (aba **Dados**)."
            if manifest_path
            else "Dados do `SyntheticPDACDataset` — ajuste em **Dados**, na aba, para usar dados reais."
        )
    )

if submitted:
    launch_run(
        overrides={
            "train": {
                "local_epochs": local_epochs, "lr": float(lr), "seed": seed,
                "lambda_aux": float(lambda_aux), "lambda_balance": float(lambda_balance),
            },
            "federated": {
                "strategy": strategy or "FedAvg",
                "fedbn": bool(fedbn),
                "dp": {
                    "enabled": bool(dp_enabled),
                    "clip_norm": 1.0,
                    "noise_multiplier": float(dp_noise),
                },
            },
            "data": {
                "synthetic_samples": synthetic_samples,
                "modality_dropout": modality_dropout,
                "manifest_csv": st.session_state.get("data_manifest_path", ""),
                "data_root": st.session_state.get("data_root", "./data/processed"),
            },
        },
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    st.rerun()


tour.maybe_start()

st.title("Pipeline Multimodal Federado — PDAC")
st.caption("Dispare e acompanhe o treino federado. Novo por aqui? Veja o **❔ Tour do painel** na barra lateral.")

runs = list_runs()
selected_run: Path | None = None
if runs:
    run_keys = [str(p) for p in runs]
    default_idx = run_keys.index(st.session_state["active_run"]) if st.session_state.get("active_run") in run_keys else 0
    selected_run = Path(
        st.selectbox(
            "Execução", run_keys, index=default_idx, format_func=lambda k: Path(k).name,
            help="Cada run vive em outputs/<run>/. Selecione para reabrir ou comparar execuções anteriores.",
        )
    )
else:
    st.info(
        "Nenhuma execução ainda. Configure os parâmetros na barra lateral e clique em "
        "**Iniciar simulação** -- ou prepare seus dados na aba **Dados** enquanto isso."
    )

tab_train, tab_attention, tab_xai, tab_data = st.tabs(
    ["Treino federado", "Atenção — histopatologia", "Explicabilidade — SHAP · Grad-CAM", "Dados"]
)

with tab_data:
    data_tab.render(selected_run)

with tab_train:
    if selected_run is None:
        st.caption("Nenhuma execução ainda.")
    elif (read_json(selected_run / "status.json") or {}).get("state") == "running":
        st.session_state.was_running = True
        st.fragment(run_every=2)(training_tab.render)(selected_run)
    else:
        training_tab.render(selected_run)

with tab_attention:
    if selected_run is None:
        st.caption("Nenhuma execução ainda.")
    else:
        attention_tab.render(selected_run)

with tab_xai:
    if selected_run is None:
        st.caption("Nenhuma execução ainda.")
    else:
        xai_tab.render(selected_run)
