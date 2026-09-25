"""Tour guiado de introdução ao painel (abre sozinho na primeira visita)."""

from __future__ import annotations

import streamlit as st

from dashboard.io import TOUR_FLAG

STEPS = [
    {
        "icon": ":material/hub:",
        "title": "O que é este painel",
        "body": (
            "Ele **dispara e acompanha** treinos do *Pipeline Multimodal Federado "
            "para PDAC*. Três ramos (TC 3D, histopatologia, genômica) são fundidos "
            "por atenção cruzada e treinados de forma **federada** com o Flower — "
            "os dados nunca saem de cada instituição.\n\n"
            "Nesta demo os dados são sintéticos (`SyntheticPDACDataset`); serve para "
            "validar o encanamento ponta a ponta."
        ),
    },
    {
        "icon": ":material/tune:",
        "title": "1 · Configurar (barra lateral)",
        "body": (
            "Na **barra lateral** você define nº de clientes, rodadas federadas, "
            "épocas locais, *learning rate*, a estratégia de agregação "
            "(**FedAvg / FedProx / FedAdam**) e o dataset sintético.\n\n"
            "Para um teste rápido use **2–3 clientes** e **2–3 rodadas** "
            "(cada rodada leva ~1 min em CPU)."
        ),
    },
    {
        "icon": ":material/play_arrow:",
        "title": "2 · Iniciar a simulação",
        "body": (
            "**Iniciar simulação** roda `federated/simulation.py` em segundo plano. "
            "A página passa a se **atualizar sozinha a cada 2 s**.\n\n"
            "A 1ª rodada demora mais (subida do Ray). Dá para **Parar** a qualquer "
            "momento."
        ),
    },
    {
        "icon": ":material/monitoring:",
        "title": "3 · Acompanhar o treino",
        "body": (
            "Na aba **Treino federado**:\n"
            "- **C-index global** — capacidade de ordenar risco/sobrevida "
            "(0,5 = acaso, 1,0 = perfeito);\n"
            "- **perda de Cox** (treino e avaliação) — deve cair;\n"
            "- gráficos por rodada e **tabela por cliente** (cada instituição)."
        ),
    },
    {
        "icon": ":material/visibility:",
        "title": "4 · Atenção da histopatologia",
        "body": (
            "A aba **Atenção — histopatologia** carrega o modelo global agregado e "
            "mostra **quais patches da lâmina** o Ramo B (attention-MIL) considerou "
            "mais informativos. Com patches sintéticos a atenção fica ~uniforme — é "
            "a demonstração do mecanismo de interpretabilidade."
        ),
    },
    {
        "icon": ":material/download:",
        "title": "5 · Exportações e execuções",
        "body": (
            "Cada execução grava em `outputs/<run>/`. No topo, o seletor "
            "**Execução** reabre/compara runs anteriores.\n\n"
            "Em *Exportações* há o PNG **C-index × rodada** e o comando do "
            "**TensorBoard** (`tensorboard --logdir outputs`) com curvas, "
            "histogramas de pesos e de atenção.\n\n"
            "Reabra este tour quando quiser em **❔ Tour do painel** na barra lateral."
        ),
    },
]


def _close() -> None:
    st.session_state.tour_open = False
    st.session_state.tour_step = 0


@st.dialog("Tour do painel", width="large", on_dismiss=_close)
def _dialog() -> None:
    step = st.session_state.get("tour_step", 0)
    total = len(STEPS)
    current = STEPS[step]

    st.progress((step + 1) / total, text=f"Passo {step + 1} de {total}")
    st.subheader(f"{current['icon']} {current['title']}")
    st.markdown(current["body"])
    st.divider()

    prev_col, skip_col, next_col = st.columns(3)
    if prev_col.button("Anterior", icon=":material/arrow_back:", width="stretch", disabled=step == 0):
        st.session_state.tour_step = max(0, step - 1)
        st.rerun()
    if skip_col.button("Pular", width="stretch"):
        _close()
        st.rerun()
    if step < total - 1:
        if next_col.button("Próximo", icon=":material/arrow_forward:", type="primary", width="stretch"):
            st.session_state.tour_step = step + 1
            st.rerun()
    elif next_col.button("Concluir", icon=":material/check:", type="primary", width="stretch"):
        _close()
        st.rerun()


def open_now() -> None:
    st.session_state.tour_open = True
    st.session_state.tour_step = 0


def maybe_start() -> None:
    """Abre o tour automaticamente na primeira vez; senão, respeita o botão da sidebar."""
    if "tour_open" not in st.session_state:
        st.session_state.tour_step = 0
        first_time = not TOUR_FLAG.exists()
        st.session_state.tour_open = first_time
        if first_time:
            try:
                TOUR_FLAG.write_text("seen\n", encoding="utf-8")
            except OSError:
                pass
    if st.session_state.get("tour_open"):
        _dialog()
