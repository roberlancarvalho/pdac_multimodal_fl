"""
Painel do Pipeline Multimodal Federado para PDAC.

Roda a simulação federada (`federated/simulation.py`) como subprocesso e
acompanha "em tela":
  - status da execução e progresso por rodada;
  - C-index global e perdas (Cox) agregadas ao longo das rodadas;
  - métricas por cliente (instituição) na última rodada;
  - visualização dos pesos de atenção do Ramo B (histopatologia).

Executar:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
OUTPUTS = PROJECT_ROOT / "outputs"
TOUR_FLAG = PROJECT_ROOT / ".tour_seen"  # marca que este ambiente já viu o tour

st.set_page_config(
    page_title="PDAC FL — painel",
    page_icon=":material/hub:",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Helpers de leitura das execuções                                            #
# --------------------------------------------------------------------------- #
def list_runs() -> list[Path]:
    if not OUTPUTS.exists():
        return []
    runs = [p for p in OUTPUTS.iterdir() if p.is_dir() and (p / "config.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "history.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return pd.DataFrame(rows)


def client_labels(hist: pd.DataFrame) -> dict[str, str]:
    """Mapeia os `cid` (ids longos do Ray) para rótulos estáveis 'Cliente N'."""
    cids: set[str] = set()
    for col in ("eval_clients", "fit_clients"):
        if col not in hist:
            continue
        for entry in hist[col].dropna():
            for client in entry or []:
                cids.add(client["cid"])
    return {cid: f"Cliente {i + 1}" for i, cid in enumerate(sorted(cids))}


def proc_alive() -> bool:
    proc = st.session_state.get("proc")
    return proc is not None and proc.poll() is None


def launch_run(overrides: dict, num_clients: int, num_rounds: int) -> None:
    import yaml

    from federated.config import load_config

    OUTPUTS.mkdir(exist_ok=True)
    run_dir = OUTPUTS / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir()

    cfg = load_config()
    for section, values in overrides.items():
        cfg[section].update(values)
    cfg_path = run_dir / "effective_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    log_handle = (run_dir / "run.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "federated.simulation",
            "--config",
            str(cfg_path),
            "--num-clients",
            str(num_clients),
            "--num-rounds",
            str(num_rounds),
            "--run-dir",
            str(run_dir),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    st.session_state.proc = proc
    st.session_state.active_run = str(run_dir)
    st.session_state.was_running = True


def stop_run() -> None:
    proc = st.session_state.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
    run_dir = st.session_state.get("active_run")
    if run_dir:
        status_path = Path(run_dir) / "status.json"
        status = read_json(status_path) or {}
        status.update({"state": "failed", "error": "cancelado pelo usuário", "updated": time.time()})
        status_path.write_text(json.dumps(status), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tour de introdução                                                          #
# --------------------------------------------------------------------------- #
TOUR_STEPS = [
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


def _close_tour() -> None:
    st.session_state.tour_open = False
    st.session_state.tour_step = 0


@st.dialog("Tour do painel", width="large", on_dismiss=_close_tour)
def _tour_dialog() -> None:
    step = st.session_state.get("tour_step", 0)
    total = len(TOUR_STEPS)
    s = TOUR_STEPS[step]

    st.progress((step + 1) / total, text=f"Passo {step + 1} de {total}")
    st.subheader(f"{s['icon']} {s['title']}")
    st.markdown(s["body"])
    st.divider()

    prev_col, skip_col, next_col = st.columns(3)
    if prev_col.button(
        "Anterior", icon=":material/arrow_back:", width="stretch", disabled=step == 0
    ):
        st.session_state.tour_step = max(0, step - 1)
        st.rerun()
    if skip_col.button("Pular", width="stretch"):
        _close_tour()
        st.rerun()
    if step < total - 1:
        if next_col.button(
            "Próximo", icon=":material/arrow_forward:", type="primary", width="stretch"
        ):
            st.session_state.tour_step = step + 1
            st.rerun()
    elif next_col.button("Concluir", icon=":material/check:", type="primary", width="stretch"):
        _close_tour()
        st.rerun()


def maybe_start_tour() -> None:
    """Abre o tour automaticamente na primeira vez; senão, respeita o botão."""
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
        _tour_dialog()


# --------------------------------------------------------------------------- #
# Barra lateral — configuração e disparo                                      #
# --------------------------------------------------------------------------- #
with st.sidebar:
    if st.button(
        "Tour do painel",
        icon=":material/help:",
        width="stretch",
        help="Reabre a introdução passo a passo do painel.",
    ):
        st.session_state.tour_open = True
        st.session_state.tour_step = 0
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
            "Learning rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3],
            value=3e-4,
            format_func=lambda v: f"{v:.0e}",
            help="Taxa de aprendizado do otimizador AdamW nos clientes.",
        )
        strategy = st.segmented_control(
            "Estratégia de agregação",
            ["FedAvg", "FedProx", "FedAdam"],
            default="FedAvg",
            help="Como o servidor combina os pesos dos clientes. FedProx penaliza "
            "desvio do modelo global (bom p/ dados não-IID); FedAdam usa momento no "
            "servidor.",
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

    st.caption(
        "A 1ª rodada leva ~1 min (subida do Ray + DenseNet3D em CPU). "
        "Dados do `SyntheticPDACDataset` — serve para validar o pipeline."
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
                "manifest_csv": "",
            },
        },
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    st.rerun()


# --------------------------------------------------------------------------- #
# Área principal                                                              #
# --------------------------------------------------------------------------- #
maybe_start_tour()

st.title("Pipeline Multimodal Federado — PDAC")
st.caption(
    "Dispare e acompanhe o treino federado. Novo por aqui? Veja o "
    "**❔ Tour do painel** na barra lateral."
)

runs = list_runs()
if not runs:
    st.info(
        "Nenhuma execução ainda. Configure os parâmetros na barra lateral e "
        "clique em **Iniciar simulação**."
    )
    st.stop()

run_keys = [str(p) for p in runs]
default_idx = 0
if st.session_state.get("active_run") in run_keys:
    default_idx = run_keys.index(st.session_state["active_run"])

selected_run = Path(
    st.selectbox(
        "Execução",
        run_keys,
        index=default_idx,
        format_func=lambda k: Path(k).name,
        help="Cada run vive em outputs/<run>/. Selecione para reabrir ou comparar "
        "execuções anteriores.",
    )
)

tab_train, tab_attention, tab_xai = st.tabs(
    ["Treino federado", "Atenção — histopatologia", "Explicabilidade — SHAP · Grad-CAM"]
)


# --------------------------------------------------------------------------- #
# Aba: treino federado                                                        #
# --------------------------------------------------------------------------- #
def render_training(run_dir: Path) -> None:
    status = read_json(run_dir / "status.json") or {}
    cfg = read_json(run_dir / "config.json") or {}
    state = status.get("state", "desconhecido")
    hist = read_history(run_dir)

    done_round = int(status.get("current_round", 0))
    total_rounds = int(status.get("num_rounds", cfg.get("num_rounds", 0)) or 0)

    badge = {
        "running": ":material/sync: em andamento",
        "done": ":material/check_circle: concluída",
        "failed": ":material/error: falhou",
    }.get(state, state)

    fed_cfg = cfg.get("federated", {})
    flags = []
    if fed_cfg.get("fedbn"):
        flags.append("FedBN")
    if (fed_cfg.get("dp") or {}).get("enabled"):
        dp_flag = f"DP σ={fed_cfg['dp'].get('noise_multiplier', 1.0)}"
        if cfg.get("dp_epsilon") is not None:
            dp_flag += f" (ε≈{cfg['dp_epsilon']:.1f})"
        flags.append(dp_flag)

    top = st.container(horizontal=True)
    top.markdown(f"**Status:** {badge}")
    top.markdown(f"**Rodada:** {done_round}/{total_rounds}")
    top.markdown(f"**Clientes:** {cfg.get('num_clients', '?')}")
    top.markdown(f"**Estratégia:** {fed_cfg.get('strategy', '?')}"
                 + (f" · {' · '.join(flags)}" if flags else ""))
    top.markdown(f"**Tempo:** {status.get('elapsed_s', 0)}s")

    if total_rounds:
        st.progress(min(done_round / total_rounds, 1.0))

    if state == "failed" and status.get("error"):
        st.error(f"Erro: {status['error']}")

    if hist.empty:
        st.info("Aguardando a primeira rodada concluir…")
    else:
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last

        def _f(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        def delta(cur, ref):
            cur, ref = _f(cur), _f(ref)
            if math.isnan(cur) or math.isnan(ref):
                return None
            return f"{cur - ref:+.4f}"

        spark = hist["c_index"].dropna().tolist() if "c_index" in hist else None

        kpis = st.container(horizontal=True)
        kpis.metric(
            "C-index global",
            f"{_f(last.get('c_index')):.4f}",
            delta(last.get("c_index"), prev.get("c_index")),
            border=True,
            chart_data=spark or None,
            chart_type="line",
            help="Concordância de Harrell agregada entre os clientes: probabilidade "
            "de o paciente com maior risco predito ter o evento antes. "
            "0,5 = acaso · 1,0 = ordenação perfeita.",
        )
        kpis.metric(
            "Perda de Cox (avaliação)",
            f"{_f(last.get('eval_loss')):.4f}",
            delta(last.get("eval_loss"), prev.get("eval_loss")),
            delta_color="inverse",
            border=True,
            help="Negative partial log-likelihood de Cox no conjunto de validação "
            "de cada cliente, agregada. Menor é melhor.",
        )
        kpis.metric(
            "Perda de Cox (treino)",
            f"{_f(last.get('train_loss')):.4f}",
            delta(last.get("train_loss"), prev.get("train_loss")),
            delta_color="inverse",
            border=True,
            help="Mesma perda, medida durante o treino local antes da agregação.",
        )

        if "auc_dx" in hist:
            task_kpis = st.container(horizontal=True)
            task_kpis.metric(
                "AUC — diagnóstico",
                f"{_f(last.get('auc_dx')):.4f}",
                delta(last.get("auc_dx"), prev.get("auc_dx")),
                border=True,
                help="Área sob a curva ROC da cabeça de diagnóstico (PDAC vs não-PDAC), "
                "agregada entre os clientes (Figura 4).",
            )
            task_kpis.metric(
                "Acurácia — subtipo molecular",
                f"{_f(last.get('acc_subtype')):.4f}",
                delta(last.get("acc_subtype"), prev.get("acc_subtype")),
                border=True,
                help="Acurácia da cabeça de subtipagem (classical vs basal-like) "
                "sobre os pacientes com subtipo conhecido.",
            )

        if "central_c_index" in hist:
            central_kpis = st.container(horizontal=True)
            central_kpis.metric(
                "C-index — validação central",
                f"{_f(last.get('central_c_index')):.4f}",
                delta(last.get("central_c_index"), prev.get("central_c_index")),
                border=True,
                help="Modelo global avaliado no servidor sobre uma coorte held-out "
                "(validação centralizada / independente).",
            )
            if "central_auc_dx" in hist:
                central_kpis.metric(
                    "AUC diag. — validação central",
                    f"{_f(last.get('central_auc_dx')):.4f}",
                    delta(last.get("central_auc_dx"), prev.get("central_auc_dx")),
                    border=True,
                )

        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.subheader(
                "C-index por rodada",
                help="Distribuído = média ponderada dos clientes. Central = modelo "
                "global na coorte do servidor. A linha pontilhada em 0,5 marca o acaso.",
                divider=False,
            )
            cidx_cols = [c for c in ("c_index", "central_c_index") if c in hist]
            st.line_chart(hist, x="round", y=cidx_cols, height=280)
        with c2, st.container(border=True):
            st.subheader(
                "Perda de Cox por rodada",
                help="Treino vs. avaliação. Divergência entre as duas curvas "
                "sugere overfitting local.",
                divider=False,
            )
            loss_cols = [c for c in ("train_loss", "eval_loss") if c in hist]
            st.line_chart(hist, x="round", y=loss_cols, height=280)

        with st.container(border=True):
            st.subheader(
                "Métricas por cliente (instituição) — última rodada",
                help="Como cada nó federado se saiu. Heterogeneidade grande entre "
                "clientes indica dados não-IID — considere FedProx.",
                divider=False,
            )
            per_client = last.get("eval_clients") or []
            if per_client:
                labels = client_labels(hist)
                fit_loss = {
                    c["cid"]: c.get("train_loss") for c in (last.get("fit_clients") or [])
                }
                df = pd.DataFrame(
                    [
                        {
                            "cliente": labels.get(c["cid"], c["cid"]),
                            "amostras (val)": c.get("num_examples"),
                            "perda Cox (treino)": fit_loss.get(c["cid"]),
                            "perda Cox (val)": c.get("loss"),
                            "C-index": c.get("c_index"),
                        }
                        for c in per_client
                    ]
                ).sort_values("cliente")
                st.dataframe(
                    df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "perda Cox (treino)": st.column_config.NumberColumn(format="%.4f"),
                        "perda Cox (val)": st.column_config.NumberColumn(format="%.4f"),
                        "C-index": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
            else:
                st.caption("Sem métricas por cliente nesta rodada.")

        gate_hist = hist[hist["modality_gate"].notna()] if "modality_gate" in hist else hist.iloc[0:0]
        if not gate_hist.empty:
            with st.container(border=True):
                st.subheader(
                    "Contribuição por modalidade (co-atenção)",
                    help="Peso médio do token [FUSION] sobre cada modalidade na "
                    "leitura final. Barras equilibradas indicam que a regularização "
                    "de balanceamento está evitando dominância de uma modalidade.",
                    divider=False,
                )
                rows = []
                for _, r in gate_hist.iterrows():
                    for mod, val in (r["modality_gate"] or {}).items():
                        rows.append({"rodada": r["round"], "modalidade": mod, "peso": val})
                gdf = pd.DataFrame(rows)
                st.bar_chart(gdf, x="rodada", y="peso", color="modalidade", height=240)

    png_path = run_dir / "c_index.png"
    tb_dir = run_dir / "tb"
    if png_path.exists() or tb_dir.exists():
        with st.expander(
            "Exportações (PNG · TensorBoard)",
            icon=":material/download:",
        ):
            if png_path.exists():
                st.image(str(png_path), caption="Gerado ao final da execução.")
            if tb_dir.exists():
                st.markdown(
                    "**TensorBoard** — curvas por rodada, histogramas de pesos "
                    "(`weights/<ramo>`) e de atenção do Ramo B:"
                )
                st.code("tensorboard --logdir outputs", language="bash")

    with st.expander("Log da execução"):
        log_path = run_dir / "run.log"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(lines[-200:]) or "(vazio)", language="text")
        else:
            st.caption("Sem log.")

    # Promove a um rerun completo quando a simulação termina, para parar o
    # auto-refresh do fragmento.
    if state in ("done", "failed") and st.session_state.get("was_running"):
        st.session_state.was_running = False
        st.toast(
            "Simulação concluída" if state == "done" else "Simulação falhou",
            icon=":material/check_circle:" if state == "done" else ":material/error:",
        )
        st.rerun()


with tab_train:
    _status = read_json(selected_run / "status.json") or {}
    _is_running = _status.get("state") == "running"
    if _is_running:
        st.session_state.was_running = True
        st.fragment(run_every=2)(render_training)(selected_run)
    else:
        render_training(selected_run)


# --------------------------------------------------------------------------- #
# Aba: atenção da histopatologia                                              #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def compute_attention(model_path: str, mtime: float, n_patches: int, seed: int):
    import torch

    from models.multimodal_pdac import MultimodalPDACModel

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MultimodalPDACModel(**ckpt["model_cfg"])
    # strict=False: execuções antigas (antes da co-atenção) têm state_dict de arquitetura diferente.
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    gen = torch.Generator().manual_seed(seed)
    feat_dim = ckpt["model_cfg"]["patch_feat_dim"]
    bag = torch.randn(1, n_patches, feat_dim, generator=gen)
    mask = torch.ones(1, n_patches, dtype=torch.bool)
    with torch.no_grad():
        _, attn = model.branch_b(bag, mask)
    return attn.squeeze(0).numpy()


with tab_attention:
    model_path = selected_run / "global_model.pt"
    st.caption(
        "O Ramo B agrega embeddings de patches de WSI por *attention-MIL*. "
        "Aqui os patches são aleatórios (sintéticos), então a atenção fica ~uniforme "
        "— a aba demonstra o encanamento de interpretabilidade."
    )

    if not model_path.exists():
        st.info("O modelo global desta execução ainda não foi salvo (aguarde a 1ª rodada).")
    else:
        ctrl = st.container(horizontal=True)
        n_patches = ctrl.slider(
            "Patches na *bag*", 32, 512, 200, step=32,
            help="Quantos patches de WSI compõem a lâmina do paciente sintético.",
        )
        patient_seed = int(ctrl.number_input(
            "Seed do paciente sintético", value=0, step=1,
            help="Muda o paciente sintético gerado (embeddings de patches aleatórios).",
        ))
        top_k = ctrl.slider(
            "Top-k patches", 5, 50, 15,
            help="Quantos patches de maior atenção listar na tabela.",
        )

        if st.button(
            "Gerar visualização de atenção",
            type="primary",
            help="Roda o Ramo B (attention-MIL) do modelo global sobre a lâmina "
            "sintética e mostra o peso de atenção de cada patch.",
        ):
            with st.spinner("Rodando o Ramo B…"):
                attn = compute_attention(
                    str(model_path), model_path.stat().st_mtime, n_patches, patient_seed
                )

            df_attn = pd.DataFrame({"patch": range(len(attn)), "atencao": attn})

            m = st.container(horizontal=True)
            m.metric(
                "Patches", len(attn), border=True,
                help="Tamanho da bag processada.",
            )
            m.metric(
                "Atenção máx.", f"{attn.max():.4f}", border=True,
                help="Maior peso atribuído a um único patch (os pesos somam 1).",
            )
            m.metric(
                "Concentração (máx/média)", f"{attn.max() / attn.mean():.2f}×", border=True,
                help="1× = atenção uniforme; quanto maior, mais o modelo se apoia "
                "em poucos patches.",
            )

            c1, c2 = st.columns([2, 3])
            with c1, st.container(border=True):
                st.subheader(
                    f"Top-{top_k} patches por atenção",
                    help="Os patches que mais pesaram no embedding da lâmina.",
                    divider=False,
                )
                st.dataframe(
                    df_attn.nlargest(top_k, "atencao").reset_index(drop=True),
                    hide_index=True,
                    width="stretch",
                )
            with c2, st.container(border=True):
                st.subheader(
                    "Mapa de atenção (grade pseudo-espacial)",
                    help="Os patches são dispostos numa grade só para visualização — "
                    "não corresponde à posição real na lâmina.",
                    divider=False,
                )
                side = math.ceil(math.sqrt(len(attn)))
                padded = list(attn) + [None] * (side * side - len(attn))
                grid = pd.DataFrame(
                    {
                        "linha": [i // side for i in range(side * side)],
                        "coluna": [i % side for i in range(side * side)],
                        "patch": list(range(side * side)),
                        "atencao": padded,
                    }
                )
                heat = (
                    alt.Chart(grid.dropna())
                    .mark_rect()
                    .encode(
                        x=alt.X("coluna:O", axis=None),
                        y=alt.Y("linha:O", axis=None),
                        color=alt.Color(
                            "atencao:Q",
                            scale=alt.Scale(scheme="magma"),
                            title="atenção",
                        ),
                        tooltip=["patch", alt.Tooltip("atencao:Q", format=".5f")],
                    )
                    .properties(height=340)
                )
                st.altair_chart(heat, width="stretch")


# --------------------------------------------------------------------------- #
# Aba: explicabilidade (SHAP + Grad-CAM)                                       #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_model_cfg(model_path: str, mtime: float) -> dict:
    import torch

    return torch.load(model_path, map_location="cpu", weights_only=False)["model_cfg"]


@st.cache_data(show_spinner=False)
def compute_gradcam(model_path: str, mtime: float, seed: int, dz: int, dy: int, dx: int):
    import torch

    from models.multimodal_pdac import MultimodalPDACModel
    from utils.xai import radiomics_gradcam

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MultimodalPDACModel(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    gen = torch.Generator().manual_seed(seed)
    ct = torch.randn(1, ckpt["model_cfg"].get("ct_in_channels", 1), dz, dy, dx, generator=gen)
    batch = {"ct_volume": ct, "mutation_status": torch.tensor([[1, 1, 0, 1]])}
    cam = radiomics_gradcam(model, batch)[0].numpy()
    return ct[0, 0].numpy(), cam


def _load_xai_model(model_path: str):
    import torch

    from models.multimodal_pdac import MultimodalPDACModel

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MultimodalPDACModel(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, ckpt["model_cfg"]


@st.cache_data(show_spinner=False)
def compute_genomic_shap(model_path: str, mtime: float, mutations: tuple[int, ...], nsamples: int):
    import torch

    from utils.xai import genomics_shap

    model, _ = _load_xai_model(model_path)
    return genomics_shap(model, torch.tensor([list(mutations)]), nsamples=nsamples)


@st.cache_data(show_spinner=False)
def compute_clinical_shap(
    model_path: str, mtime: float, num: tuple[float, ...], cat: tuple[int, ...], nsamples: int
):
    import torch

    from utils.xai import clinical_shap

    model, _ = _load_xai_model(model_path)
    return clinical_shap(
        model, torch.tensor([list(num)]), torch.tensor([list(cat)]), nsamples=nsamples
    )


def _signed_shap_chart(df: "pd.DataFrame"):
    df = df.copy()
    df["sentido"] = df["shap"].apply(lambda v: "↑ risco" if v > 0 else "↓ risco")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("shap:Q", title="valor SHAP (log-hazard)"),
            y=alt.Y("campo:N", sort="-x", title=None),
            color=alt.Color(
                "sentido:N",
                scale=alt.Scale(domain=["↑ risco", "↓ risco"], range=["#C24B4B", "#3B7DD8"]),
                title=None,
            ),
            tooltip=["campo", alt.Tooltip("shap:Q", format="+.4f")],
        )
        .properties(height=240)
    )


with tab_xai:
    model_path = selected_run / "global_model.pt"
    st.caption(
        "Grad-CAM 3D no Ramo A e SHAP por gene no Ramo C — cf. Seção 6 do artigo "
        "(*explicabilidade via SHAP e Grad-CAM*). Com dados sintéticos os mapas não "
        "têm significado clínico; validam o mecanismo."
    )

    if not model_path.exists():
        st.info("O modelo global desta execução ainda não foi salvo.")
    else:
        cfg_m = _load_model_cfg(str(model_path), model_path.stat().st_mtime)
        coattn_ok = cfg_m.get("fusion_mode", "coattention") == "coattention"

        gc_tab, shap_tab, clin_tab = st.tabs(
            ["Grad-CAM 3D (Ramo A)", "SHAP genômico (Ramo C)", "SHAP clínico (Ramo D)"]
        )

        with gc_tab:
            ctrl = st.container(horizontal=True)
            gseed = int(ctrl.number_input("Seed do volume sintético", value=0, step=1, key="gc_seed"))
            dim = ctrl.slider("Lado do volume (D=H=W)", 32, 96, 48, step=16, key="gc_dim")
            if st.button("Gerar Grad-CAM", type="primary", key="gc_btn"):
                with st.spinner("Retropropagando o risco pelo DenseNet3D…"):
                    vol, cam = compute_gradcam(
                        str(model_path), model_path.stat().st_mtime, gseed, dim, dim, dim
                    )
                st.session_state["gc_result"] = (vol, cam)

            if "gc_result" in st.session_state:
                import numpy as np

                vol, cam = st.session_state["gc_result"]
                z = st.slider("Fatia axial", 0, vol.shape[0] - 1, vol.shape[0] // 2, key="gc_z")
                m = st.container(horizontal=True)
                m.metric("Volume", "×".join(map(str, vol.shape)), border=True)
                m.metric(
                    "Fração saliente (CAM > 0,5)", f"{float((cam > 0.5).mean()):.1%}", border=True,
                    help="Proporção de voxels que mais influenciaram o risco predito.",
                )
                v = vol[z]
                v = (v - v.min()) / (float(np.ptp(v)) + 1e-8)
                rgba = np.zeros((*v.shape, 4))
                rgba[..., 0] = 1.0
                rgba[..., 3] = np.clip(cam[z], 0, 1) * 0.6
                c1, c2 = st.columns(2)
                with c1, st.container(border=True):
                    st.markdown("**TC sintética (fatia)**")
                    st.image(v, clamp=True, width="stretch")
                with c2, st.container(border=True):
                    st.markdown("**Grad-CAM sobreposto**")
                    st.image(np.stack([v] * 3, axis=-1), clamp=True, width="stretch")
                    st.image(rgba, clamp=True, width="stretch")

        with shap_tab:
            if not coattn_ok:
                st.warning("SHAP genômico requer `fusion_mode: coattention` (esta execução usa o modo legado).")
            else:
                genes = ["KRAS", "TP53", "SMAD4", "CDKN2A"]
                st.markdown("**Status mutacional do paciente sintético**")
                cols = st.columns(4)
                muts = []
                for g, col in zip(genes, cols, strict=True):
                    choice = col.segmented_control(
                        g, ["wt", "mut"], default="mut" if g in ("KRAS", "TP53") else "wt",
                        key=f"shap_{g}",
                    )
                    muts.append(1 if choice == "mut" else 0)
                nsamples = st.slider("Amostras do KernelExplainer", 32, 512, 200, step=32, key="shap_ns")

                if st.button("Calcular SHAP", type="primary", key="shap_btn"):
                    with st.spinner("Estimando valores SHAP…"):
                        res = compute_genomic_shap(
                            str(model_path), model_path.stat().st_mtime, tuple(muts), nsamples
                        )
                    m = st.container(horizontal=True)
                    m.metric("Risco base (tudo wt)", f"{res['base_value']:+.3f}", border=True)
                    m.metric("Risco predito", f"{res['prediction']:+.3f}", border=True)
                    sdf = pd.DataFrame({"gene": res["genes"], "shap": res["shap"]})
                    sdf["sentido"] = sdf["shap"].apply(lambda v: "↑ risco" if v > 0 else "↓ risco")
                    chart = (
                        alt.Chart(sdf)
                        .mark_bar()
                        .encode(
                            x=alt.X("shap:Q", title="valor SHAP (log-hazard)"),
                            y=alt.Y("gene:N", sort="-x", title=None),
                            color=alt.Color(
                                "sentido:N",
                                scale=alt.Scale(
                                    domain=["↑ risco", "↓ risco"], range=["#C24B4B", "#3B7DD8"]
                                ),
                                title=None,
                            ),
                            tooltip=["gene", alt.Tooltip("shap:Q", format="+.4f")],
                        )
                        .properties(height=220)
                    )
                    with st.container(border=True):
                        st.subheader(
                            "Contribuição de cada gene driver",
                            help="Quanto a mutação de cada gene desloca o log-hazard "
                            "predito em relação ao paciente todo wild-type. Soma dos "
                            "SHAP + risco base = risco predito.",
                            divider=False,
                        )
                        st.altair_chart(chart, width="stretch")

        with clin_tab:
            if not coattn_ok or not cfg_m.get("enable_clinical", True):
                st.warning("SHAP clínico requer `fusion_mode: coattention` e `enable_clinical: true`.")
            else:
                n_cont = int(cfg_m.get("clinical_n_continuous", 5))
                cards = list(cfg_m.get("clinical_cat_cardinalities", [2, 4, 5, 3]))
                st.markdown("**Paciente sintético — variáveis clínicas** (contínuas em z-score)")
                cc = st.columns(min(n_cont, 5))
                num = [
                    cc[i % len(cc)].number_input(f"cont_{i}", value=0.0, step=0.5, key=f"cn_{i}")
                    for i in range(n_cont)
                ]
                ck = st.columns(min(len(cards), 5))
                cat = [
                    ck[i % len(ck)].number_input(
                        f"cat_{i} (0..{c - 1})", min_value=0, max_value=c - 1, value=0, key=f"cc_{i}"
                    )
                    for i, c in enumerate(cards)
                ]
                nsc = st.slider("Amostras do KernelExplainer", 32, 512, 200, step=32, key="cshap_ns")
                if st.button("Calcular SHAP clínico", type="primary", key="cshap_btn"):
                    with st.spinner("Estimando valores SHAP…"):
                        res = compute_clinical_shap(
                            str(model_path), model_path.stat().st_mtime,
                            tuple(float(x) for x in num), tuple(int(x) for x in cat), nsc,
                        )
                    mm = st.container(horizontal=True)
                    mm.metric("Risco base", f"{res['base_value']:+.3f}", border=True)
                    mm.metric("Risco predito", f"{res['prediction']:+.3f}", border=True)
                    cdf = pd.DataFrame({"campo": res["fields"], "shap": res["shap"]})
                    with st.container(border=True):
                        st.subheader(
                            "Contribuição de cada variável clínica",
                            help="Deslocamento do log-hazard vs. o paciente de referência "
                            "(contínuas na média, categóricas na categoria 0).",
                            divider=False,
                        )
                        st.altair_chart(_signed_shap_chart(cdf), width="stretch")
