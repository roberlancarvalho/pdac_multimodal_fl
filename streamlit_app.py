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
# Barra lateral — configuração e disparo                                      #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Configuração da simulação")

    with st.form("config"):
        num_clients = st.slider("Clientes (instituições)", 2, 6, 2)
        num_rounds = st.slider("Rodadas federadas", 1, 20, 3)
        local_epochs = st.slider("Épocas locais por rodada", 1, 5, 1)
        lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3],
            value=3e-4,
            format_func=lambda v: f"{v:.0e}",
        )
        strategy = st.segmented_control(
            "Estratégia de agregação",
            ["FedAvg", "FedProx", "FedAdam"],
            default="FedAvg",
        )
        synthetic_samples = st.slider("Amostras sintéticas (pool)", 16, 256, 64, step=16)
        modality_dropout = st.slider("Dropout de modalidade", 0.0, 0.5, 0.1, step=0.05)
        seed = int(st.number_input("Seed", value=42, step=1))
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
            "train": {"local_epochs": local_epochs, "lr": float(lr), "seed": seed},
            "federated": {"strategy": strategy or "FedAvg"},
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
st.title("Pipeline Multimodal Federado — PDAC")

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
    )
)

tab_train, tab_attention = st.tabs(["Treino federado", "Atenção — histopatologia"])


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

    top = st.container(horizontal=True)
    top.markdown(f"**Status:** {badge}")
    top.markdown(f"**Rodada:** {done_round}/{total_rounds}")
    top.markdown(f"**Clientes:** {cfg.get('num_clients', '?')}")
    top.markdown(f"**Estratégia:** {cfg.get('federated', {}).get('strategy', '?')}")
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
        )
        kpis.metric(
            "Perda de Cox (avaliação)",
            f"{_f(last.get('eval_loss')):.4f}",
            delta(last.get("eval_loss"), prev.get("eval_loss")),
            delta_color="inverse",
            border=True,
        )
        kpis.metric(
            "Perda de Cox (treino)",
            f"{_f(last.get('train_loss')):.4f}",
            delta(last.get("train_loss"), prev.get("train_loss")),
            delta_color="inverse",
            border=True,
        )

        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.markdown("**C-index global por rodada**")
            st.line_chart(hist, x="round", y="c_index", height=280)
        with c2, st.container(border=True):
            st.markdown("**Perda de Cox por rodada**")
            loss_cols = [c for c in ("train_loss", "eval_loss") if c in hist]
            st.line_chart(hist, x="round", y=loss_cols, height=280)

        with st.container(border=True):
            st.markdown("**Métricas por cliente — última rodada**")
            per_client = last.get("eval_clients") or []
            if per_client:
                df = pd.DataFrame(per_client)
                df = df.rename(
                    columns={
                        "cid": "cliente",
                        "num_examples": "amostras (val)",
                        "loss": "perda Cox",
                        "c_index": "C-index",
                    }
                ).sort_values("cliente")
                st.dataframe(df, hide_index=True, width="stretch")
            else:
                st.caption("Sem métricas por cliente nesta rodada.")

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
    model.load_state_dict(ckpt["state_dict"])
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
        n_patches = ctrl.slider("Patches na *bag*", 32, 512, 200, step=32)
        patient_seed = int(ctrl.number_input("Seed do paciente sintético", value=0, step=1))
        top_k = ctrl.slider("Top-k patches", 5, 50, 15)

        if st.button("Gerar visualização de atenção", type="primary"):
            with st.spinner("Rodando o Ramo B…"):
                attn = compute_attention(
                    str(model_path), model_path.stat().st_mtime, n_patches, patient_seed
                )

            df_attn = pd.DataFrame({"patch": range(len(attn)), "atencao": attn})

            m = st.container(horizontal=True)
            m.metric("Patches", len(attn), border=True)
            m.metric("Atenção máx.", f"{attn.max():.4f}", border=True)
            m.metric("Concentração (máx/média)", f"{attn.max() / attn.mean():.2f}×", border=True)

            c1, c2 = st.columns([2, 3])
            with c1, st.container(border=True):
                st.markdown(f"**Top-{top_k} patches por atenção**")
                st.dataframe(
                    df_attn.nlargest(top_k, "atencao").reset_index(drop=True),
                    hide_index=True,
                    width="stretch",
                )
            with c2, st.container(border=True):
                st.markdown("**Mapa de atenção (grade pseudo-espacial)**")
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
