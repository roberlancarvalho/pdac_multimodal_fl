"""Aba "Treino federado" -- status, KPIs, gráficos e métricas por cliente."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.charts import round_line_chart
from dashboard.io import client_labels, read_history, read_json

_STATE_BADGE = {
    "running": ":material/sync: em andamento",
    "done": ":material/check_circle: concluída",
    "failed": ":material/error: falhou",
}


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _delta(cur, ref) -> str | None:
    cur, ref = _as_float(cur), _as_float(ref)
    if math.isnan(cur) or math.isnan(ref):
        return None
    return f"{cur - ref:+.4f}"


def _status_line(cfg: dict, status: dict) -> None:
    fed_cfg = cfg.get("federated", {})
    flags = []
    if fed_cfg.get("fedbn"):
        flags.append("FedBN")
    dp_cfg = fed_cfg.get("dp") or {}
    if dp_cfg.get("enabled"):
        flag = f"DP σ={dp_cfg.get('noise_multiplier', 1.0)}"
        if cfg.get("dp_epsilon") is not None:
            flag += f" (ε≈{cfg['dp_epsilon']:.1f})"
        flags.append(flag)
    strategy_line = fed_cfg.get("strategy", "?")
    if flags:
        strategy_line += f" · {' · '.join(flags)}"

    total_rounds = int(status.get("num_rounds", cfg.get("num_rounds", 0)) or 0)
    row = st.container(horizontal=True)
    row.markdown(f"**Status:** {_STATE_BADGE.get(status.get('state'), status.get('state', '?'))}")
    row.markdown(f"**Rodada:** {int(status.get('current_round', 0))}/{total_rounds}")
    row.markdown(f"**Clientes:** {cfg.get('num_clients', '?')}")
    row.markdown(f"**Estratégia:** {strategy_line}")
    row.markdown(f"**Tempo:** {status.get('elapsed_s', 0)}s")
    if total_rounds:
        pct = min(int(status.get("current_round", 0)) / total_rounds, 1.0)
        st.progress(pct, text=f"{pct:.0%}")


# Catálogo de cards de KPI -- usado tanto para renderizar quanto para o drawer
# de personalização (barra lateral) escolher quais mostrar e em que ordem.
KPI_CARDS: dict[str, str] = {
    "c_index": "C-index global",
    "eval_loss": "Perda de Cox (avaliação)",
    "train_loss": "Perda de Cox (treino)",
    "auc_dx": "AUC — diagnóstico",
    "acc_subtype": "Acurácia — subtipo molecular",
    "central_c_index": "C-index — validação central",
    "central_auc_dx": "AUC diag. — validação central",
}
LABEL_TO_ID: dict[str, str] = {label: cid for cid, label in KPI_CARDS.items()}
VISIBLE_HEADER = "Visíveis"
HIDDEN_HEADER = "Ocultos"


def default_kpi_layout() -> dict[str, list[str]]:
    return {VISIBLE_HEADER: list(KPI_CARDS.values()), HIDDEN_HEADER: []}

_CARD_HELP: dict[str, str] = {
    "c_index": "Concordância de Harrell agregada entre os clientes: probabilidade de o "
    "paciente com maior risco predito ter o evento antes. 0,5 = acaso · 1,0 = ordenação perfeita.",
    "eval_loss": "Negative partial log-likelihood de Cox no conjunto de validação de cada "
    "cliente, agregada. Menor é melhor.",
    "train_loss": "Mesma perda, medida durante o treino local antes da agregação.",
    "auc_dx": "Área sob a curva ROC da cabeça de diagnóstico (PDAC vs não-PDAC), agregada "
    "entre os clientes (Figura 4).",
    "acc_subtype": "Acurácia da cabeça de subtipagem (classical vs basal-like) sobre os "
    "pacientes com subtipo conhecido.",
    "central_c_index": "Modelo global avaliado no servidor sobre uma coorte held-out "
    "(validação centralizada / independente).",
    "central_auc_dx": "Mesma AUC de diagnóstico, medida na coorte held-out do servidor.",
}
_INVERSE_CARDS = {"eval_loss", "train_loss"}


def _sparkline(hist: pd.DataFrame, col: str) -> list[float] | None:
    if col not in hist:
        return None
    return hist[col].dropna().tolist() or None


def _build_card(card_id: str, last: pd.Series, prev: pd.Series, hist: pd.DataFrame) -> dict | None:
    if card_id not in hist:
        return None
    kwargs = dict(
        label=KPI_CARDS[card_id],
        value=f"{_as_float(last.get(card_id)):.4f}",
        delta=_delta(last.get(card_id), prev.get(card_id)),
        chart_data=_sparkline(hist, card_id),
        help=_CARD_HELP.get(card_id, ""),
    )
    if card_id in _INVERSE_CARDS:
        kwargs["delta_color"] = "inverse"
    return kwargs


def _metric_row(items: list[dict]) -> None:
    """Uma linha com exatamente `len(items)` colunas -- nunca sobra card órfão."""
    for col, item in zip(st.columns(len(items)), items, strict=True):
        col.metric(**item, border=True, chart_type="line")


def _kpi_row(last: pd.Series, prev: pd.Series, hist: pd.DataFrame) -> None:
    st.caption(
        "Δ compara com a rodada anterior. Arraste os cards em **🎛️ Personalizar cards**, "
        "na barra lateral, para reordenar ou esconder."
    )
    layout = st.session_state.get("kpi_layout") or default_kpi_layout()
    order = [LABEL_TO_ID[label] for label in layout.get(VISIBLE_HEADER, []) if label in LABEL_TO_ID]
    cards = [card for cid in order if (card := _build_card(cid, last, prev, hist)) is not None]
    if not cards:
        st.info("Nenhum card selecionado -- escolha em 🎛️ Personalizar cards, na barra lateral.")
        return
    for i in range(0, len(cards), 3):
        _metric_row(cards[i : i + 3])


def _round_charts(hist: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("C-index por rodada", divider=False)
        st.caption(
            "1,0 = ordenação perfeita do risco · 0,5 (linha pontilhada) = acaso. "
            "**Distribuído** = média dos clientes · **central** = modelo global na coorte do servidor. "
            "Arraste/role para dar zoom."
        )
        cols = [c for c in ("c_index", "central_c_index") if c in hist]
        labels = {"c_index": "distribuído", "central_c_index": "central"}
        st.altair_chart(
            round_line_chart(hist, cols, labels, "C-index", baseline=0.5, baseline_label="acaso"),
            width="stretch",
        )
    with c2, st.container(border=True):
        st.subheader("Perda de Cox por rodada", divider=False)
        st.caption(
            "Menor é melhor. **Treino** vs. **avaliação** — se avaliação sobe enquanto "
            "treino cai, é sinal de overfitting local."
        )
        cols = [c for c in ("train_loss", "eval_loss") if c in hist]
        labels = {"train_loss": "treino", "eval_loss": "avaliação"}
        st.altair_chart(round_line_chart(hist, cols, labels, "perda de Cox"), width="stretch")


def _client_table(hist: pd.DataFrame, last: pd.Series) -> None:
    with st.container(border=True):
        st.subheader("Métricas por cliente (instituição) — última rodada", divider=False)
        per_client = last.get("eval_clients") or []
        if not per_client:
            st.caption("Sem métricas por cliente nesta rodada.")
            return
        st.caption(
            "Como cada instituição se saiu isoladamente. Barras muito desiguais no C-index "
            "indicam dados não-IID entre clientes — considere a estratégia FedProx."
        )

        labels = client_labels(hist)
        fit_loss = {c["cid"]: c.get("train_loss") for c in (last.get("fit_clients") or [])}
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
                "C-index": st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0),
            },
        )


def _modality_gate_chart(hist: pd.DataFrame) -> None:
    if "modality_gate" not in hist:
        return
    gate_hist = hist[hist["modality_gate"].notna()]
    if gate_hist.empty:
        return
    with st.container(border=True):
        st.subheader("Contribuição por modalidade (co-atenção)", divider=False)
        st.caption(
            "Peso médio do token [FUSION] sobre cada modalidade. Barras equilibradas = a "
            "regularização de balanceamento está evitando que uma modalidade domine sozinha."
        )
        rows = [
            {"rodada": r["round"], "modalidade": mod, "peso": val}
            for _, r in gate_hist.iterrows()
            for mod, val in (r["modality_gate"] or {}).items()
        ]
        st.bar_chart(pd.DataFrame(rows), x="rodada", y="peso", color="modalidade", height=240)


def _exports(run_dir: Path) -> None:
    png_path = run_dir / "c_index.png"
    tb_dir = run_dir / "tb"
    if not png_path.exists() and not tb_dir.exists():
        return
    with st.expander("Exportações (PNG · TensorBoard)", icon=":material/download:"):
        if png_path.exists():
            st.image(str(png_path), caption="Gerado ao final da execução.")
        if tb_dir.exists():
            st.markdown(
                "**TensorBoard** — curvas por rodada, histogramas de pesos "
                "(`weights/<ramo>`) e de atenção do Ramo B:"
            )
            st.code("tensorboard --logdir outputs", language="bash")


def _log_expander(run_dir: Path) -> None:
    with st.expander("Log da execução"):
        log_path = run_dir / "run.log"
        if not log_path.exists():
            st.caption("Sem log.")
            return
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        st.code("\n".join(lines[-200:]) or "(vazio)", language="text")


def render(run_dir: Path) -> None:
    status = read_json(run_dir / "status.json") or {}
    cfg = read_json(run_dir / "config.json") or {}
    hist = read_history(run_dir)

    _status_line(cfg, status)

    if status.get("state") == "failed" and status.get("error"):
        st.error(f"Erro: {status['error']}")

    if hist.empty:
        st.info("Aguardando a primeira rodada concluir…")
    else:
        last, prev = hist.iloc[-1], hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
        _kpi_row(last, prev, hist)
        _round_charts(hist)
        _client_table(hist, last)
        _modality_gate_chart(hist)

    _exports(run_dir)
    _log_expander(run_dir)

    # Promove a um rerun completo quando a simulação termina, para parar o
    # auto-refresh do fragmento que chama esta função.
    if status.get("state") in ("done", "failed") and st.session_state.get("was_running"):
        st.session_state.was_running = False
        done = status.get("state") == "done"
        st.toast(
            "Simulação concluída" if done else "Simulação falhou",
            icon=":material/check_circle:" if done else ":material/error:",
        )
        st.rerun()
