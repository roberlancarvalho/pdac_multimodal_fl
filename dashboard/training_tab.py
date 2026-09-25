"""Aba "Treino federado" -- status, KPIs, gráficos e métricas por cliente."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

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
        st.progress(min(int(status.get("current_round", 0)) / total_rounds, 1.0))


def _kpi_row(last: pd.Series, prev: pd.Series, hist: pd.DataFrame) -> None:
    spark = hist["c_index"].dropna().tolist() if "c_index" in hist else None
    kpis = st.container(horizontal=True)
    kpis.metric(
        "C-index global",
        f"{_as_float(last.get('c_index')):.4f}",
        _delta(last.get("c_index"), prev.get("c_index")),
        border=True,
        chart_data=spark or None,
        chart_type="line",
        help="Concordância de Harrell agregada entre os clientes: probabilidade "
        "de o paciente com maior risco predito ter o evento antes. "
        "0,5 = acaso · 1,0 = ordenação perfeita.",
    )
    kpis.metric(
        "Perda de Cox (avaliação)",
        f"{_as_float(last.get('eval_loss')):.4f}",
        _delta(last.get("eval_loss"), prev.get("eval_loss")),
        delta_color="inverse",
        border=True,
        help="Negative partial log-likelihood de Cox no conjunto de validação "
        "de cada cliente, agregada. Menor é melhor.",
    )
    kpis.metric(
        "Perda de Cox (treino)",
        f"{_as_float(last.get('train_loss')):.4f}",
        _delta(last.get("train_loss"), prev.get("train_loss")),
        delta_color="inverse",
        border=True,
        help="Mesma perda, medida durante o treino local antes da agregação.",
    )

    if "auc_dx" in hist:
        task_kpis = st.container(horizontal=True)
        task_kpis.metric(
            "AUC — diagnóstico",
            f"{_as_float(last.get('auc_dx')):.4f}",
            _delta(last.get("auc_dx"), prev.get("auc_dx")),
            border=True,
            help="Área sob a curva ROC da cabeça de diagnóstico (PDAC vs não-PDAC), "
            "agregada entre os clientes (Figura 4).",
        )
        task_kpis.metric(
            "Acurácia — subtipo molecular",
            f"{_as_float(last.get('acc_subtype')):.4f}",
            _delta(last.get("acc_subtype"), prev.get("acc_subtype")),
            border=True,
            help="Acurácia da cabeça de subtipagem (classical vs basal-like) "
            "sobre os pacientes com subtipo conhecido.",
        )

    if "central_c_index" in hist:
        central_kpis = st.container(horizontal=True)
        central_kpis.metric(
            "C-index — validação central",
            f"{_as_float(last.get('central_c_index')):.4f}",
            _delta(last.get("central_c_index"), prev.get("central_c_index")),
            border=True,
            help="Modelo global avaliado no servidor sobre uma coorte held-out "
            "(validação centralizada / independente).",
        )
        if "central_auc_dx" in hist:
            central_kpis.metric(
                "AUC diag. — validação central",
                f"{_as_float(last.get('central_auc_dx')):.4f}",
                _delta(last.get("central_auc_dx"), prev.get("central_auc_dx")),
                border=True,
            )


def _round_charts(hist: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader(
            "C-index por rodada",
            help="Distribuído = média ponderada dos clientes. Central = modelo "
            "global na coorte do servidor. A linha pontilhada em 0,5 marca o acaso.",
            divider=False,
        )
        cols = [c for c in ("c_index", "central_c_index") if c in hist]
        st.line_chart(hist, x="round", y=cols, height=280)
    with c2, st.container(border=True):
        st.subheader(
            "Perda de Cox por rodada",
            help="Treino vs. avaliação. Divergência entre as duas curvas "
            "sugere overfitting local.",
            divider=False,
        )
        cols = [c for c in ("train_loss", "eval_loss") if c in hist]
        st.line_chart(hist, x="round", y=cols, height=280)


def _client_table(hist: pd.DataFrame, last: pd.Series) -> None:
    with st.container(border=True):
        st.subheader(
            "Métricas por cliente (instituição) — última rodada",
            help="Como cada nó federado se saiu. Heterogeneidade grande entre "
            "clientes indica dados não-IID — considere FedProx.",
            divider=False,
        )
        per_client = last.get("eval_clients") or []
        if not per_client:
            st.caption("Sem métricas por cliente nesta rodada.")
            return

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
                col: st.column_config.NumberColumn(format="%.4f")
                for col in ("perda Cox (treino)", "perda Cox (val)", "C-index")
            },
        )


def _modality_gate_chart(hist: pd.DataFrame) -> None:
    if "modality_gate" not in hist:
        return
    gate_hist = hist[hist["modality_gate"].notna()]
    if gate_hist.empty:
        return
    with st.container(border=True):
        st.subheader(
            "Contribuição por modalidade (co-atenção)",
            help="Peso médio do token [FUSION] sobre cada modalidade na leitura "
            "final. Barras equilibradas indicam que a regularização de "
            "balanceamento está evitando dominância de uma modalidade.",
            divider=False,
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
