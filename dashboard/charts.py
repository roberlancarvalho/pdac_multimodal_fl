"""Construtores de gráficos Altair reutilizados entre as abas."""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd


def round_line_chart(
    hist: pd.DataFrame,
    y_cols: list[str],
    series_labels: dict[str, str],
    y_title: str,
    height: int = 280,
    baseline: float | None = None,
    baseline_label: str = "",
) -> alt.Chart:
    """Linha por rodada com pontos, tooltip e (opcional) régua de referência tracejada."""
    long_df = hist[["round", *y_cols]].melt("round", var_name="serie", value_name="valor").dropna(subset=["valor"])
    long_df["serie"] = long_df["serie"].map(series_labels).fillna(long_df["serie"])
    chart = (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("round:Q", title="rodada", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("valor:Q", title=y_title),
            color=alt.Color("serie:N", title=None),
            tooltip=["serie", "round", alt.Tooltip("valor:Q", format=".4f")],
        )
    )
    if baseline is not None:
        ref = pd.DataFrame({"y": [baseline]})
        rule = alt.Chart(ref).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        label = rule.mark_text(
            text=baseline_label, align="left", baseline="bottom", dx=4, color="gray"
        ).encode()
        chart = chart + rule + label
    return chart.properties(height=height).interactive()


def signed_shap_chart(df: pd.DataFrame, label_col: str, height: int = 240) -> alt.Chart:
    """Barras horizontais assinadas (↑/↓ risco) para valores SHAP."""
    df = df.copy()
    df["sentido"] = df["shap"].apply(lambda v: "↑ risco" if v > 0 else "↓ risco")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("shap:Q", title="valor SHAP (log-hazard)"),
            y=alt.Y(f"{label_col}:N", sort="-x", title=None),
            color=alt.Color(
                "sentido:N",
                scale=alt.Scale(domain=["↑ risco", "↓ risco"], range=["#C24B4B", "#3B7DD8"]),
                title=None,
            ),
            tooltip=[label_col, alt.Tooltip("shap:Q", format="+.4f")],
        )
        .properties(height=height)
    )


def attention_heatmap(attention: np.ndarray, height: int = 340) -> alt.Chart:
    """Mapa de calor dos pesos de atenção numa grade pseudo-quadrada (só p/ visualização)."""
    side = math.ceil(math.sqrt(len(attention)))
    padded = list(attention) + [None] * (side * side - len(attention))
    grid = pd.DataFrame(
        {
            "linha": [i // side for i in range(side * side)],
            "coluna": [i % side for i in range(side * side)],
            "patch": list(range(side * side)),
            "atencao": padded,
        }
    )
    return (
        alt.Chart(grid.dropna())
        .mark_rect()
        .encode(
            x=alt.X("coluna:O", axis=None),
            y=alt.Y("linha:O", axis=None),
            color=alt.Color("atencao:Q", scale=alt.Scale(scheme="magma"), title="atenção"),
            tooltip=["patch", alt.Tooltip("atencao:Q", format=".5f")],
        )
        .properties(height=height)
    )
