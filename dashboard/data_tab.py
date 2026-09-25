"""Aba "Dados" -- de onde vêm os dados da execução e onde enviar/editar um manifesto real."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.io import PROJECT_ROOT, read_json

MANIFEST_TEMPLATE = PROJECT_ROOT / "data" / "manifest_template.csv"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
DATA_GUIDE = PROJECT_ROOT / "docs" / "GUIA_DE_DADOS.md"


def _current_source(run_dir: Path | None) -> None:
    st.subheader("Fonte de dados desta execução", divider=False)
    if run_dir is None:
        st.info("Nenhuma execução ainda -- a fonte de dados aparece aqui assim que uma rodar.")
        return

    cfg = read_json(run_dir / "config.json") or {}
    data_cfg = cfg.get("data", {})
    manifest = data_cfg.get("manifest_csv") or ""

    if not manifest:
        st.info(
            f"🧪 **Dados sintéticos** (gerados na hora, sem paciente real) -- "
            f"{data_cfg.get('synthetic_samples', '?')} amostras, "
            f"dropout de modalidade {float(data_cfg.get('modality_dropout', 0)):.0%}."
        )
        return

    st.success(f"📄 **Manifesto real:** `{manifest}`  ·  raiz dos arquivos: `{data_cfg.get('data_root', '.')}`")
    path = Path(manifest)
    if path.exists():
        try:
            df = pd.read_csv(path)
            st.caption(f"{len(df)} pacientes · colunas: {', '.join(df.columns)}")
        except Exception as exc:
            st.warning(f"Não consegui ler esse manifesto: {exc}")
    else:
        st.warning("Esse caminho de manifesto não existe mais no disco.")


def _manifest_editor() -> None:
    st.subheader("Usar dados reais (próxima execução)", divider=False)
    st.caption(
        "Envie um `manifest.csv` ou aponte um já existente no disco, revise/edite as linhas "
        "e salve -- a próxima simulação disparada pelo painel usa esse manifesto em vez do "
        "sintético. As colunas/arquivos que o manifesto referencia (TC, WSI, mutações) precisam "
        f"existir de fato -- veja o passo a passo em `{DATA_GUIDE.relative_to(PROJECT_ROOT)}`."
    )

    if MANIFEST_TEMPLATE.exists():
        st.download_button(
            "Baixar modelo de manifesto (CSV)",
            data=MANIFEST_TEMPLATE.read_bytes(),
            file_name="manifest_template.csv",
            icon=":material/download:",
        )

    use_manifest = st.toggle(
        "Usar manifesto em vez de dados sintéticos",
        value=bool(st.session_state.get("data_manifest_path")),
    )
    if not use_manifest:
        st.session_state.data_manifest_path = ""
        return

    upload = st.file_uploader("Enviar manifest.csv", type="csv")
    path_input = st.text_input(
        "...ou caminho de um CSV já no disco",
        value=st.session_state.get("data_manifest_path", ""),
        placeholder=r"D:\dados\manifest.csv",
    )

    df = None
    if upload is not None:
        df = pd.read_csv(upload)
    elif path_input and Path(path_input).exists():
        df = pd.read_csv(path_input)
    elif path_input:
        st.warning("Esse caminho não existe.")

    if df is None:
        st.caption("Nenhum manifesto carregado ainda.")
        return

    st.caption("Edite células, adicione ou apague linhas direto na tabela (célula vazia = modalidade ausente).")
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", key="manifest_editor")

    data_root = st.text_input(
        "Raiz dos arquivos (data_root) -- onde os caminhos relativos do manifesto começam",
        value=st.session_state.get("data_root", "./data/processed"),
    )

    if st.button("Salvar e usar este manifesto", type="primary", icon=":material/save:"):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = UPLOADS_DIR / "manifest.csv"
        edited.to_csv(out_path, index=False)
        st.session_state.data_manifest_path = str(out_path)
        st.session_state.data_root = data_root
        st.success(f"Salvo em `{out_path}` -- será usado ao clicar em **Iniciar simulação**.")


def render(run_dir: Path | None) -> None:
    _current_source(run_dir)
    st.divider()
    _manifest_editor()
