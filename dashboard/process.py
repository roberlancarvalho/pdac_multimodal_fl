"""Disparo e controle da simulação federada como subprocesso."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from dashboard.io import OUTPUTS, PROJECT_ROOT, read_json


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
    if not run_dir:
        return
    status_path = Path(run_dir) / "status.json"
    status = read_json(status_path) or {}
    status.update({"state": "failed", "error": "cancelado pelo usuário", "updated": time.time()})
    status_path.write_text(json.dumps(status), encoding="utf-8")
