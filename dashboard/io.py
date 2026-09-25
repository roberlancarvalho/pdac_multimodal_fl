"""Leitura dos artefatos de uma execução em `outputs/<run>/`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
TOUR_FLAG = PROJECT_ROOT / ".tour_seen"


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
