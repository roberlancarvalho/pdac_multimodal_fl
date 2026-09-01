"""
Registro de métricas de uma execução federada -> arquivos em `outputs/<run>/`.

Usado pelo dashboard Streamlit (`streamlit_app.py`) para acompanhar o treino
"em tela". Escreve, de forma incremental:

    outputs/<run>/config.json        -- configuração efetiva da execução
    outputs/<run>/status.json        -- {state, current_round, num_rounds, ...}
    outputs/<run>/history.jsonl      -- uma linha JSON por rodada concluída
    outputs/<run>/global_model.pt    -- último modelo global agregado (para a aba de atenção)
    outputs/<run>/run.log            -- stdout/stderr do processo (escrito pelo lançador)

`RecordingStrategy` embrulha qualquer `flwr` Strategy e intercepta
`aggregate_fit` / `aggregate_evaluate` para alimentar o `RunRecorder`, sem
alterar a lógica de agregação (FedAvg/FedProx/FedAdam).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import Strategy


class RunRecorder:
    """Persiste o progresso de uma execução federada em `run_dir`."""

    def __init__(self, run_dir: str | Path, cfg: dict, num_clients: int, num_rounds: int) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.num_rounds = num_rounds
        self.history_path = self.run_dir / "history.jsonl"
        self.status_path = self.run_dir / "status.json"
        self.model_path = self.run_dir / "global_model.pt"
        self._buffer: dict[int, dict[str, Any]] = {}
        self._t0 = time.time()

        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": cfg["model"],
                    "train": cfg["train"],
                    "federated": cfg["federated"],
                    "data": cfg["data"],
                    "num_clients": num_clients,
                    "num_rounds": num_rounds,
                    "started": self._t0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.write_status("running", 0)

    # -- status ---------------------------------------------------------------
    def write_status(self, state: str, current_round: int, error: str | None = None) -> None:
        self.status_path.write_text(
            json.dumps(
                {
                    "state": state,  # running | done | failed
                    "current_round": current_round,
                    "num_rounds": self.num_rounds,
                    "elapsed_s": round(time.time() - self._t0, 1),
                    "updated": time.time(),
                    "error": error,
                }
            ),
            encoding="utf-8",
        )

    def finish(self, error: str | None = None) -> None:
        self.write_status("failed" if error else "done", self.num_rounds, error)

    # -- métricas por rodada ------------------------------------------------------
    def record_fit(self, rnd: int, aggregated: dict[str, Any], per_client: list[dict]) -> None:
        entry = self._buffer.setdefault(rnd, {"round": rnd})
        entry["train_loss"] = _get_float(aggregated, "train_loss")
        entry["fit_clients"] = per_client

    def record_evaluate(
        self, rnd: int, agg_loss: float | None, aggregated: dict[str, Any], per_client: list[dict]
    ) -> None:
        entry = self._buffer.setdefault(rnd, {"round": rnd})
        entry["eval_loss"] = _as_float(agg_loss)
        entry["c_index"] = _get_float(aggregated, "c_index")
        entry["eval_clients"] = per_client
        entry["wall_time"] = time.time()
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self._buffer.pop(rnd, None)
        self.write_status("running", rnd)

    def save_global_model(self, parameters) -> None:
        try:
            import torch

            from models.multimodal_pdac import MultimodalPDACModel
            from utils.common import set_parameters

            ndarrays = parameters_to_ndarrays(parameters)
            model = MultimodalPDACModel(**self.cfg["model"])
            set_parameters(model, ndarrays)
            torch.save(
                {"state_dict": model.state_dict(), "model_cfg": self.cfg["model"]},
                self.model_path,
            )
        except Exception:  # noqa: BLE001 -- salvar o modelo é best-effort
            pass


class RecordingStrategy(Strategy):
    """Decorator de Strategy que registra métricas sem mudar a agregação."""

    def __init__(self, inner: Strategy, recorder: RunRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def initialize_parameters(self, client_manager):
        return self.inner.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.inner.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = self.inner.aggregate_fit(server_round, results, failures)
        params, metrics = aggregated
        per_client = [
            {
                "cid": str(cp.cid),
                "num_examples": res.num_examples,
                "train_loss": _get_float(res.metrics, "train_loss"),
            }
            for cp, res in results
        ]
        self.recorder.record_fit(server_round, metrics or {}, per_client)
        if params is not None:
            self.recorder.save_global_model(params)
        return aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.inner.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = self.inner.aggregate_evaluate(server_round, results, failures)
        loss, metrics = aggregated
        per_client = [
            {
                "cid": str(cp.cid),
                "num_examples": res.num_examples,
                "loss": _as_float(res.loss),
                "c_index": _get_float(res.metrics, "c_index"),
            }
            for cp, res in results
        ]
        self.recorder.record_evaluate(server_round, loss, metrics or {}, per_client)
        return aggregated

    def evaluate(self, server_round, parameters):
        return self.inner.evaluate(server_round, parameters)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_float(mapping: Any, key: str) -> float | None:
    if not mapping:
        return None
    return _as_float(mapping.get(key))
