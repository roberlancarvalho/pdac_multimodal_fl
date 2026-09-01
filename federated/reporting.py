"""
Registro de métricas de uma execução federada -> arquivos em `outputs/<run>/`.

Usado pelo dashboard Streamlit (`streamlit_app.py`) e pelo TensorBoard.
Escreve, de forma incremental:

    outputs/<run>/config.json        -- configuração efetiva da execução
    outputs/<run>/status.json        -- {state, current_round, num_rounds, ...}
    outputs/<run>/history.jsonl      -- uma linha JSON por rodada concluída
    outputs/<run>/global_model.pt    -- último modelo global agregado
    outputs/<run>/tb/                 -- eventos do TensorBoard (scalars/histogramas)
    outputs/<run>/c_index.png        -- gráfico C-index x rodada (gerado ao final)
    outputs/<run>/run.log            -- stdout/stderr do processo (escrito pelo lançador)

`RecordingStrategy` embrulha qualquer `flwr` Strategy e intercepta
`aggregate_fit` / `aggregate_evaluate` para alimentar o `RunRecorder`, sem
alterar a lógica de agregação (FedAvg/FedProx/FedAdam).

TensorBoard:
    tensorboard --logdir outputs
Scalars: global/{c_index,loss_eval,loss_train}, clients/<cliente>/{...},
attention_branch_b/entropy_norm. Histogramas: weights/<ramo>, attention_branch_b/weights.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import Strategy

from models.fusion_attention import MODALITIES

# Bag sintética fixa para acompanhar a evolução da atenção do Ramo B entre rodadas.
_ATTENTION_PROBE_PATCHES = 128
_ATTENTION_PROBE_SEED = 0


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
        self.png_path = self.run_dir / "c_index.png"
        self._buffer: dict[int, dict[str, Any]] = {}
        self._client_labels: dict[str, str] = {}
        self._last_model = None  # nn.Module do último modelo global agregado
        self._t0 = time.time()

        # Orçamento de privacidade (ε) quando DP-FedAvg está ligado.
        self.epsilon: float | None = None
        dp = (cfg["federated"].get("dp") or {})
        if dp.get("enabled"):
            from federated.privacy import epsilon_estimate

            self.epsilon = epsilon_estimate(
                noise_multiplier=dp.get("noise_multiplier", 1.0),
                num_rounds=num_rounds,
                sample_rate=cfg["federated"].get("fraction_fit", 1.0),
            )
            if self.epsilon is not None:
                print(f"[DP] epsilon ~ {self.epsilon:.2f} (delta=1e-5) apos {num_rounds} rodadas")

        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": cfg["model"],
                    "train": cfg["train"],
                    "federated": cfg["federated"],
                    "data": cfg["data"],
                    "num_clients": num_clients,
                    "num_rounds": num_rounds,
                    "dp_epsilon": self.epsilon,
                    "started": self._t0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._tb = _TensorBoard(self.run_dir / "tb")
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
        self._render_png()
        self._tb.close()

    # -- métricas por rodada -------------------------------------------------------
    def record_fit(self, rnd: int, aggregated: dict[str, Any], per_client: list[dict]) -> None:
        entry = self._buffer.setdefault(rnd, {"round": rnd})
        entry["train_loss"] = _get_float(aggregated, "train_loss")
        entry["fit_clients"] = per_client
        self._register_clients(per_client)

    def record_central(self, rnd: int, loss: float | None, metrics: dict[str, Any]) -> None:
        """Avaliação centralizada no servidor (Flower `evaluate_fn`)."""
        entry = self._buffer.setdefault(rnd, {"round": rnd})
        entry["central_loss"] = _as_float(loss)
        for key, value in (metrics or {}).items():
            if key.startswith("central_"):
                entry[key] = _as_float(value)
        if rnd == 0:  # avaliação dos pesos iniciais -- não há record_evaluate p/ a rodada 0
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            self._log_tensorboard(0, entry)
            self._buffer.pop(0, None)

    def record_evaluate(
        self, rnd: int, agg_loss: float | None, aggregated: dict[str, Any], per_client: list[dict]
    ) -> None:
        entry = self._buffer.setdefault(rnd, {"round": rnd})
        entry["eval_loss"] = _as_float(agg_loss)
        entry["c_index"] = _get_float(aggregated, "c_index")
        for key in ("auc_dx", "acc_subtype"):
            if key in (aggregated or {}):
                entry[key] = _get_float(aggregated, key)
        gate = {
            m: _get_float(aggregated, f"gate_{m}")
            for m in MODALITIES
            if f"gate_{m}" in (aggregated or {})
        }
        if gate:
            entry["modality_gate"] = gate
        entry["eval_clients"] = per_client
        entry["wall_time"] = time.time()
        self._register_clients(per_client)

        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        self._log_tensorboard(rnd, entry)
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
            model.eval()
            torch.save(
                {"state_dict": model.state_dict(), "model_cfg": self.cfg["model"]},
                self.model_path,
            )
            self._last_model = model
        except Exception:
            self._last_model = None

    # -- helpers internos --------------------------------------------------------
    def _register_clients(self, per_client: list[dict]) -> None:
        for c in per_client:
            self._client_labels.setdefault(c["cid"], "")
        for i, cid in enumerate(sorted(self._client_labels)):
            self._client_labels[cid] = f"Cliente {i + 1}"

    def _log_tensorboard(self, rnd: int, entry: dict[str, Any]) -> None:
        if not self._tb.enabled:
            return
        self._tb.scalar("global/c_index", entry.get("c_index"), rnd)
        self._tb.scalar("global/loss_eval", entry.get("eval_loss"), rnd)
        self._tb.scalar("global/loss_train", entry.get("train_loss"), rnd)
        self._tb.scalar("global/auc_dx", entry.get("auc_dx"), rnd)
        self._tb.scalar("global/acc_subtype", entry.get("acc_subtype"), rnd)
        for key, value in entry.items():
            if key.startswith("central_"):
                self._tb.scalar(f"central/{key[len('central_'):]}", value, rnd)
        for m, g in entry.get("modality_gate", {}).items():
            self._tb.scalar(f"modality_gate/{m}", g, rnd)

        fit_loss = {c["cid"]: c.get("train_loss") for c in entry.get("fit_clients", [])}
        for c in entry.get("eval_clients", []):
            label = self._client_labels.get(c["cid"], c["cid"])
            self._tb.scalar(f"clients/{label}/c_index", c.get("c_index"), rnd)
            self._tb.scalar(f"clients/{label}/loss_eval", c.get("loss"), rnd)
            self._tb.scalar(f"clients/{label}/loss_train", fit_loss.get(c["cid"]), rnd)

        if self._last_model is not None:
            self._tb.weight_histograms(self._last_model, rnd)
            entropy = self._tb.attention_histogram(
                self._last_model, self.cfg["model"].get("patch_feat_dim", 1024), rnd
            )
            self._tb.scalar("attention_branch_b/entropy_norm", entropy, rnd)

    def _render_png(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rounds, cidx, l_tr, l_ev = [], [], [], []
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                rounds.append(d.get("round"))
                cidx.append(d.get("c_index"))
                l_tr.append(d.get("train_loss"))
                l_ev.append(d.get("eval_loss"))
            if not rounds:
                return

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(rounds, cidx, "o-", color="#3B7DD8", label="C-index global")
            ax1.axhline(0.5, ls=":", color="gray", lw=1)
            ax1.set_xticks([r for r in rounds if r is not None])
            ax1.set_xlabel("rodada federada")
            ax1.set_ylabel("C-index", color="#3B7DD8")
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis="y", labelcolor="#3B7DD8")

            ax2 = ax1.twinx()
            ax2.plot(rounds, l_tr, "s--", color="#C24B4B", alpha=0.8, label="perda Cox (treino)")
            ax2.plot(rounds, l_ev, "^--", color="#E08A3C", alpha=0.8, label="perda Cox (avaliação)")
            ax2.set_ylabel("perda de Cox")

            lines = ax1.get_lines()[:1] + ax2.get_lines()
            ax1.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)
            fig.suptitle(f"{self.run_dir.name} — {self.cfg['federated'].get('strategy', 'FedAvg')}")
            fig.tight_layout()
            fig.savefig(self.png_path, dpi=150)
            plt.close(fig)
        except Exception:
            pass


class _TensorBoard:
    """Wrapper fino do SummaryWriter -- silenciosamente inerte se indisponível."""

    def __init__(self, log_dir: Path) -> None:
        self.enabled = False
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.enabled = True
        except Exception:
            pass

    def scalar(self, tag: str, value: Any, step: int) -> None:
        v = _as_float(value)
        if self.enabled and v is not None and math.isfinite(v):
            self.writer.add_scalar(tag, v, step)

    def weight_histograms(self, model, step: int) -> None:
        try:
            import torch

            groups: dict[str, list] = {}
            for name, param in model.named_parameters():
                groups.setdefault(name.split(".", 1)[0], []).append(param.detach().reshape(-1))
            for group, tensors in groups.items():
                flat = torch.cat(tensors)
                if flat.numel():
                    self.writer.add_histogram(f"weights/{group}", flat, step)
        except Exception:
            pass

    def attention_histogram(self, model, patch_feat_dim: int, step: int) -> float | None:
        try:
            import torch

            gen = torch.Generator().manual_seed(_ATTENTION_PROBE_SEED)
            bag = torch.randn(1, _ATTENTION_PROBE_PATCHES, patch_feat_dim, generator=gen)
            mask = torch.ones(1, _ATTENTION_PROBE_PATCHES, dtype=torch.bool)
            with torch.no_grad():
                _, attn = model.branch_b(bag, mask)
            attn = attn.squeeze(0).clamp_min(1e-12)
            if self.enabled:
                self.writer.add_histogram("attention_branch_b/weights", attn, step)
            entropy = float(-(attn * attn.log()).sum())
            return entropy / math.log(_ATTENTION_PROBE_PATCHES)  # 1.0 = uniforme, ~0 = concentrada
        except Exception:
            return None

    def close(self) -> None:
        if self.enabled:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
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
        result = self.inner.evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.recorder.record_central(server_round, loss, metrics or {})
        return result


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_float(mapping: Any, key: str) -> float | None:
    if not mapping:
        return None
    return _as_float(mapping.get(key))
