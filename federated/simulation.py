"""
Simulação federada local (um processo, N clientes virtuais).

Forma mais rápida de validar o pipeline ponta a ponta sem abrir vários
terminais. Usa `flwr.simulation.start_simulation`.

Uso:
    python -m federated.simulation --num-clients 3 --num-rounds 5
    python -m federated.simulation --num-clients 3 --num-rounds 5 --run-dir outputs/run_x
"""

from __future__ import annotations

import argparse
import traceback

import flwr as fl

from federated.client import MultimodalPDACClient
from federated.config import load_config
from federated.server import build_strategy
from utils.common import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulação Flower -- Multimodal PDAC FL")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--num-clients", type=int, default=2)
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Se definido, registra métricas/modelo em outputs/<run> para o dashboard.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    num_rounds = args.num_rounds or cfg["federated"]["num_rounds"]

    # Garante que a estratégia aceite o nº de clientes da simulação.
    cfg["federated"]["min_fit_clients"] = args.num_clients
    cfg["federated"]["min_evaluate_clients"] = args.num_clients
    cfg["federated"]["min_available_clients"] = args.num_clients

    def client_fn(context: fl.common.Context) -> fl.client.Client:
        cid = int(context.node_config["partition-id"])
        return MultimodalPDACClient(cfg, cid, args.num_clients).to_client()

    strategy = build_strategy(cfg)

    recorder = None
    if args.run_dir:
        from federated.reporting import RecordingStrategy, RunRecorder

        recorder = RunRecorder(args.run_dir, cfg, args.num_clients, num_rounds)
        strategy = RecordingStrategy(strategy, recorder)

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001
        if recorder is not None:
            recorder.finish(error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    else:
        if recorder is not None:
            recorder.finish()


if __name__ == "__main__":
    main()
