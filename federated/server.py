"""
Servidor Flower -- orquestrador do treinamento federado.

Responsável por:
  - inicializar os pesos globais do `MultimodalPDACModel`;
  - selecionar clientes a cada rodada e agregar os pesos recebidos (FedAvg /
    FedProx / FedAdam);
  - agregar as métricas de avaliação (C-index) ponderadas por nº de amostras.

Nenhum dado de paciente passa pelo servidor -- apenas tensores de pesos.

Uso:
    python -m federated.server
    python -m federated.server --config configs/default.yaml --num-rounds 20
"""

from __future__ import annotations

import argparse

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters

from federated.config import load_config
from models.multimodal_pdac import MultimodalPDACModel
from utils.common import get_parameters


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Agrega métricas dos clientes ponderando pelo nº de exemplos avaliados."""
    total = sum(n for n, _ in metrics) or 1
    agg: dict[str, float] = {}
    for n, m in metrics:
        for key, value in m.items():
            agg[key] = agg.get(key, 0.0) + float(value) * n
    return {key: value / total for key, value in agg.items()}


def build_strategy(cfg: dict) -> fl.server.strategy.Strategy:
    fed = cfg["federated"]

    # Pesos iniciais a partir de um modelo recém-instanciado (mesma seed em todos).
    init_model = MultimodalPDACModel(**cfg["model"])
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    common = dict(
        fraction_fit=fed["fraction_fit"],
        fraction_evaluate=fed["fraction_evaluate"],
        min_fit_clients=fed["min_fit_clients"],
        min_evaluate_clients=fed["min_evaluate_clients"],
        min_available_clients=fed["min_available_clients"],
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    name = fed.get("strategy", "FedAvg")
    if name == "FedProx":
        return fl.server.strategy.FedProx(proximal_mu=0.01, **common)
    if name == "FedAdam":
        return fl.server.strategy.FedAdam(eta=1e-3, **common)
    return fl.server.strategy.FedAvg(**common)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Servidor Flower -- Multimodal PDAC FL")
    p.add_argument("--config", type=str, default=None, help="Caminho do YAML de config")
    p.add_argument("--server-address", type=str, default=None)
    p.add_argument("--num-rounds", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    server_address = args.server_address or cfg["federated"]["server_address"]
    num_rounds = args.num_rounds or cfg["federated"]["num_rounds"]

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=build_strategy(cfg),
    )


if __name__ == "__main__":
    main()
