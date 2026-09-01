"""
Servidor Flower -- orquestrador do treinamento federado.

Responsável por:
  - inicializar os pesos globais do `MultimodalPDACModel`;
  - selecionar clientes a cada rodada e agregar os pesos recebidos (FedAvg /
    FedProx / FedAdam), opcionalmente com DP-FedAvg;
  - agregar as métricas dos clientes (avaliação distribuída);
  - **avaliação centralizada** (`federated.central_eval`): rodar o modelo global
    a cada rodada sobre uma coorte de validação mantida no servidor (held-out /
    validação externa designada pelo consórcio) -- métricas `central_*`.

Nenhum dado de PACIENTE dos clientes passa pelo servidor -- apenas tensores de
pesos. A coorte de validação central é um dado próprio do servidor.

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
    """Agrega métricas dos clientes ponderando pelo nº de exemplos avaliados.

    O denominador é por chave: uma métrica que só alguns clientes reportam (ex.:
    `auc_dx` quando o *split* local tem uma única classe) é normalizada apenas
    pelos exemplos dos clientes que a enviaram.
    """
    agg: dict[str, float] = {}
    den: dict[str, int] = {}
    for n, m in metrics:
        for key, value in m.items():
            agg[key] = agg.get(key, 0.0) + float(value) * n
            den[key] = den.get(key, 0) + n
    return {key: value / max(den[key], 1) for key, value in agg.items()}


def make_central_eval_fn(cfg: dict):
    """Função de avaliação centralizada (Flower `evaluate_fn`), ou None se desligada.

    Roda o modelo global sobre a coorte de validação do servidor. No demo é um
    `SyntheticPDACDataset` com seed disjunta dos clientes; na prática, uma coorte
    externa designada para validação centralizada/independente.
    """
    ce = cfg["federated"].get("central_eval") or {}
    if not ce.get("enabled"):
        return None

    from torch.utils.data import DataLoader

    from data.dataset import (
        ModalityShapes,
        MultimodalPDACDataset,
        SyntheticPDACDataset,
        collate_multimodal,
    )
    from federated.engine import evaluate as eval_loop
    from utils.common import get_device, set_parameters

    device = get_device()
    mc = cfg["model"]
    shapes = ModalityShapes(
        ct_channels=mc.get("ct_in_channels", 2),
        patch_feat_dim=mc.get("patch_feat_dim", 1024),
        n_genes=mc.get("n_genes", 4),
        n_variant_types=mc.get("n_variant_types", 6),
    )
    manifest = ce.get("manifest")  # coorte de validação PRÓPRIA do servidor
    if manifest:
        from data.preprocessing import ct_transforms

        dataset = MultimodalPDACDataset(
            manifest_csv=manifest,
            data_root=ce.get("data_root", cfg["data"].get("data_root", ".")),
            shapes=shapes,
            split=ce.get("split"),
            ct_transform=ct_transforms(train=False),
            clinical_continuous_cols=cfg["data"].get("clinical_continuous_cols") or [],
            clinical_categorical_cols=cfg["data"].get("clinical_categorical_cols") or [],
        )
    else:
        dataset = SyntheticPDACDataset(
            n_samples=int(ce.get("samples", 96)),
            shapes=shapes,
            modality_dropout=cfg["data"]["modality_dropout"],
            seed=int(ce.get("seed", 999)),
        )
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
        collate_fn=collate_multimodal,
    )
    model = MultimodalPDACModel(**cfg["model"]).to(device)

    def evaluate_fn(server_round: int, parameters, config):
        set_parameters(model, parameters)  # `parameters` já vem como lista de ndarrays
        metrics = eval_loop(model, loader, device)
        loss = metrics.pop("loss")
        gates = {k: v for k, v in metrics.items() if k.startswith("gate_")}
        core = {k: v for k, v in metrics.items() if not k.startswith("gate_")}
        return float(loss), {**{f"central_{k}": v for k, v in core.items()}, **gates}

    return evaluate_fn


def build_strategy(cfg: dict) -> fl.server.strategy.Strategy:
    fed = cfg["federated"]

    # Pesos iniciais a partir de um modelo recém-instanciado (mesma seed em todos).
    init_model = MultimodalPDACModel(**cfg["model"])
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    # Informa a rodada atual aos clientes (usado no logging local por passo).
    def round_config(server_round: int) -> dict:
        return {"server_round": server_round}

    common = dict(
        fraction_fit=fed["fraction_fit"],
        fraction_evaluate=fed["fraction_evaluate"],
        min_fit_clients=fed["min_fit_clients"],
        min_evaluate_clients=fed["min_evaluate_clients"],
        min_available_clients=fed["min_available_clients"],
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=round_config,
        on_evaluate_config_fn=round_config,
        evaluate_fn=make_central_eval_fn(cfg),
    )

    name = fed.get("strategy", "FedAvg")
    if name == "FedProx":
        strategy: fl.server.strategy.Strategy = fl.server.strategy.FedProx(
            proximal_mu=0.01, **common
        )
    elif name == "FedAdam":
        strategy = fl.server.strategy.FedAdam(eta=1e-3, **common)
    else:
        strategy = fl.server.strategy.FedAvg(**common)

    # DP-FedAvg (McMahan et al. 2018): clipping da atualização de cada cliente +
    # ruído gaussiano no agregado, no servidor -- "treinamento com privacidade" (LGPD).
    dp = fed.get("dp") or {}
    if dp.get("enabled"):
        strategy = fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=float(dp.get("noise_multiplier", 1.0)),
            clipping_norm=float(dp.get("clip_norm", 1.0)),
            num_sampled_clients=int(fed["min_fit_clients"]),
        )
    return strategy


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
