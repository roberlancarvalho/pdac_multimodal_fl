"""
Cliente Flower -- nó institucional do consórcio federado.

Cada hospital/centro executa este script apontando para os seus próprios dados
(nunca compartilhados). O cliente:
  1. instancia o `MultimodalPDACModel` local;
  2. recebe os pesos globais do servidor (`set_parameters`);
  3. treina localmente por `local_epochs` (`fit`);
  4. devolve ao servidor apenas os pesos atualizados + métricas (`evaluate`).

Uso:
    python -m federated.client --cid 0 --num-clients 2
    python -m federated.client --cid 1 --num-clients 2 --config configs/default.yaml

Se `data.manifest_csv` estiver vazio no config, usa `SyntheticPDACDataset`
(smoke test) -- particionado de forma disjunta por `--cid`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import SyntheticPDACDataset, collate_multimodal
from federated.config import load_config
from federated.engine import evaluate, train_one_epoch
from models.multimodal_pdac import MultimodalPDACModel
from utils.common import get_device, get_parameters, set_parameters, set_seed


def build_dataloaders(cfg: dict, cid: int, num_clients: int):
    """Cria os loaders de treino/validação para o cliente `cid`.

    Substitua este bloco pelo carregamento do manifesto real da sua instituição
    (`MultimodalPDACDataset`). Aqui, particionamos um dataset sintético.
    """
    data_cfg = cfg["data"]
    if data_cfg.get("manifest_csv"):
        raise NotImplementedError(
            "Carregue MultimodalPDACDataset a partir de "
            f"{data_cfg['manifest_csv']!r} para este cliente."
        )

    full = SyntheticPDACDataset(
        n_samples=data_cfg["synthetic_samples"],
        modality_dropout=data_cfg["modality_dropout"],
        seed=cid,
    )
    # Partição disjunta e determinística por cliente.
    indices = list(range(cid, len(full), num_clients))
    split = int(0.8 * len(indices)) or 1
    train_ds = Subset(full, indices[:split])
    val_ds = Subset(full, indices[split:] or indices[:1])

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_multimodal)
    return train_loader, val_loader


_LOCAL_WRITERS: dict[int, object] = {}  # cache por processo (atores Ray são reutilizados)


def _local_writer(cid: int):
    """SummaryWriter por cliente sob outputs/<run>/tb/ -- só quando PDAC_RUN_DIR existe."""
    run_dir = os.environ.get("PDAC_RUN_DIR")
    if not run_dir:
        return None
    if cid not in _LOCAL_WRITERS:
        try:
            from torch.utils.tensorboard import SummaryWriter

            _LOCAL_WRITERS[cid] = SummaryWriter(
                log_dir=str(Path(run_dir) / "tb" / f"local_cliente_{cid + 1}")
            )
        except Exception:  # noqa: BLE001
            _LOCAL_WRITERS[cid] = None
    return _LOCAL_WRITERS[cid]


class MultimodalPDACClient(fl.client.NumPyClient):
    """Implementação `NumPyClient` do Flower para o modelo multimodal."""

    def __init__(self, cfg: dict, cid: int, num_clients: int) -> None:
        self.cfg = cfg
        self.cid = cid
        self.device = get_device()
        self.model = MultimodalPDACModel(**cfg["model"]).to(self.device)
        self.train_loader, self.val_loader = build_dataloaders(cfg, cid, num_clients)
        self.writer = _local_writer(cid)

    # -- Flower API -----------------------------------------------------------
    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        local_epochs = self.cfg["train"]["local_epochs"]
        rnd = int(config.get("server_round", 1))
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.cfg["train"]["weight_decay"],
        )
        metrics = {}
        for epoch in range(local_epochs):
            step = (rnd - 1) * local_epochs + epoch + 1
            metrics = train_one_epoch(
                self.model, self.train_loader, opt, self.device,
                writer=self.writer, step=step,
            )
        n_examples = len(self.train_loader.dataset)
        return get_parameters(self.model), n_examples, {"train_loss": metrics["loss"]}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = evaluate(
            self.model, self.val_loader, self.device,
            writer=self.writer, step=int(config.get("server_round", 1)),
        )
        n_examples = len(self.val_loader.dataset)
        return float(metrics["loss"]), n_examples, {"c_index": metrics["c_index"]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cliente Flower -- Multimodal PDAC FL")
    p.add_argument("--cid", type=int, required=True, help="ID do cliente (0..num-clients-1)")
    p.add_argument("--num-clients", type=int, default=2)
    p.add_argument("--server-address", type=str, default=None)
    p.add_argument("--config", type=str, default=None, help="Caminho do YAML de config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"] + args.cid)

    server_address = args.server_address or cfg["federated"]["server_address"]
    client = MultimodalPDACClient(cfg, args.cid, args.num_clients).to_client()
    fl.client.start_client(server_address=server_address, client=client)


if __name__ == "__main__":
    main()
