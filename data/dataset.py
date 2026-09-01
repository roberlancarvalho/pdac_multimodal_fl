"""
Dataset multimodal para PDAC.

`MultimodalPDACDataset` é um ESQUELETO. A intenção é que cada instituição
participante do consórcio federado implemente o carregamento a partir do seu
próprio manifesto (CSV) apontando para:
    - volume de TC pré-processado (.nii.gz)
    - arquivo de embeddings de patches da WSI (.pt / .npy) gerado offline pelo
      foundation model de patologia
    - status mutacional dos 4 genes driver
    - rótulos de sobrevida (tempo, evento)

`SyntheticPDACDataset` gera dados aleatórios com as mesmas shapes/chaves, para
testar o pipeline federado ponta a ponta sem dados reais.

Cada item é um dict compatível com `MultimodalPDACModel.forward` + rótulos:
    {
        "ct_volume": (1, D, H, W) float32            | ausente,
        "patch_embeddings": (N, patch_feat_dim) f32   | ausente,
        "patch_mask": (N,) bool                       | ausente,
        "mutation_status": (4,) long                  | ausente,
        "time": () float32,
        "event": () float32,
        "patient_id": str,
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ModalityShapes:
    ct_shape: tuple[int, int, int] = (48, 64, 64)   # (D, H, W) após pré-processamento
    n_patches: int = 256
    patch_feat_dim: int = 1024
    n_genes: int = 4


class MultimodalPDACDataset(Dataset):
    """Carrega amostras multimodais a partir de um manifesto CSV (a implementar).

    Args:
        manifest_csv: CSV com uma linha por paciente e colunas de caminho/rótulo.
        data_root: Raiz para resolver caminhos relativos do manifesto.
        shapes: Shapes esperadas das modalidades (para padding de patches, etc.).
        transform: Transform MONAI/torchvision aplicada ao volume de TC (opcional).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        data_root: str | Path,
        shapes: ModalityShapes | None = None,
        transform=None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.data_root = Path(data_root)
        self.shapes = shapes or ModalityShapes()
        self.transform = transform
        # TODO: carregar o CSV (pandas) e validar colunas.
        raise NotImplementedError(
            "Implemente o carregamento a partir do manifesto da sua instituição. "
            "Use SyntheticPDACDataset para testar o pipeline federado."
        )

    def __len__(self) -> int:  # pragma: no cover - esqueleto
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:  # pragma: no cover - esqueleto
        raise NotImplementedError


class SyntheticPDACDataset(Dataset):
    """Dataset sintético com as mesmas chaves/shapes do dataset real.

    Útil para smoke tests do servidor/cliente Flower e do modelo.
    """

    def __init__(
        self,
        n_samples: int = 64,
        shapes: ModalityShapes | None = None,
        modality_dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.shapes = shapes or ModalityShapes()
        self.modality_dropout = modality_dropout
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        s = self.shapes
        r = np.random.default_rng(idx + 1)

        item: dict = {
            "patient_id": f"SYN-{idx:04d}",
            "time": torch.tensor(float(r.uniform(1.0, 60.0))),          # meses
            "event": torch.tensor(float(r.integers(0, 2))),
        }

        def keep() -> bool:
            return r.random() >= self.modality_dropout

        if keep():
            item["ct_volume"] = torch.from_numpy(
                r.standard_normal((1, *s.ct_shape)).astype("float32")
            )
        if keep():
            item["patch_embeddings"] = torch.from_numpy(
                r.standard_normal((s.n_patches, s.patch_feat_dim)).astype("float32")
            )
            item["patch_mask"] = torch.ones(s.n_patches, dtype=torch.bool)
        if keep():
            item["mutation_status"] = torch.from_numpy(
                r.integers(0, 2, size=s.n_genes).astype("int64")
            )

        # Garante ao menos uma modalidade presente.
        if not any(k in item for k in ("ct_volume", "patch_embeddings", "mutation_status")):
            item["mutation_status"] = torch.zeros(s.n_genes, dtype=torch.long)

        return item


def collate_multimodal(batch: list[dict]) -> dict:
    """Collate que empilha modalidades presentes em todo o lote e monta a máscara.

    Simplificação: assume que, quando presente, uma modalidade tem a mesma shape
    em todas as amostras do lote (o dataset real deve fazer padding de patches).
    Amostras sem uma modalidade recebem tensores zerados + `modality_mask=False`.
    """
    keys_order = ("radiomics", "histology", "genomics")
    b = len(batch)

    out: dict = {
        "time": torch.stack([x["time"] for x in batch]),
        "event": torch.stack([x["event"] for x in batch]),
        "patient_id": [x["patient_id"] for x in batch],
    }
    modality_mask = torch.zeros(b, 3, dtype=torch.bool)

    def stack_optional(key: str, mod_idx: int):
        present = [i for i, x in enumerate(batch) if key in x]
        if not present:
            return None
        ref = batch[present[0]][key]
        full = torch.zeros(b, *ref.shape, dtype=ref.dtype)
        for i in present:
            full[i] = batch[i][key]
            modality_mask[i, mod_idx] = True
        return full

    ct = stack_optional("ct_volume", 0)
    if ct is not None:
        out["ct_volume"] = ct

    pe = stack_optional("patch_embeddings", 1)
    if pe is not None:
        out["patch_embeddings"] = pe
        pm = stack_optional("patch_mask", 1)
        # onde a amostra não tem patches, marca tudo como inválido
        if pm is None:
            pm = torch.zeros(b, pe.shape[1], dtype=torch.bool)
        out["patch_mask"] = pm.bool()

    ms = stack_optional("mutation_status", 2)
    if ms is not None:
        out["mutation_status"] = ms.long()

    out["modality_mask"] = modality_mask
    return out
