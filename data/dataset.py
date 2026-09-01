"""
Dataset multimodal para PDAC.

`MultimodalPDACDataset` carrega amostras a partir de um **manifesto CSV** (uma
linha por paciente). Colunas -- todas as de modalidade são opcionais (célula
vazia = modalidade ausente):

    patient_id                (obrigatória)
    ct_ap, ct_vp   OU  ct_path   caminho(s) NIfTI de TC (fases AP/VP ou 1 volume)
    wsi_emb                       .npy/.pt com embeddings de patches (N, F) da WSI
    mutations      OU  <GENE>_status,<GENE>_vtype,<GENE>_vaf   MAF/CSV ou colunas diretas
    time, event                  sobrevida (Cox)
    dx                           diagnóstico 0/1  (-1 ou vazio = desconhecido)
    subtype                      0=classical, 1=basal-like (-1/vazio = desconhecido)
    split                        train | val | test (opcional)

`SyntheticPDACDataset` gera dados aleatórios com as mesmas chaves/shapes, para
testar o pipeline sem dados reais.

Cada item é um dict compatível com `MultimodalPDACModel.forward` + rótulos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.branch_c_genomics import PDAC_DRIVER_GENES
from models.fusion_attention import MODALITIES


def _isna(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip() == ""


@dataclass
class ModalityShapes:
    ct_shape: tuple[int, int, int] = (48, 64, 64)   # (D, H, W) após pré-processamento
    ct_channels: int = 2                             # 1 = single-phase; 2 = AP + VP
    n_patches: int = 256
    patch_feat_dim: int = 1024
    n_genes: int = 4
    n_variant_types: int = 6
    clinical_n_continuous: int = 5
    clinical_cat_cardinalities: tuple[int, ...] = (2, 4, 5, 3)


class MultimodalPDACDataset(Dataset):
    """Carrega amostras multimodais a partir de um manifesto CSV (ver docstring do módulo).

    Args:
        manifest_csv: CSV com uma linha por paciente.
        data_root: raiz para resolver caminhos relativos.
        shapes: `ModalityShapes` (n_patches / patch_feat_dim usados no pad da WSI).
        ct_transform: `Compose` da MONAI para a TC (default: `preprocessing.ct_transforms()`).
        split: se dado e houver coluna `split`, filtra as linhas.
        genes: genes driver, na ordem das colunas de `mutation_status`.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        data_root: str | Path = ".",
        shapes: ModalityShapes | None = None,
        ct_transform=None,
        split: str | None = None,
        genes: tuple[str, ...] = PDAC_DRIVER_GENES,
        clinical_continuous_cols: list[str] | None = None,
        clinical_categorical_cols: list[str] | None = None,
    ) -> None:
        import pandas as pd

        self.data_root = Path(data_root)
        self.shapes = shapes or ModalityShapes()
        self.ct_transform = ct_transform
        self.genes = list(genes)
        self.clin_cont = list(clinical_continuous_cols or [])
        self.clin_cat = list(clinical_categorical_cols or [])

        df = pd.read_csv(manifest_csv)
        if "patient_id" not in df.columns:
            raise ValueError("manifesto sem coluna obrigatória 'patient_id'")
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str) == split]
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, value) -> Path | None:
        if _isna(value):
            return None
        p = Path(str(value))
        return p if p.is_absolute() else self.data_root / p

    def _load_ct(self, row) -> torch.Tensor | None:
        phase_paths = [self._resolve(row[c]) for c in ("ct_ap", "ct_vp") if c in row.index]
        phase_paths = [p for p in phase_paths if p is not None]
        if not phase_paths and "ct_path" in row.index:
            single = self._resolve(row["ct_path"])
            phase_paths = [single] if single is not None else []
        if not phase_paths:
            return None

        from data.preprocessing import load_ct

        if len(phase_paths) == 1:
            return load_ct(phase_paths[0], self.ct_transform).float()
        return load_ct(None, self.ct_transform, phases=phase_paths).float()

    def _load_wsi(self, row):
        path = self._resolve(row["wsi_emb"]) if "wsi_emb" in row.index else None
        if path is None:
            return None
        from data.preprocessing import load_patch_embeddings

        return load_patch_embeddings(path, self.shapes.n_patches, self.shapes.patch_feat_dim)

    def _load_genomics(self, row):
        mpath = self._resolve(row["mutations"]) if "mutations" in row.index else None
        if mpath is not None:
            from data.preprocessing import parse_mutations

            parsed = parse_mutations(mpath, tuple(self.genes))
            triples = [parsed[g] for g in self.genes]
        elif f"{self.genes[0]}_status" in row.index:
            triples = [
                (
                    int(row.get(f"{g}_status", 0) or 0),
                    int(row.get(f"{g}_vtype", 0) or 0),
                    float(row.get(f"{g}_vaf", 0.0) or 0.0),
                )
                for g in self.genes
            ]
        else:
            return None
        status, vtype, vaf = zip(*triples, strict=True)
        return (
            torch.tensor(status, dtype=torch.long),
            torch.tensor(vtype, dtype=torch.long),
            torch.tensor(vaf, dtype=torch.float32),
        )

    def _load_clinical(self, row):
        if not self.clin_cont and not self.clin_cat:
            return None
        if all(_isna(row.get(c)) for c in (*self.clin_cont, *self.clin_cat)):
            return None
        num = torch.tensor(
            [0.0 if _isna(row.get(c)) else float(row[c]) for c in self.clin_cont],
            dtype=torch.float32,
        )
        cat = torch.tensor(
            [0 if _isna(row.get(c)) else int(row[c]) for c in self.clin_cat], dtype=torch.long
        )
        return num, cat

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item: dict = {"patient_id": str(row.get("patient_id", idx))}
        item["time"] = torch.tensor(float(row.get("time", 0.0) or 0.0))
        item["event"] = torch.tensor(float(row.get("event", 0.0) or 0.0))
        item["dx"] = torch.tensor(-1.0 if _isna(row.get("dx")) else float(row["dx"]))
        item["subtype"] = torch.tensor(-1 if _isna(row.get("subtype")) else int(row["subtype"]))

        ct = self._load_ct(row)
        if ct is not None:
            item["ct_volume"] = ct

        wsi = self._load_wsi(row)
        if wsi is not None:
            item["patch_embeddings"], item["patch_mask"] = wsi

        genomics = self._load_genomics(row)
        if genomics is not None:
            item["mutation_status"], item["variant_type"], item["vaf"] = genomics

        clinical = self._load_clinical(row)
        if clinical is not None:
            item["clinical_num"], item["clinical_cat"] = clinical

        if not any(
            k in item for k in ("ct_volume", "patch_embeddings", "mutation_status", "clinical_num")
        ):
            raise ValueError(f"paciente {item['patient_id']} não tem nenhuma modalidade no manifesto")
        return item


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
            "event": torch.tensor(float(r.integers(0, 2))),            # prognóstico (Cox)
            "dx": torch.tensor(float(r.integers(0, 2))),               # diagnóstico: 0=não-PDAC, 1=PDAC
            "subtype": torch.tensor(int(r.integers(-1, 2))),           # subtipo: -1=NA, 0=classical, 1=basal-like
        }

        def keep() -> bool:
            return r.random() >= self.modality_dropout

        if keep():
            item["ct_volume"] = torch.from_numpy(
                r.standard_normal((s.ct_channels, *s.ct_shape)).astype("float32")
            )
        if keep():
            item["patch_embeddings"] = torch.from_numpy(
                r.standard_normal((s.n_patches, s.patch_feat_dim)).astype("float32")
            )
            item["patch_mask"] = torch.ones(s.n_patches, dtype=torch.bool)
        if keep():
            mut = r.integers(0, 2, size=s.n_genes).astype("int64")
            item["mutation_status"] = torch.from_numpy(mut)
            # tipo de variante e VAF só fazem sentido nos genes mutados
            vtype = np.where(mut == 1, r.integers(1, s.n_variant_types, size=s.n_genes), 0)
            vaf = np.where(mut == 1, r.uniform(0.05, 0.95, size=s.n_genes), 0.0)
            item["variant_type"] = torch.from_numpy(vtype.astype("int64"))
            item["vaf"] = torch.from_numpy(vaf.astype("float32"))
        if keep():
            item["clinical_num"] = torch.from_numpy(
                r.standard_normal(s.clinical_n_continuous).astype("float32")
            )
            item["clinical_cat"] = torch.tensor(
                [int(r.integers(0, c)) for c in s.clinical_cat_cardinalities], dtype=torch.long
            )

        # Garante ao menos uma modalidade presente.
        if not any(
            k in item
            for k in ("ct_volume", "patch_embeddings", "mutation_status", "clinical_num")
        ):
            item["mutation_status"] = torch.zeros(s.n_genes, dtype=torch.long)

        return item


def collate_multimodal(batch: list[dict]) -> dict:
    """Collate que empilha modalidades presentes em todo o lote e monta a máscara.

    Simplificação: assume que, quando presente, uma modalidade tem a mesma shape
    em todas as amostras do lote (o dataset real deve fazer padding de patches).
    Amostras sem uma modalidade recebem tensores zerados + `modality_mask=False`.
    """
    b = len(batch)

    out: dict = {
        "time": torch.stack([x["time"] for x in batch]),
        "event": torch.stack([x["event"] for x in batch]),
        "patient_id": [x["patient_id"] for x in batch],
    }
    for label in ("dx", "subtype"):
        if label in batch[0]:
            out[label] = torch.stack([x[label] for x in batch])
    modality_mask = torch.zeros(b, len(MODALITIES), dtype=torch.bool)

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
        vt = stack_optional("variant_type", 2)
        if vt is not None:
            out["variant_type"] = vt.long()
        vaf = stack_optional("vaf", 2)
        if vaf is not None:
            out["vaf"] = vaf.float()

    cn = stack_optional("clinical_num", 3)
    if cn is not None:
        out["clinical_num"] = cn.float()
        cc = stack_optional("clinical_cat", 3)
        out["clinical_cat"] = (cc if cc is not None else torch.zeros(b, 0, dtype=torch.long)).long()

    out["modality_mask"] = modality_mask
    return out
