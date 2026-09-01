"""
Pré-processamento **offline** para o Pipeline Multimodal Federado para PDAC.

Cada instituição roda isto uma vez sobre os dados brutos e alimenta o
`MultimodalPDACDataset` (via manifesto CSV). Nada aqui roda no laço federado.

  - TC (NIfTI)  -> `ct_transforms` / `load_ct`  : resample isotrópico, janelamento
                  HU pancreático, crop da ROI, resize; suporta fases AP + VP.
  - WSI         -> `load_patch_embeddings`      : lê os embeddings de patches
                  (gerados por UNI/Virchow) e faz pad/truncate para `n_patches`.
  - Genômica    -> `parse_mutations`            : MAF (TCGA) ou CSV simples ->
                  (status, tipo de variante, VAF) por gene driver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.branch_c_genomics import PDAC_DRIVER_GENES

# MAF `Variant_Classification` -> código do Ramo C (ver models/branch_c_genomics.VARIANT_TYPES).
VARIANT_CLASS_CODE: dict[str, int] = {
    "Missense_Mutation": 1,
    "Nonsense_Mutation": 2,
    "Nonstop_Mutation": 2,
    "Frame_Shift_Del": 3,
    "Frame_Shift_Ins": 3,
    "Splice_Site": 4,
    "Splice_Region": 4,
    "In_Frame_Del": 5,
    "In_Frame_Ins": 5,
    "Translation_Start_Site": 5,
}


# --------------------------------------------------------------------------- #
# Tomografia computadorizada                                                   #
# --------------------------------------------------------------------------- #
def ct_transforms(
    train: bool = False,
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0),
    hu_window: tuple[float, float] = (-100.0, 240.0),
    roi: tuple[int, int, int] = (64, 64, 64),
):
    """`monai.transforms.Compose` para pré-processar um volume de TC.

    Resample isotrópico -> janelamento HU pancreático -> [0,1] -> crop da ROI de
    tecido -> resize/pad para `roi`. Com `train=True` adiciona *augmentation* leve.
    """
    from monai import transforms as mt

    keys = ["image"]
    steps = [
        mt.LoadImaged(keys, image_only=True),
        mt.EnsureChannelFirstd(keys),
        mt.Orientationd(keys, axcodes="RAS"),
        mt.Spacingd(keys, pixdim=spacing, mode="bilinear"),
        mt.ScaleIntensityRanged(
            keys, a_min=hu_window[0], a_max=hu_window[1], b_min=0.0, b_max=1.0, clip=True
        ),
        mt.CropForegroundd(keys, source_key="image"),
        mt.ResizeWithPadOrCropd(keys, spatial_size=roi),
    ]
    if train:
        steps += [
            mt.RandFlipd(keys, prob=0.3, spatial_axis=0),
            mt.RandGaussianNoised(keys, prob=0.2, std=0.02),
        ]
    steps.append(mt.EnsureTyped(keys, dtype=torch.float32))
    return mt.Compose(steps)


def load_ct(
    path: str | Path | None,
    transform=None,
    phases: list[str | Path] | None = None,
) -> torch.Tensor:
    """Carrega e pré-processa 1 volume, ou N fases empilhadas.

    Returns:
        Tensor (n_phases, D, H, W) -- 1 canal por fase, no mesmo grid.
    """
    tf = transform or ct_transforms()
    if phases:
        vols = [tf({"image": str(p)})["image"] for p in phases]  # cada (1, D, H, W)
        return torch.cat([v.as_tensor() if hasattr(v, "as_tensor") else v for v in vols], dim=0)
    vol = tf({"image": str(path)})["image"]
    return vol.as_tensor() if hasattr(vol, "as_tensor") else vol


# --------------------------------------------------------------------------- #
# Histopatologia (WSI)                                                         #
# --------------------------------------------------------------------------- #
def load_patch_embeddings(
    path: str | Path, n_patches: int, feat_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lê embeddings de patches (.npy/.pt de shape (N, F)) -> (n_patches, feat_dim) + máscara.

    Faz pad (zeros) ou trunca para `n_patches`; a máscara marca os patches reais.
    """
    p = Path(path)
    if p.suffix == ".pt":
        arr = torch.load(p, map_location="cpu")
        arr = arr if torch.is_tensor(arr) else torch.as_tensor(arr)
    else:
        arr = torch.from_numpy(np.load(p))
    arr = arr.float()
    if arr.ndim != 2:
        raise ValueError(f"{p}: esperado (N, F), recebido {tuple(arr.shape)}")

    n = min(arr.shape[0], n_patches)
    out = torch.zeros(n_patches, feat_dim)
    out[:n] = arr[:n, :feat_dim]
    mask = torch.zeros(n_patches, dtype=torch.bool)
    mask[:n] = True
    return out, mask


# --------------------------------------------------------------------------- #
# Genômica                                                                     #
# --------------------------------------------------------------------------- #
def parse_mutations(
    path: str | Path, genes: tuple[str, ...] = PDAC_DRIVER_GENES
) -> dict[str, tuple[int, int, float]]:
    """Lê um MAF (TCGA) ou CSV simples -> {gene: (status, variant_type, vaf)}.

    MAF: colunas `Hugo_Symbol`, `Variant_Classification` e, se houver,
    `t_alt_count` + `t_depth` (VAF derivada) ou `VAF`.
    CSV simples: colunas `gene`, `status` (0/1), `variant_type` (0-5), `vaf` (0-1).
    Genes não listados no arquivo ficam como (0, 0, 0.0) = wild-type.
    """
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    cols = {c.lower(): c for c in df.columns}
    result: dict[str, tuple[int, int, float]] = {g: (0, 0, 0.0) for g in genes}

    if "hugo_symbol" in cols:  # formato MAF
        gcol, vcol = cols["hugo_symbol"], cols.get("variant_classification")
        for g in genes:
            rows = df[df[gcol] == g]
            if rows.empty:
                continue
            row = rows.iloc[0]
            vtype = VARIANT_CLASS_CODE.get(str(row[vcol]), 5) if vcol else 1
            vaf = 0.0
            if "t_alt_count" in cols and "t_depth" in cols:
                depth = float(row[cols["t_depth"]] or 0)
                if depth > 0:
                    vaf = float(row[cols["t_alt_count"]]) / depth
            elif "vaf" in cols:
                vaf = float(row[cols["vaf"]])
            result[g] = (1, int(vtype), float(np.clip(vaf, 0.0, 1.0)))
    else:  # CSV simples
        gcol = cols.get("gene", df.columns[0])
        for g in genes:
            rows = df[df[gcol] == g]
            if rows.empty:
                continue
            row = rows.iloc[0]
            status = int(row[cols["status"]]) if "status" in cols else 1
            vtype = int(row[cols["variant_type"]]) if "variant_type" in cols else (1 if status else 0)
            vaf = float(row[cols["vaf"]]) if "vaf" in cols else 0.0
            result[g] = (status, vtype, float(np.clip(vaf, 0.0, 1.0)))

    return result
