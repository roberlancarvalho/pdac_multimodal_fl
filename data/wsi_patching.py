"""
Extração **offline** de patches de WSI + embeddings por Foundation Model.

Roda FORA do laço federado, uma vez por lâmina, e produz o `.npy` `(N, F)` que
o `MultimodalPDACDataset` consome (coluna `wsi_emb` do manifesto).

Dependências externas (NÃO no requirements por serem pesadas/gated):
  - `openslide-python` + binário OpenSlide  -> leitura de .svs/.ndpi/.tiff
  - o Foundation Model de patologia (UNI/Virchow/CONCH) via HuggingFace (acesso
    concedido) -> encoder de patch.

Uso:
    python -m data.wsi_patching --slide LÂMINA.svs --out wsi_emb/LÂMINA.npy \\
        --encoder uni --patch 256 --mpp 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Detecção de tecido (Otsu sobre uma thumbnail em escala de cinza)             #
# --------------------------------------------------------------------------- #
def tissue_mask(thumbnail_rgb: np.ndarray, sat_threshold: float = 0.1) -> np.ndarray:
    """Máscara booleana de tecido (True) a partir de uma thumbnail RGB `(h, w, 3)`.

    Heurística leve: tecido tem saturação alta e não é branco. Substitua por um
    segmentador dedicado (ex.: o do CLAM/Trident) para uso sério.
    """
    x = thumbnail_rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0.0)
    not_white = mx < 0.92
    return (sat > sat_threshold) & not_white


def iter_patch_coords(
    slide_dims: tuple[int, int],
    tissue: np.ndarray,
    patch_px: int,
    thumb_downsample: float,
    min_tissue_frac: float = 0.35,
):
    """Gera coordenadas `(x, y)` (nível 0) dos patches com tecido suficiente."""
    w, h = slide_dims
    step_thumb = max(1, int(patch_px / thumb_downsample))
    th, tw = tissue.shape
    for ty in range(0, th - step_thumb + 1, step_thumb):
        for tx in range(0, tw - step_thumb + 1, step_thumb):
            block = tissue[ty : ty + step_thumb, tx : tx + step_thumb]
            if block.mean() < min_tissue_frac:
                continue
            x0 = int(tx * thumb_downsample)
            y0 = int(ty * thumb_downsample)
            if x0 + patch_px <= w and y0 + patch_px <= h:
                yield x0, y0


# --------------------------------------------------------------------------- #
# Encoders (Foundation Models de patologia)                                    #
# --------------------------------------------------------------------------- #
def load_encoder(name: str):
    """Retorna `callable((N,3,224,224) float -> (N, F) np.ndarray)`.

    `name`:
      - "uni"     -> `timm` `hf-hub:MahmoodLab/UNI` (requer HF_TOKEN + acesso)
      - "random"  -> projeção aleatória fixa (F=1024) -- só para testar o pipeline.
    """
    if name == "random":
        rng = np.random.default_rng(0)
        w = rng.standard_normal((3 * 224 * 224, 1024)).astype("float32") / 224.0

        def enc(batch: np.ndarray) -> np.ndarray:
            flat = batch.reshape(batch.shape[0], -1)
            return np.tanh(flat @ w)

        return enc

    if name in {"uni", "virchow", "conch"}:
        try:
            import timm
            import torch

            hub = {
                "uni": "hf-hub:MahmoodLab/UNI",
                "virchow": "hf-hub:paige-ai/Virchow",
                "conch": "hf-hub:MahmoodLab/CONCH",
            }[name]
            model = timm.create_model(hub, pretrained=True, num_classes=0).eval()

            @torch.no_grad()
            def enc(batch: np.ndarray) -> np.ndarray:
                t = torch.from_numpy(batch).float()
                return model(t).cpu().numpy()

            return enc
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível carregar o encoder {name!r}. Verifique `timm`, o "
                "acesso ao modelo no HuggingFace e a variável HF_TOKEN."
            ) from exc

    raise ValueError(f"encoder desconhecido: {name!r}")


# --------------------------------------------------------------------------- #
# Pipeline principal                                                           #
# --------------------------------------------------------------------------- #
def embed_wsi(
    slide_path: str | Path,
    out_path: str | Path,
    encoder: str = "uni",
    patch_px: int = 256,
    target_mpp: float = 0.5,
    max_patches: int = 4096,
    batch_size: int = 64,
) -> np.ndarray:
    """Extrai patches de tecido de uma WSI e salva os embeddings `(N, F)` em `.npy`."""
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    base_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
    scale = target_mpp / base_mpp
    read_px = round(patch_px * scale)

    thumb_ds = 32.0
    thumb = np.array(slide.get_thumbnail((int(slide.dimensions[0] / thumb_ds),
                                          int(slide.dimensions[1] / thumb_ds))))
    tissue = tissue_mask(thumb[..., :3])

    coords = list(iter_patch_coords(slide.dimensions, tissue, read_px, thumb_ds))[:max_patches]
    enc = load_encoder(encoder)

    feats = []
    for i in range(0, len(coords), batch_size):
        batch = []
        for x, y in coords[i : i + batch_size]:
            region = np.array(slide.read_region((x, y), 0, (read_px, read_px)))[..., :3]
            # resize para 224x224 (entrada padrão dos FMs) via PIL
            from PIL import Image

            img = np.asarray(Image.fromarray(region).resize((224, 224))).astype("float32") / 255.0
            batch.append(img.transpose(2, 0, 1))
        feats.append(enc(np.stack(batch)))
    emb = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1024), "float32")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb.astype("float32"))
    return emb


def _cli() -> None:
    p = argparse.ArgumentParser(description="WSI -> patches -> embeddings (.npy)")
    p.add_argument("--slide", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="uni")
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--mpp", type=float, default=0.5)
    p.add_argument("--max-patches", type=int, default=4096)
    a = p.parse_args()
    emb = embed_wsi(a.slide, a.out, a.encoder, a.patch, a.mpp, a.max_patches)
    print(f"{a.slide} -> {a.out}  ({emb.shape[0]} patches, F={emb.shape[1] if emb.size else 0})")


if __name__ == "__main__":
    _cli()
