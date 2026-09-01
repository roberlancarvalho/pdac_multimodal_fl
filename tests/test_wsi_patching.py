"""data/wsi_patching -- detecção de tecido, geração de coords e encoder 'random'."""

import numpy as np

from data.wsi_patching import iter_patch_coords, load_encoder, tissue_mask


def test_tissue_mask_flags_colored_regions():
    thumb = np.full((40, 40, 3), 245, dtype=np.uint8)  # fundo branco
    thumb[10:30, 10:30] = (150, 60, 120)               # bloco "tecido"
    m = tissue_mask(thumb)
    assert m[20, 20] and not m[2, 2]
    assert 0.1 < m.mean() < 0.5


def test_iter_patch_coords_respects_tissue_and_bounds():
    tissue = np.zeros((32, 32), dtype=bool)
    tissue[8:24, 8:24] = True
    coords = list(iter_patch_coords((1024, 1024), tissue, patch_px=256, thumb_downsample=32.0))
    assert coords and all(0 <= x <= 768 and 0 <= y <= 768 for x, y in coords)


def test_random_encoder_shapes():
    enc = load_encoder("random")
    out = enc(np.random.rand(5, 3, 224, 224).astype("float32"))
    assert out.shape == (5, 1024)
