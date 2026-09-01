import torch

from data.dataset import ModalityShapes, SyntheticPDACDataset, collate_multimodal
from models.fusion_attention import MODALITIES

_MODALITY_KEYS = ("ct_volume", "patch_embeddings", "mutation_status", "clinical_num")


def test_synthetic_item_keys_and_labels():
    ds = SyntheticPDACDataset(n_samples=8, seed=0)
    item = ds[0]
    assert {"time", "event", "dx", "subtype", "patient_id"} <= set(item)
    if "mutation_status" in item:
        assert item["mutation_status"].shape == (4,)
        assert item["variant_type"].shape == (4,)
        assert item["vaf"].shape == (4,)
        # variant/vaf só nos genes mutados
        wt = item["mutation_status"] != 1
        assert torch.all(item["variant_type"][wt] == 0)
        assert torch.all(item["vaf"][wt] == 0)


def test_modality_dropout_keeps_at_least_one():
    ds = SyntheticPDACDataset(n_samples=64, modality_dropout=0.9, seed=1)
    for i in range(len(ds)):
        item = ds[i]
        assert any(k in item for k in _MODALITY_KEYS)


def test_collate_shapes_and_modality_mask():
    shapes = ModalityShapes(ct_shape=(32, 32, 32), n_patches=6, patch_feat_dim=16, ct_channels=2)
    ds = SyntheticPDACDataset(n_samples=8, shapes=shapes, seed=0)
    b = collate_multimodal([ds[i] for i in range(4)])
    assert b["modality_mask"].shape == (4, len(MODALITIES))
    assert b["time"].shape == (4,) and b["dx"].shape == (4,)
    if "ct_volume" in b:
        assert b["ct_volume"].shape[1] == 2
