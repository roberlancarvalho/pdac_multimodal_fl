"""MultimodalPDACDataset + data/preprocessing sobre arquivos falsos em tmp_path."""

from __future__ import annotations

import numpy as np

from data.dataset import ModalityShapes, MultimodalPDACDataset, collate_multimodal
from data.preprocessing import load_patch_embeddings, parse_mutations


def _fake_nifti(path, shape=(24, 24, 24)):
    import nibabel as nib

    nib.save(nib.Nifti1Image(np.random.rand(*shape).astype("float32") * 400 - 100, np.eye(4)), str(path))


def _write_maf(path):
    path.write_text(
        "Hugo_Symbol\tVariant_Classification\tt_alt_count\tt_depth\n"
        "KRAS\tMissense_Mutation\t30\t100\n"
        "TP53\tNonsense_Mutation\t45\t90\n",
        encoding="utf-8",
    )


def test_parse_mutations_maf(tmp_path):
    maf = tmp_path / "p.maf"
    _write_maf(maf)
    parsed = parse_mutations(maf, ("KRAS", "TP53", "SMAD4", "CDKN2A"))
    assert parsed["KRAS"] == (1, 1, 0.3)          # missense, VAF 30/100
    assert parsed["TP53"][:2] == (1, 2)           # nonsense
    assert parsed["SMAD4"] == (0, 0, 0.0)         # ausente -> wild-type


def test_parse_mutations_simple_csv(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("gene,status,variant_type,vaf\nKRAS,1,1,0.42\nSMAD4,1,3,0.2\n", encoding="utf-8")
    parsed = parse_mutations(csv, ("KRAS", "TP53", "SMAD4", "CDKN2A"))
    assert parsed["KRAS"] == (1, 1, 0.42)
    assert parsed["SMAD4"] == (1, 3, 0.2)


def test_load_patch_embeddings_pad_and_truncate(tmp_path):
    np.save(tmp_path / "emb.npy", np.random.rand(5, 8).astype("float32"))
    emb, mask = load_patch_embeddings(tmp_path / "emb.npy", n_patches=10, feat_dim=8)
    assert emb.shape == (10, 8)
    assert mask.sum() == 5 and mask[:5].all() and not mask[5:].any()


def test_dataset_from_manifest(tmp_path):
    (tmp_path / "ct").mkdir()
    (tmp_path / "wsi").mkdir()
    (tmp_path / "gen").mkdir()
    _fake_nifti(tmp_path / "ct" / "p1_ap.nii.gz")
    _fake_nifti(tmp_path / "ct" / "p1_vp.nii.gz")
    np.save(tmp_path / "wsi" / "p1.npy", np.random.rand(20, 16).astype("float32"))
    _write_maf(tmp_path / "gen" / "p1.maf")
    _fake_nifti(tmp_path / "ct" / "p2_ap.nii.gz")
    _fake_nifti(tmp_path / "ct" / "p2_vp.nii.gz")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "patient_id,ct_ap,ct_vp,wsi_emb,mutations,time,event,dx,subtype,split\n"
        "p1,ct/p1_ap.nii.gz,ct/p1_vp.nii.gz,wsi/p1.npy,gen/p1.maf,12.0,1,1,0,train\n"
        "p2,ct/p2_ap.nii.gz,ct/p2_vp.nii.gz,,,30.0,0,,,train\n",
        encoding="utf-8",
    )

    from data.preprocessing import ct_transforms

    shapes = ModalityShapes(n_patches=16, patch_feat_dim=16)
    ds = MultimodalPDACDataset(
        manifest, data_root=tmp_path, shapes=shapes, split="train",
        ct_transform=ct_transforms(train=False, roi=(32, 32, 32)),
    )
    assert len(ds) == 2

    p1 = ds[0]
    assert p1["ct_volume"].shape == (2, 32, 32, 32)   # AP + VP
    assert p1["patch_embeddings"].shape == (16, 16)
    assert p1["mutation_status"].tolist() == [1, 1, 0, 0]
    assert p1["dx"].item() == 1.0 and p1["subtype"].item() == 0

    p2 = ds[1]
    assert "patch_embeddings" not in p2 and "mutation_status" not in p2
    assert p2["dx"].item() == -1.0 and p2["subtype"].item() == -1

    b = collate_multimodal([p1, p2])
    assert b["ct_volume"].shape == (2, 2, 32, 32, 32)
    # ordem: radiomics, histology, genomics, clinical
    assert b["modality_mask"].tolist() == [[True, True, True, False], [True, False, False, False]]
