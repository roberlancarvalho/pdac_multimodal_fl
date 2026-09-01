import torch

from models.fusion_attention import CrossModalAttentionFusion
from models.fusion_coattention import CrossModalCoAttentionFusion

D = 16


def _tokens(b=3):
    return {
        "radiomics": torch.randn(b, 8, D),
        "histology": torch.randn(b, 4, D),
        "genomics": torch.randn(b, 4, D),
    }


def test_coattention_outputs():
    fusion = CrossModalCoAttentionFusion(embed_dim=D, n_layers=1, n_heads=2, aux_heads=True)
    out = fusion(_tokens())
    assert out["risk"].shape == (3, 1)
    assert out["fused"].shape == (3, D)
    assert set(out["modality_gate"]) == {"radiomics", "histology", "genomics"}
    assert set(out["aux_risk"]) == {"radiomics", "histology", "genomics"}


def test_coattention_per_sample_missing_modality_no_nan():
    fusion = CrossModalCoAttentionFusion(embed_dim=D, n_layers=1, n_heads=2)
    present = torch.tensor([[True, True, True], [True, False, True], [False, False, True]])
    out = fusion(_tokens(), present=present)
    assert torch.isfinite(out["risk"]).all()
    assert torch.isfinite(out["fused"]).all()


def test_coattention_single_modality():
    fusion = CrossModalCoAttentionFusion(embed_dim=D, n_layers=1, n_heads=2)
    out = fusion({"genomics": torch.randn(2, 4, D)})
    assert out["risk"].shape == (2, 1)
    assert torch.isfinite(out["risk"]).all()


def test_legacy_transformer_fusion():
    fusion = CrossModalAttentionFusion(embed_dim=D, n_layers=1, n_heads=2)
    emb = {m: torch.randn(2, D) for m in ("radiomics", "histology", "genomics")}
    out = fusion(emb)
    assert out["risk"].shape == (2, 1)
    out_missing = fusion({"radiomics": torch.randn(2, D)})
    assert torch.isfinite(out_missing["risk"]).all()
