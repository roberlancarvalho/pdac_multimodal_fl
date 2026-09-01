"""Fixtures compartilhadas -- configuração mínima e lotes sintéticos rápidos."""

from __future__ import annotations

import pytest
import torch

EMBED_DIM = 16


@pytest.fixture(scope="session")
def model_cfg() -> dict:
    """Config de modelo pequena o suficiente para os testes rodarem em segundos."""
    return {
        "embed_dim": EMBED_DIM,
        "ct_in_channels": 2,
        "radiomics_phases": 2,
        "patch_feat_dim": 32,
        "n_genes": 4,
        "n_gene_states": 3,
        "fusion_layers": 1,
        "fusion_heads": 2,
        "dropout": 0.0,
        "n_outputs": 1,
        "fusion_mode": "coattention",
        "radiomics_token_grid": [2, 2, 2],
        "histology_tokens": 4,
        "fusion_aux_heads": True,
        "genomics_use_variant_type": True,
        "genomics_use_vaf": True,
        "enable_clinical": True,
        "clinical_n_continuous": 5,
        "clinical_cat_cardinalities": [2, 4, 5, 3],
        "enable_diagnosis": True,
        "enable_subtype": True,
        "n_subtypes": 2,
    }


@pytest.fixture
def batch() -> dict:
    """Lote multimodal sintético (B=4) com as quatro modalidades e os rótulos."""
    g = torch.Generator().manual_seed(0)
    b = 4
    return {
        # 32 é o menor lado que sobrevive aos 5 downsamplings do DenseNet121-3D.
        "ct_volume": torch.randn(b, 2, 32, 32, 32, generator=g),
        "patch_embeddings": torch.randn(b, 10, 32, generator=g),
        "patch_mask": torch.ones(b, 10, dtype=torch.bool),
        "mutation_status": torch.tensor([[1, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),
        "variant_type": torch.tensor([[1, 3, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]]),
        "vaf": torch.tensor(
            [[0.4, 0.7, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
        ),
        "clinical_num": torch.randn(b, 5, generator=g),
        "clinical_cat": torch.tensor([[0, 1, 2, 0], [1, 3, 4, 2], [0, 0, 0, 1], [1, 2, 3, 0]]),
        "modality_mask": torch.tensor(
            [
                [True, True, True, True],
                [True, False, True, True],
                [False, True, True, False],
                [True, True, True, True],
            ]
        ),
        "time": torch.tensor([12.0, 30.0, 5.0, 44.0]),
        "event": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "dx": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "subtype": torch.tensor([0, 1, -1, 1]),
    }
