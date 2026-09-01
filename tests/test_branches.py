import torch

from models.branch_a_radiomics import RadiomicsBranch3D
from models.branch_b_histology import HistologyBranch
from models.branch_c_genomics import GenomicsBranch
from models.branch_d_clinical import ClinicalBranch

D = 16


def test_radiomics_shared_encoder_phases_and_tokens():
    m = RadiomicsBranch3D(in_channels=2, embed_dim=D, n_phases=2, token_grid=(2, 2, 2))
    x = torch.randn(2, 2, 32, 32, 32)
    assert m(x).shape == (2, D)
    tokens = m(x, return_tokens=True)
    assert tokens.shape == (2, 2 * 8, D)  # n_phases * (2*2*2)


def test_histology_multitoken_and_empty_bag():
    m = HistologyBranch(input_feat_dim=32, embed_dim=D, n_output_tokens=4, n_transformer_layers=1)
    bag = torch.randn(2, 12, 32)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1] = False  # bag vazia
    tokens, attn = m(bag, mask=mask, return_tokens=True)
    assert tokens.shape == (2, 4, D)
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens[1]) == 0  # bag vazia -> tokens zerados
    emb, _ = m(bag, mask=mask)
    assert emb.shape == (2, D)


def test_genomics_status_variant_vaf():
    m = GenomicsBranch(embed_dim=D, use_variant_type=True, use_vaf=True)
    mut = torch.tensor([[1, 1, 0, 0], [1, 0, 2, 0]])
    vtype = torch.tensor([[1, 3, 0, 0], [2, 0, 0, 0]])
    vaf = torch.tensor([[0.4, 0.7, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]])
    assert m(mut, vtype, vaf).shape == (2, D)
    assert m(mut, vtype, vaf, return_tokens=True).shape == (2, 4, D)
    assert m(mut).shape == (2, D)  # variant/vaf opcionais


def test_clinical_branch_continuous_and_categorical():
    m = ClinicalBranch(n_continuous=5, cat_cardinalities=(2, 4, 5, 3), embed_dim=D)
    num = torch.randn(3, 5)
    cat = torch.tensor([[0, 2, 1, 0], [1, 3, 4, 2], [0, 0, 0, 1]])
    assert m(num, cat).shape == (3, D)
    assert m(num, cat, return_tokens=True).shape == (3, 5 + 4, D)  # 1 token por campo
    # categórico fora do intervalo não deve estourar
    assert torch.isfinite(m(num, torch.tensor([[9, 9, 9, 9]] * 3))).all()
