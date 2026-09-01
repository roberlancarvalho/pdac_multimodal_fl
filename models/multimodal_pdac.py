"""
Modelo completo -- Pipeline Multimodal Federado para PDAC.

Orquestra os três ramos independentes e o módulo de fusão por atenção cruzada:

    TC 3D  ---> RadiomicsBranch3D --\
    WSI    ---> HistologyBranch  ----> CrossModalAttentionFusion ---> risco/sobrevida
    Genes  ---> GenomicsBranch  ---/

É esta a `nn.Module` cujos parâmetros o Flower serializa/agrega no Aprendizado
Federado (ver `federated/client.py`). Cada ramo pode ser congelado
individualmente (`freeze_*`) para treinar apenas a fusão -- estratégia comum
quando os ramos usam backbones/foundation models pré-treinados.

Entrada (forward): dict `batch` com as chaves presentes para cada paciente:
    "ct_volume":        Tensor (B, 1, D, H, W)         -- Ramo A  (opcional)
    "patch_embeddings": Tensor (B, N, patch_feat_dim)  -- Ramo B  (opcional)
    "patch_mask":       Tensor bool (B, N)             -- Ramo B  (opcional)
    "mutation_status":  Tensor long (B, 4)             -- Ramo C  (opcional)
    "modality_mask":    Tensor bool (B, 3)             -- forçar modalidade ausente (opcional)

Saída (forward): dict com
    "risk":  Tensor (B, n_outputs)   -- log-hazard para perda de Cox
    "fused": Tensor (B, embed_dim)
    "attention_histology": Tensor (B, N) -- pesos de atenção dos patches (se Ramo B usado)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.branch_a_radiomics import RadiomicsBranch3D
from models.branch_b_histology import HistologyBranch
from models.branch_c_genomics import GenomicsBranch
from models.fusion_attention import MODALITIES, CrossModalAttentionFusion


class MultimodalPDACModel(nn.Module):
    """Modelo multimodal federado para predição de risco em PDAC.

    Args:
        embed_dim: Dimensão compartilhada dos embeddings dos três ramos.
        ct_in_channels: Canais do volume de TC (Ramo A).
        patch_feat_dim: Dimensão do embedding de patch do foundation model (Ramo B).
        n_genes: Nº de genes driver (Ramo C).
        n_gene_states: Estados por gene (2 = wt/mut, 3 inclui desconhecido).
        fusion_layers / fusion_heads: Profundidade/cabeças do Transformer de fusão.
        n_outputs: Saídas da cabeça de risco (1 = Cox).
        freeze_radiomics / freeze_histology / freeze_genomics: Congela o ramo.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        ct_in_channels: int = 1,
        patch_feat_dim: int = 1024,
        n_genes: int = 4,
        n_gene_states: int = 3,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.2,
        n_outputs: int = 1,
        freeze_radiomics: bool = False,
        freeze_histology: bool = False,
        freeze_genomics: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.branch_a = RadiomicsBranch3D(
            in_channels=ct_in_channels, embed_dim=embed_dim,
            backbone_dropout=dropout, freeze_backbone=freeze_radiomics,
        )
        self.branch_b = HistologyBranch(
            input_feat_dim=patch_feat_dim, embed_dim=embed_dim, dropout=dropout,
        )
        self.branch_c = GenomicsBranch(
            n_genes=n_genes, n_states=n_gene_states, embed_dim=embed_dim, dropout=dropout,
        )
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim, n_layers=fusion_layers, n_heads=fusion_heads,
            dropout=dropout, n_outputs=n_outputs,
        )

        if freeze_histology:
            self._freeze(self.branch_b)
        if freeze_genomics:
            self._freeze(self.branch_c)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        embeddings: dict[str, torch.Tensor] = {}
        aux: dict[str, torch.Tensor] = {}

        if batch.get("ct_volume") is not None:
            embeddings["radiomics"] = self.branch_a(batch["ct_volume"].to(device))

        if batch.get("patch_embeddings") is not None:
            slide_emb, patch_attn = self.branch_b(
                batch["patch_embeddings"].to(device),
                mask=batch["patch_mask"].to(device) if batch.get("patch_mask") is not None else None,
            )
            embeddings["histology"] = slide_emb
            aux["attention_histology"] = patch_attn

        if batch.get("mutation_status") is not None:
            embeddings["genomics"] = self.branch_c(batch["mutation_status"].to(device))

        if not embeddings:
            raise ValueError("Nenhuma modalidade fornecida ao MultimodalPDACModel.forward().")

        out = self.fusion(embeddings, mask=batch.get("modality_mask"))
        out.update(aux)
        return out

    # ---- Helpers para o Flower (serialização de parâmetros) -------------------
    def get_trainable_parameter_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]


if __name__ == "__main__":
    model = MultimodalPDACModel(embed_dim=128, patch_feat_dim=1024)
    batch = {
        "ct_volume": torch.randn(2, 1, 48, 64, 64),
        "patch_embeddings": torch.randn(2, 200, 1024),
        "patch_mask": torch.ones(2, 200, dtype=torch.bool),
        "mutation_status": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 1]]),
    }
    out = model(batch)
    print({k: tuple(v.shape) for k, v in out.items()})
    print("Modalidades:", MODALITIES)
