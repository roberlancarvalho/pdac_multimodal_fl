"""
Modelo completo -- Pipeline Multimodal Federado para PDAC.

Orquestra os três ramos independentes e o módulo de fusão:

    TC 3D  ---> RadiomicsBranch3D --\
    WSI    ---> HistologyBranch  ----> Fusão ---> risco/sobrevida
    Genes  ---> GenomicsBranch  ---/

`fusion_mode`:
    - "coattention" (padrão): co-atenção cross-modal par-a-par direcional
      (`CrossModalCoAttentionFusion`, estilo MCAT, Seção 6.1 do artigo). Os ramos
      emitem sequências de tokens.
    - "transformer" (legado): auto-atenção conjunta sobre 1 token por modalidade
      (`CrossModalAttentionFusion`).

É esta a `nn.Module` cujos parâmetros o Flower serializa/agrega no Aprendizado
Federado (ver `federated/client.py`). Cada ramo pode ser congelado
individualmente (`freeze_*`) para treinar apenas a fusão -- estratégia comum
quando os ramos usam backbones/foundation models pré-treinados.

Entrada (forward): dict `batch` com as chaves presentes para cada paciente:
    "ct_volume":        Tensor (B, C, D, H, W)         -- Ramo A  (C=1 ou 2 p/ AP+VP) (opcional)
    "patch_embeddings": Tensor (B, N, patch_feat_dim)  -- Ramo B  (opcional)
    "patch_mask":       Tensor bool (B, N)             -- Ramo B  (opcional)
    "mutation_status":  Tensor long (B, 4)             -- Ramo C  (opcional)
    "variant_type":     Tensor long (B, 4)             -- Ramo C  (opcional)
    "vaf":              Tensor float (B, 4)            -- Ramo C  (opcional)
    "clinical_num":     Tensor float (B, n_cont)       -- Ramo D  (opcional)
    "clinical_cat":     Tensor long (B, n_cat)         -- Ramo D  (opcional)
    "modality_mask":    Tensor bool (B, 4)             -- forçar modalidade ausente (opcional)

Saída (forward): dict com as três tarefas clínicas da Figura 4 do artigo
    "risk":           Tensor (B, n_outputs) -- log-hazard (prognóstico de sobrevida, Cox)
    "dx_logit":       Tensor (B, 1)         -- diagnóstico (PDAC vs não-PDAC), se habilitado
    "subtype_logits": Tensor (B, n_subtypes) -- subtipagem molecular (classical/basal-like), se habilitado
    "fused":          Tensor (B, embed_dim)
    "attention_histology": Tensor (B, N) -- pesos de atenção dos patches (se Ramo B usado)
"""

from __future__ import annotations

import torch
from torch import nn

from models.branch_a_radiomics import RadiomicsBranch3D
from models.branch_b_histology import HistologyBranch
from models.branch_c_genomics import GenomicsBranch
from models.branch_d_clinical import ClinicalBranch
from models.fusion_attention import MODALITIES, CrossModalAttentionFusion
from models.fusion_coattention import CrossModalCoAttentionFusion


def _mlp_head(embed_dim: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(embed_dim // 2, out_dim),
    )


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
        ct_in_channels: int = 2,
        radiomics_phases: int = 2,
        patch_feat_dim: int = 1024,
        n_genes: int = 4,
        n_gene_states: int = 3,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.2,
        n_outputs: int = 1,
        fusion_mode: str = "coattention",
        radiomics_token_grid: tuple[int, int, int] = (2, 2, 2),
        histology_tokens: int = 8,
        fusion_genomics_query_only: bool = False,
        fusion_aux_heads: bool = True,
        genomics_use_variant_type: bool = True,
        genomics_use_vaf: bool = True,
        enable_clinical: bool = True,
        clinical_n_continuous: int = 5,
        clinical_cat_cardinalities: tuple[int, ...] = (2, 4, 5, 3),
        enable_diagnosis: bool = True,
        enable_subtype: bool = True,
        n_subtypes: int = 2,
        freeze_radiomics: bool = False,
        freeze_histology: bool = False,
        freeze_genomics: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode
        self._coattn = fusion_mode == "coattention"

        self.branch_a = RadiomicsBranch3D(
            in_channels=ct_in_channels, embed_dim=embed_dim,
            backbone_dropout=dropout, freeze_backbone=freeze_radiomics,
            token_grid=radiomics_token_grid, n_phases=radiomics_phases,
        )
        self.branch_b = HistologyBranch(
            input_feat_dim=patch_feat_dim, embed_dim=embed_dim, dropout=dropout,
            n_output_tokens=histology_tokens if self._coattn else 1,
        )
        self.branch_c = GenomicsBranch(
            n_genes=n_genes, n_states=n_gene_states, embed_dim=embed_dim, dropout=dropout,
            use_variant_type=genomics_use_variant_type, use_vaf=genomics_use_vaf,
        )
        self.enable_clinical = enable_clinical
        if enable_clinical:
            self.branch_d = ClinicalBranch(
                n_continuous=clinical_n_continuous,
                cat_cardinalities=tuple(clinical_cat_cardinalities),
                embed_dim=embed_dim, dropout=dropout,
            )

        if self._coattn:
            self.fusion = CrossModalCoAttentionFusion(
                embed_dim=embed_dim, n_layers=fusion_layers, n_heads=fusion_heads,
                dropout=dropout, n_outputs=n_outputs,
                genomics_query_only=fusion_genomics_query_only,
                aux_heads=fusion_aux_heads,
            )
        else:
            self.fusion = CrossModalAttentionFusion(
                embed_dim=embed_dim, n_layers=fusion_layers, n_heads=fusion_heads,
                dropout=dropout, n_outputs=n_outputs,
            )

        # Cabeças clínicas adicionais (Figura 4): diagnóstico e subtipagem molecular.
        self.enable_diagnosis = enable_diagnosis
        self.enable_subtype = enable_subtype
        if enable_diagnosis:
            self.dx_head = _mlp_head(embed_dim, 1, dropout)
        if enable_subtype:
            self.subtype_head = _mlp_head(embed_dim, n_subtypes, dropout)

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
        reps: dict[str, torch.Tensor] = {}
        aux: dict[str, torch.Tensor] = {}
        rt = self._coattn

        if batch.get("ct_volume") is not None:
            reps["radiomics"] = self.branch_a(batch["ct_volume"].to(device), return_tokens=rt)

        if batch.get("patch_embeddings") is not None:
            patch_mask = batch.get("patch_mask")
            hist, patch_attn = self.branch_b(
                batch["patch_embeddings"].to(device),
                mask=patch_mask.to(device) if patch_mask is not None else None,
                return_tokens=rt,
            )
            reps["histology"] = hist
            aux["attention_histology"] = patch_attn

        if batch.get("mutation_status") is not None:
            vtype = batch.get("variant_type")
            vaf = batch.get("vaf")
            reps["genomics"] = self.branch_c(
                batch["mutation_status"].to(device),
                variant_type=vtype.to(device) if vtype is not None else None,
                vaf=vaf.to(device) if vaf is not None else None,
                return_tokens=rt,
            )

        if self.enable_clinical and batch.get("clinical_num") is not None:
            reps["clinical"] = self.branch_d(
                batch["clinical_num"].to(device),
                batch["clinical_cat"].to(device),
                return_tokens=rt,
            )

        if not reps:
            raise ValueError("Nenhuma modalidade fornecida ao MultimodalPDACModel.forward().")

        if self._coattn:
            present = self._present_matrix(batch.get("modality_mask"), reps, device)
            out = self.fusion(reps, present=present)
        else:
            out = self.fusion(reps, mask=batch.get("modality_mask"))
        out.update(aux)

        if self.enable_diagnosis:
            out["dx_logit"] = self.dx_head(out["fused"])           # (B, 1)
        if self.enable_subtype:
            out["subtype_logits"] = self.subtype_head(out["fused"])  # (B, n_subtypes)
        return out

    def _present_matrix(self, modality_mask, reps: dict, device) -> torch.Tensor:
        """(B, len(MODALITIES)) bool -- presença por amostra de cada modalidade."""
        b = next(iter(reps.values())).size(0)
        mm = modality_mask.to(device) if modality_mask is not None else None
        cols = []
        for i, name in enumerate(MODALITIES):
            if name not in reps:
                cols.append(torch.zeros(b, dtype=torch.bool, device=device))
            elif mm is not None and i < mm.shape[1]:
                cols.append(mm[:, i].bool())
            else:
                cols.append(torch.ones(b, dtype=torch.bool, device=device))
        return torch.stack(cols, dim=1)

    # ---- Helpers para o Flower (serialização de parâmetros) -------------------
    def get_trainable_parameter_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]


if __name__ == "__main__":
    batch = {
        "ct_volume": torch.randn(2, 2, 48, 64, 64),  # fases AP + VP
        "patch_embeddings": torch.randn(2, 200, 1024),
        "patch_mask": torch.ones(2, 200, dtype=torch.bool),
        "mutation_status": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 1]]),
        "variant_type": torch.tensor([[1, 3, 0, 0], [2, 0, 0, 4]]),
        "vaf": torch.tensor([[0.4, 0.7, 0.0, 0.0], [0.3, 0.0, 0.0, 0.6]]),
        "clinical_num": torch.randn(2, 5),
        "clinical_cat": torch.tensor([[0, 1, 2, 0], [1, 3, 4, 2]]),
        "modality_mask": torch.tensor([[True, True, True, True], [True, False, True, True]]),
    }
    for mode in ("coattention", "transformer"):
        model = MultimodalPDACModel(embed_dim=128, patch_feat_dim=1024, fusion_mode=mode)
        out = model(batch)
        shapes = {k: (tuple(v.shape) if torch.is_tensor(v) else "dict") for k, v in out.items()}
        print(f"[{mode}]", shapes)
    print("Modalidades:", MODALITIES)
