"""
Ramo D -- Clínico Tabular.

Variáveis clínicas de rotina que vários estudos da revisão mostram serem
prognósticas (ex.: idade, CA 19-9, estágio AJCC, ECOG PS, ressecabilidade,
localização). Não está na especificação estrita da Seção 6.1 (que é Radiômica +
Histologia + Genômica), mas aparece na proposta geral da Seção 6 e nos modelos
"Patologia+Clínica" (C-index 0,86) e "RSF clínico" da revisão.

Entrada (forward):
    clinical_num: Tensor float (B, n_continuous) -- já padronizado (z-score) offline.
    clinical_cat: Tensor long  (B, n_categorical) -- códigos inteiros por campo.
Saída (forward):
    - `return_tokens=False`: (B, embed_dim)
    - `return_tokens=True`:  (B, n_continuous + n_categorical, embed_dim) -- 1 token/campo.
"""

from __future__ import annotations

import torch
from torch import nn


class ClinicalBranch(nn.Module):
    """Encoder tabular para variáveis clínicas (contínuas + categóricas)."""

    def __init__(
        self,
        n_continuous: int = 5,
        cat_cardinalities: tuple[int, ...] = (2, 4, 5, 3),
        field_dim: int = 32,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.cat_cardinalities = tuple(cat_cardinalities)
        self.n_fields = n_continuous + len(self.cat_cardinalities)
        self.embed_dim = embed_dim

        # Contínuas: projeção escalar compartilhada + embedding de identidade do campo.
        self.cont_proj = nn.Linear(1, field_dim)
        self.cont_field_id = nn.Embedding(max(n_continuous, 1), field_dim)
        # Categóricas: um embedding por campo (+1 índice reservado para "desconhecido").
        self.cat_embed = nn.ModuleList(
            [nn.Embedding(card + 1, field_dim) for card in self.cat_cardinalities]
        )

        self.token_proj = nn.Sequential(
            nn.Linear(field_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.n_fields * field_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def _field_vectors(self, clinical_num: torch.Tensor, clinical_cat: torch.Tensor) -> torch.Tensor:
        b = clinical_num.size(0) if self.n_continuous else clinical_cat.size(0)
        parts = []
        if self.n_continuous:
            idx = torch.arange(self.n_continuous, device=clinical_num.device)
            cont = self.cont_proj(clinical_num.unsqueeze(-1).float())  # (B, n_cont, field_dim)
            parts.append(cont + self.cont_field_id(idx).unsqueeze(0))
        for i, emb in enumerate(self.cat_embed):
            col = clinical_cat[:, i].long().clamp(0, self.cat_cardinalities[i])
            parts.append(emb(col).unsqueeze(1))  # (B, 1, field_dim)
        return torch.cat(parts, dim=1) if parts else torch.zeros(b, 0, 1)

    def forward(
        self,
        clinical_num: torch.Tensor,
        clinical_cat: torch.Tensor,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        fields = self._field_vectors(clinical_num, clinical_cat)  # (B, n_fields, field_dim)
        if return_tokens:
            return self.token_proj(fields)                        # (B, n_fields, embed_dim)
        return self.mlp(fields.flatten(start_dim=1))              # (B, embed_dim)


if __name__ == "__main__":
    m = ClinicalBranch(n_continuous=5, cat_cardinalities=(2, 4, 5, 3), embed_dim=64)
    num = torch.randn(3, 5)
    cat = torch.tensor([[0, 2, 1, 0], [1, 3, 4, 2], [0, 0, 0, 1]])
    print("Ramo D -- vetor:", m(num, cat).shape)                       # (3, 64)
    print("Ramo D -- tokens:", m(num, cat, return_tokens=True).shape)  # (3, 9, 64)
