"""
Ramo C -- Genômico Tabular (status mutacional + tipo de variante + VAF).

Ramo mais leve do pipeline. Recebe, para os quatro drivers canônicos do PDAC
(KRAS, TP53, SMAD4, CDKN2A):
    - status mutacional     (wt / mutado / desconhecido)
    - tipo de variante      (missense, nonsense, frameshift, splice, outro)
    - frequência alélica    (VAF ∈ [0, 1])
e produz uma representação na dimensão compartilhada pelos ramos.

Cada gene é codificado por embeddings aprendíveis (status e tipo de variante) +
projeção da VAF; a soma vira o vetor do gene.

Codificação de `mutation_status` por gene:
    0 = wild-type · 1 = mutado · 2 = desconhecido / não avaliado
Codificação de `variant_type` por gene:
    0 = nenhuma/NA · 1 = missense · 2 = nonsense · 3 = frameshift · 4 = splice · 5 = outra

Entrada (forward):
    mutation_status: Tensor long (B, n_genes) -- obrigatório.
    variant_type:    Tensor long (B, n_genes) -- opcional (default = zeros).
    vaf:             Tensor float (B, n_genes) em [0, 1] -- opcional (default = zeros).
Saída  (forward):
    - `return_tokens=False`: Tensor (B, embed_dim) -- embedding por paciente.
    - `return_tokens=True` : Tensor (B, n_genes, embed_dim) -- um token por gene driver.
"""

from __future__ import annotations

import torch
from torch import nn

PDAC_DRIVER_GENES: tuple[str, ...] = ("KRAS", "TP53", "SMAD4", "CDKN2A")
VARIANT_TYPES: tuple[str, ...] = ("NA", "missense", "nonsense", "frameshift", "splice", "outra")


class GenomicsBranch(nn.Module):
    """Encoder tabular para o perfil mutacional dos drivers do PDAC.

    Args:
        n_genes: Número de genes de entrada (padrão 4).
        n_states: Estados de `mutation_status` por gene (2 = wt/mut; 3 inclui "desconhecido").
        n_variant_types: Categorias de `variant_type` por gene.
        gene_embed_dim: Dimensão do embedding por gene.
        embed_dim: Dimensão de saída, compartilhada entre os ramos.
        hidden_dim: Largura da MLP interna.
        dropout: Taxa de dropout.
        use_variant_type / use_vaf: Habilita cada sinal extra.
    """

    def __init__(
        self,
        n_genes: int = 4,
        n_states: int = 3,
        n_variant_types: int = 6,
        gene_embed_dim: int = 32,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_variant_type: bool = True,
        use_vaf: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim
        self.n_states = n_states
        self.n_variant_types = n_variant_types
        self.use_variant_type = use_variant_type
        self.use_vaf = use_vaf

        # Um espaço de embedding por (gene, estado). O offset por gene garante que
        # "KRAS mutado" e "TP53 mutado" tenham vetores distintos.
        self.gene_state_embed = nn.Embedding(n_genes * n_states, gene_embed_dim)
        self.register_buffer("gene_offsets", torch.arange(n_genes) * n_states, persistent=False)

        if use_variant_type:
            self.variant_embed = nn.Embedding(n_genes * n_variant_types, gene_embed_dim)
            self.register_buffer(
                "variant_offsets", torch.arange(n_genes) * n_variant_types, persistent=False
            )
        if use_vaf:
            self.vaf_proj = nn.Linear(1, gene_embed_dim)

        # Projeção de cada gene para um token na dimensão compartilhada (co-atenção).
        self.token_proj = nn.Sequential(
            nn.Linear(gene_embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_genes * gene_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def _gene_vectors(
        self,
        mutation_status: torch.Tensor,
        variant_type: torch.Tensor | None,
        vaf: torch.Tensor | None,
    ) -> torch.Tensor:
        """(B, n_genes, gene_embed_dim) -- soma dos sinais por gene."""
        gene_vecs = self.gene_state_embed(mutation_status.long() + self.gene_offsets)

        if self.use_variant_type:
            if variant_type is None:
                variant_type = torch.zeros_like(mutation_status)
            gene_vecs = gene_vecs + self.variant_embed(
                variant_type.long().clamp(0, self.n_variant_types - 1) + self.variant_offsets
            )

        if self.use_vaf:
            if vaf is None:
                vaf = torch.zeros_like(mutation_status, dtype=gene_vecs.dtype)
            gene_vecs = gene_vecs + self.vaf_proj(vaf.to(gene_vecs.dtype).unsqueeze(-1))

        return gene_vecs

    def forward(
        self,
        mutation_status: torch.Tensor,
        variant_type: torch.Tensor | None = None,
        vaf: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        """Gera a representação genômica.

        Returns:
            (B, embed_dim) se `return_tokens=False`; (B, n_genes, embed_dim) caso contrário.
        """
        gene_vecs = self._gene_vectors(mutation_status, variant_type, vaf)

        if return_tokens:
            return self.token_proj(gene_vecs)          # (B, n_genes, embed_dim)

        flat = gene_vecs.flatten(start_dim=1)          # (B, n_genes * gene_embed_dim)
        return self.mlp(flat)                          # (B, embed_dim)


if __name__ == "__main__":
    model = GenomicsBranch(embed_dim=256)
    mut = torch.tensor([[1, 1, 0, 0], [1, 0, 2, 0]])
    vtype = torch.tensor([[1, 3, 0, 0], [1, 0, 0, 0]])
    vaf = torch.tensor([[0.4, 0.7, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]])
    print("Ramo C -- vetor:", model(mut, vtype, vaf).shape)                     # (2, 256)
    print("Ramo C -- tokens:", model(mut, vtype, vaf, return_tokens=True).shape)  # (2, 4, 256)
    print("Ramo C -- só status:", model(mut).shape)                             # (2, 256)
