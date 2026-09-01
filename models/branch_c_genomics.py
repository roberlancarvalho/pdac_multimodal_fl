"""
Ramo C -- Genômico Tabular (status mutacional).

Ramo mais leve do pipeline. Recebe o status mutacional (mutado / wild-type) dos
quatro drivers canônicos do PDAC -- KRAS, TP53, SMAD4, CDKN2A -- e produz um
embedding na dimensão compartilhada pelos ramos.

Cada gene é codificado por um embedding aprendível (nn.Embedding) em vez de um
simples 0/1, o que permite ao modelo representar melhor o efeito de cada driver
e lida naturalmente com um estado "desconhecido/não sequenciado" (índice 2).

Codificação de entrada por gene:
    0 = wild-type
    1 = mutado
    2 = desconhecido / não avaliado  (opcional; útil em coortes federadas heterogêneas)

Entrada (forward): Tensor long de shape (B, 4)
    -- colunas na ordem: [KRAS, TP53, SMAD4, CDKN2A].
Saída  (forward): Tensor float32 de shape (B, embed_dim)
    -- embedding genômico por paciente.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PDAC_DRIVER_GENES: tuple[str, ...] = ("KRAS", "TP53", "SMAD4", "CDKN2A")


class GenomicsBranch(nn.Module):
    """Encoder tabular para o status mutacional dos drivers do PDAC.

    Args:
        n_genes: Número de genes de entrada (padrão 4).
        n_states: Número de estados por gene (2 = wt/mut; 3 inclui "desconhecido").
        gene_embed_dim: Dimensão do embedding por gene.
        embed_dim: Dimensão do embedding de saída, compartilhada entre os ramos.
        hidden_dim: Largura da MLP interna.
        dropout: Taxa de dropout.
    """

    def __init__(
        self,
        n_genes: int = 4,
        n_states: int = 3,
        gene_embed_dim: int = 32,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim

        # Um espaço de embedding por (gene, estado). O offset por gene garante que
        # "KRAS mutado" e "TP53 mutado" tenham vetores distintos.
        self.gene_state_embed = nn.Embedding(n_genes * n_states, gene_embed_dim)
        self.register_buffer(
            "gene_offsets", torch.arange(n_genes) * n_states, persistent=False
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

    def forward(self, mutation_status: torch.Tensor) -> torch.Tensor:
        """Gera o embedding genômico.

        Args:
            mutation_status: Tensor long (B, n_genes) com valores em {0, 1, (2)}.

        Returns:
            Tensor (B, embed_dim) com o embedding genômico.
        """
        if mutation_status.dtype != torch.long:
            mutation_status = mutation_status.long()

        indices = mutation_status + self.gene_offsets  # (B, n_genes)
        gene_vecs = self.gene_state_embed(indices)     # (B, n_genes, gene_embed_dim)
        flat = gene_vecs.flatten(start_dim=1)          # (B, n_genes * gene_embed_dim)
        return self.mlp(flat)                          # (B, embed_dim)


if __name__ == "__main__":
    model = GenomicsBranch(embed_dim=256)
    # 2 pacientes: um KRAS+TP53 mutados; outro KRAS mutado e SMAD4 desconhecido
    x = torch.tensor([[1, 1, 0, 0], [1, 0, 2, 0]])
    out = model(x)
    print("Ramo C -- saída:", out.shape)  # esperado: torch.Size([2, 256])
