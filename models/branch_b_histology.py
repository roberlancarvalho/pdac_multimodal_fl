"""
Ramo B -- Histopatologia (Whole Slide Images / WSI).

Este ramo NÃO processa pixels da lâmina diretamente. Assume-se um pré-processamento
offline em que cada WSI é segmentada em patches (ex.: 256x256 @ 20x), e cada patch
é convertido em um vetor de features por um Foundation Model de patologia
(ex.: UNI, CONCH, Virchow, Prov-GigaPath, CTransPath). O resultado é uma
"bag" de embeddings de patches por paciente/lâmina.

O agregador aqui implementado é um MIL baseado em atenção (Attention-Based
Multiple Instance Learning, Ilse et al. 2018), com opção de um bloco
Transformer para modelar interações entre patches antes do pooling. A saída é
um embedding em nível de lâmina/paciente, na dimensão compartilhada pelos ramos.

Entrada (forward):
    patch_embeddings: Tensor float32 (B, N, feat_dim)
        -- N patches por amostra; N pode variar entre amostras (use padding + máscara).
    mask: Tensor bool (B, N) opcional
        -- True = patch válido, False = padding. Se None, todos os patches são válidos.

Saída (forward):
    slide_embedding: Tensor float32 (B, embed_dim)
    attention:       Tensor float32 (B, N) -- pesos de atenção por patch (interpretabilidade).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """Pooling por atenção com mecanismo de gating (Ilse et al., 2018)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.attn_v = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attn_u = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, in_dim)
        scores = self.attn_w(self.attn_v(x) * self.attn_u(x)).squeeze(-1)  # (B, N)
        if mask is not None:
            # `bag` sem nenhum patch válido -> evita softmax de tudo -inf (NaN).
            safe_mask = mask.clone()
            empty = ~safe_mask.any(dim=1)
            safe_mask[empty] = True
            scores = scores.masked_fill(~safe_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)                             # (B, N)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)             # (B, in_dim)
        if mask is not None:
            pooled = pooled.masked_fill(empty.unsqueeze(-1), 0.0)
            weights = weights.masked_fill(empty.unsqueeze(-1), 0.0)
        return pooled, weights


class HistologyBranch(nn.Module):
    """Agregador MIL para embeddings de patches de WSI gerados por Foundation Model.

    Args:
        input_feat_dim: Dimensão do embedding de patch produzido pelo Foundation Model
            (ex.: 1024 para UNI, 1536 para Prov-GigaPath).
        embed_dim: Dimensão do embedding de saída, compartilhada entre os ramos.
        hidden_dim: Dimensão interna da projeção e da atenção.
        n_transformer_layers: Nº de camadas Transformer para interações entre patches
            (0 desativa esse bloco e usa apenas atenção + pooling).
        n_heads: Nº de cabeças de atenção no Transformer.
        dropout: Taxa de dropout.
    """

    def __init__(
        self,
        input_feat_dim: int = 1024,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_transformer_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if n_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
            )
        else:
            self.transformer = None

        self.attn_pool = GatedAttentionPool(hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Agrega embeddings de patches em um embedding de lâmina.

        Args:
            patch_embeddings: Tensor (B, N, input_feat_dim).
            mask: Tensor bool (B, N) com True para patches válidos (opcional).

        Returns:
            slide_embedding: Tensor (B, embed_dim).
            attention: Tensor (B, N) com os pesos de atenção por patch.
        """
        x = self.input_proj(patch_embeddings)  # (B, N, hidden_dim)

        empty_bag = None
        if mask is not None:
            empty_bag = ~mask.any(dim=1)  # (B,) amostras sem nenhum patch válido

        if self.transformer is not None:
            key_padding_mask = None
            if mask is not None:
                # Uma linha totalmente `True` (bag vazia) geraria NaN no softmax da
                # atenção -> deixamos essa linha "toda válida" e zeramos a saída depois.
                safe_mask = mask.clone()
                safe_mask[empty_bag] = True
                key_padding_mask = ~safe_mask
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        pooled, attention = self.attn_pool(x, mask=mask)  # (B, hidden_dim), (B, N)
        slide_embedding = self.head(pooled)               # (B, embed_dim)

        if empty_bag is not None and empty_bag.any():
            slide_embedding = slide_embedding.masked_fill(empty_bag.unsqueeze(-1), 0.0)

        return slide_embedding, attention


if __name__ == "__main__":
    model = HistologyBranch(input_feat_dim=1024, embed_dim=256)
    bag = torch.randn(2, 500, 1024)          # 2 lâminas, 500 patches cada
    valid = torch.ones(2, 500, dtype=torch.bool)
    valid[1, 300:] = False                    # a 2ª lâmina tem apenas 300 patches
    emb, attn = model(bag, mask=valid)
    print("Ramo B -- embedding:", emb.shape, "| atenção:", attn.shape)
