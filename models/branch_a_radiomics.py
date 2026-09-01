"""
Ramo A -- Radiômico 3D (Tomografia Computadorizada / NIfTI).

Este ramo processa volumes de TC do pâncreas (formato NIfTI, já pré-processados:
resample isotrópico, janelamento HU, crop/pad em torno da ROI pancreática) e
produz um vetor de representação (embedding) que descreve a textura e a forma
radiômica do tumor.

Backbone: rede convolucional 3D da biblioteca MONAI (por padrão um DenseNet121
3D). A cabeça de classificação original é substituída por uma projeção linear
para a dimensão de embedding compartilhada pelos três ramos, de modo que o
mecanismo de fusão (Cross-Modal Attention) receba tokens de mesma dimensão.

Entrada  (forward): Tensor float32 de shape (B, in_channels, D, H, W)
                    -- volume 3D de TC. `in_channels` = 1 (single-phase) ou 2
                    (fases arterial pancreática + venosa portal, encoder compartilhado).
Saída    (forward):
    - `return_tokens=False` (padrão): Tensor (B, embed_dim) -- embedding por paciente.
    - `return_tokens=True`: Tensor (B, T, embed_dim) -- sequência de tokens
      espaciais latentes (T = produto de `token_grid`), preservando a localização
      da lesão para a co-atenção cruzada com a histologia (ver `fusion_coattention.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from monai.networks.nets import DenseNet121
except ImportError as exc:  # pragma: no cover - dependência opcional em dev
    raise ImportError(
        "MONAI é necessário para o Ramo A. Instale com `pip install monai`."
    ) from exc


class RadiomicsBranch3D(nn.Module):
    """Extrator de features radiômicas a partir de volumes 3D de TC.

    Args:
        in_channels: Número de canais do volume de entrada (1 para TC single-phase).
        embed_dim: Dimensão do embedding de saída, compartilhada entre os ramos.
        backbone_dropout: Dropout aplicado dentro do backbone MONAI.
        pretrained: Se True, tenta carregar pesos pré-treinados do backbone
            (quando disponíveis para a variante 3D escolhida).
        freeze_backbone: Se True, congela os parâmetros do backbone e treina
            apenas a cabeça de projeção (útil em regimes federados com poucos dados).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        backbone_dropout: float = 0.2,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        token_grid: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_grid = tuple(token_grid)

        # DenseNet121 3D da MONAI. Usamos `out_channels` como dimensão intermediária
        # e substituímos a última camada por uma projeção para `embed_dim`.
        backbone_features = 1024
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=backbone_features,
            dropout_prob=backbone_dropout,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Cabeça para o vetor único por paciente (fusão legada / uso isolado).
        self.head = nn.Sequential(
            nn.LayerNorm(backbone_features),
            nn.Linear(backbone_features, embed_dim),
            nn.GELU(),
            nn.Dropout(backbone_dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        # Cabeça para os tokens espaciais (co-atenção cruzada).
        self.token_pool = nn.AdaptiveAvgPool3d(self.token_grid)
        self.token_head = nn.Sequential(
            nn.LayerNorm(backbone_features),
            nn.Linear(backbone_features, embed_dim),
            nn.GELU(),
            nn.Dropout(backbone_dropout),
        )

    def forward(self, ct_volume: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """Extrai a representação radiômica de um lote de volumes 3D.

        Args:
            ct_volume: Tensor (B, in_channels, D, H, W).
            return_tokens: Se True, devolve a sequência de tokens espaciais.

        Returns:
            (B, embed_dim) se `return_tokens=False`; (B, T, embed_dim) caso contrário.
        """
        if not return_tokens:
            return self.head(self.backbone(ct_volume))  # (B, embed_dim)

        fmap = torch.relu(self.backbone.features(ct_volume))  # (B, C, d, h, w)
        grid = self.token_pool(fmap)                          # (B, C, gz, gy, gx)
        seq = grid.flatten(2).transpose(1, 2)                 # (B, T, C)
        return self.token_head(seq)                           # (B, T, embed_dim)


if __name__ == "__main__":
    model = RadiomicsBranch3D(in_channels=1, embed_dim=256)
    dummy = torch.randn(2, 1, 64, 96, 96)
    print("Ramo A -- vetor:", model(dummy).shape)                 # (2, 256)
    print("Ramo A -- tokens:", model(dummy, return_tokens=True).shape)  # (2, 8, 256)
