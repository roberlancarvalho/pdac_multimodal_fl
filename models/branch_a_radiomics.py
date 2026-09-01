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
                    -- volume 3D de TC. `in_channels` costuma ser 1.
Saída    (forward): Tensor float32 de shape (B, embed_dim)
                    -- embedding radiômico por paciente.
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
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

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

        self.head = nn.Sequential(
            nn.LayerNorm(backbone_features),
            nn.Linear(backbone_features, embed_dim),
            nn.GELU(),
            nn.Dropout(backbone_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, ct_volume: torch.Tensor) -> torch.Tensor:
        """Extrai o embedding radiômico de um lote de volumes 3D.

        Args:
            ct_volume: Tensor (B, in_channels, D, H, W).

        Returns:
            Tensor (B, embed_dim) com o embedding radiômico.
        """
        features = self.backbone(ct_volume)  # (B, backbone_features)
        embedding = self.head(features)      # (B, embed_dim)
        return embedding


if __name__ == "__main__":
    model = RadiomicsBranch3D(in_channels=1, embed_dim=256)
    dummy = torch.randn(2, 1, 64, 96, 96)
    out = model(dummy)
    print("Ramo A -- saída:", out.shape)  # esperado: torch.Size([2, 256])
