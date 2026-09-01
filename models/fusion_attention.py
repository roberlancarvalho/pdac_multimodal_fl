"""
Fusão -- Cross-Modal Attention.

Une as representações dos três ramos (A: radiômico, B: histopatologia,
C: genômico) e produz uma predição de risco/sobrevida.

Estratégia:
  1. Cada embedding de modalidade (dimensão `embed_dim`) recebe um "modality
     token" aprendível (tipo de modalidade) somado a ele -- análogo a um
     positional/segment embedding.
  2. As 3 modalidades formam uma sequência (B, 3, embed_dim) que passa por
     blocos de atenção multi-cabeça. Como a atenção é computada entre as três
     modalidades, cada representação é reponderada em função das outras
     (cross-modal). Modalidades ausentes são tratadas via máscara.
  3. Um token [FUSION] (tipo CLS) agrega a sequência; sua saída alimenta a
     cabeça de predição.

Robustez a modalidade ausente: em coortes federadas nem todo paciente tem as três
modalidades. `forward` aceita uma máscara (B, 3) indicando modalidades presentes.

Entrada (forward):
    embeddings: dict[str, Tensor] com chaves em {"radiomics", "histology", "genomics"},
        cada valor um Tensor (B, embed_dim). Chaves ausentes = modalidade ausente.
    (alternativamente) Tensor (B, 3, embed_dim) + mask (B, 3).

Saída (forward): dict com
    "risk":        Tensor (B, 1)  -- escore de risco (log-hazard) para perda de Cox.
    "fused":       Tensor (B, embed_dim) -- representação multimodal fundida.
    "attention":   Tensor (B, n_layers, n_heads, 3+1, 3+1) -- mapas de atenção (opcional).
"""

from __future__ import annotations

import torch
import torch.nn as nn

MODALITIES: tuple[str, ...] = ("radiomics", "histology", "genomics")


class CrossModalAttentionFusion(nn.Module):
    """Fusão multimodal por atenção cruzada + cabeça de predição de risco.

    Args:
        embed_dim: Dimensão dos embeddings de entrada (igual nos três ramos).
        n_layers: Nº de blocos Transformer de fusão.
        n_heads: Nº de cabeças de atenção.
        dropout: Taxa de dropout.
        n_outputs: Nº de saídas da cabeça (1 = risco Cox; >1 = intervalos discretos
            de sobrevida / classificação de estágio).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = len(MODALITIES)

        # Tokens de tipo de modalidade + token de fusão (CLS).
        self.modality_tokens = nn.Parameter(torch.zeros(self.n_modalities, embed_dim))
        self.fusion_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.modality_tokens, std=0.02)
        nn.init.trunc_normal_(self.fusion_token, std=0.02)

        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.risk_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_outputs),
        )

    def _stack_inputs(
        self,
        embeddings: dict[str, torch.Tensor] | torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normaliza a entrada para (B, 3, embed_dim) + máscara (B, 3)."""
        if isinstance(embeddings, torch.Tensor):
            seq = embeddings
            if mask is None:
                mask = torch.ones(seq.size(0), self.n_modalities, dtype=torch.bool, device=seq.device)
            return seq, mask

        # dict -> stack na ordem canônica; modalidade ausente = zeros + mask False.
        ref = next(iter(embeddings.values()))
        b = ref.size(0)
        seq = torch.zeros(b, self.n_modalities, self.embed_dim, device=ref.device, dtype=ref.dtype)
        built_mask = torch.zeros(b, self.n_modalities, dtype=torch.bool, device=ref.device)
        for i, name in enumerate(MODALITIES):
            if name in embeddings and embeddings[name] is not None:
                seq[:, i] = embeddings[name]
                built_mask[:, i] = True
        if mask is not None:
            built_mask = built_mask & mask
        return seq, built_mask

    def forward(
        self,
        embeddings: dict[str, torch.Tensor] | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Funde as modalidades e prediz o risco.

        Args:
            embeddings: dict {modalidade: (B, embed_dim)} ou Tensor (B, 3, embed_dim).
            mask: Tensor bool (B, 3), True = modalidade presente (opcional).

        Returns:
            dict com "risk" (B, n_outputs) e "fused" (B, embed_dim).
        """
        seq, mod_mask = self._stack_inputs(embeddings, mask)  # (B, 3, D), (B, 3)
        b = seq.size(0)

        # Zera modalidades ausentes (e sanitiza qualquer NaN/Inf que um ramo tenha
        # produzido para uma amostra sem aquela modalidade) antes da normalização.
        seq = torch.nan_to_num(seq) * mod_mask.unsqueeze(-1).to(seq.dtype)

        seq = self.input_norm(seq) + self.modality_tokens.unsqueeze(0)

        fusion = self.fusion_token.expand(b, 1, self.embed_dim)
        tokens = torch.cat([fusion, seq], dim=1)  # (B, 1+3, D)

        # O token de fusão está sempre presente; modalidades ausentes são ignoradas.
        fusion_present = torch.ones(b, 1, dtype=torch.bool, device=seq.device)
        key_padding_mask = ~torch.cat([fusion_present, mod_mask], dim=1)  # True = ignora

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)  # (B, 1+3, D)
        fused = encoded[:, 0]  # representação do token de fusão -> (B, D)

        risk = self.risk_head(fused)  # (B, n_outputs)
        return {"risk": risk, "fused": fused}


if __name__ == "__main__":
    fusion = CrossModalAttentionFusion(embed_dim=256)
    emb = {
        "radiomics": torch.randn(2, 256),
        "histology": torch.randn(2, 256),
        "genomics": torch.randn(2, 256),
    }
    out = fusion(emb)
    print("Fusão -- risco:", out["risk"].shape, "| fundido:", out["fused"].shape)

    # Paciente 2 sem histopatologia:
    emb_missing = {"radiomics": torch.randn(2, 256), "genomics": torch.randn(2, 256)}
    out2 = fusion(emb_missing)
    print("Fusão (sem Ramo B) -- risco:", out2["risk"].shape)
