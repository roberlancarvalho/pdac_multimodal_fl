"""
Fusão -- Co-Atenção Cross-Modal par-a-par (estilo MCAT).

Implementa a especificação da Seção 6.1 do artigo: em vez de concatenar as
modalidades e aplicar auto-atenção conjunta (`fusion_attention.py`), aqui a
atenção é computada **entre pares de modalidades**, de forma **direcional**:

    Histologia -> Radiômica : Q = tokens histológicos, K/V = tokens radiômicos
    Radiômica  -> Histologia : simétrico
    Genômica   -> {Radiômica, Histologia} : genômica como Query condicionante
    (e os demais pares, salvo se `genomics_query_only=True`)

Cada bloco segue a atenção escalonada por produto interno (Eq. 2 do artigo,
Vaswani et al. 2017):  Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V,  com
normalização das projeções antes da atenção (mitigação de dominância, item i).

Entrada (forward):
    tokens: dict[str, Tensor] -- chaves em {"radiomics","histology","genomics"},
        cada valor (B, N_m, D). Só as modalidades presentes no lote precisam
        constar. N_m pode variar por modalidade (radiômica/histologia/genômica).
    present: Tensor bool (B, 3) opcional -- presença POR AMOSTRA de cada
        modalidade na ordem `MODALITIES`. Amostras sem uma modalidade são
        mascaradas nas co-atenções e na leitura final.

Saída (forward): dict com
    "risk":          Tensor (B, n_outputs) -- log-hazard para a perda de Cox.
    "fused":         Tensor (B, D)         -- representação multimodal fundida.
    "modality_gate": dict[str, Tensor (B,)] -- peso do token [FUSION] sobre cada
        modalidade presente (interpretabilidade: contribuição por modalidade).
    "aux_risk":      dict[str, Tensor (B, n_outputs)] -- se `aux_heads=True`, um
        log-hazard unimodal por modalidade presente. Usado por
        `utils.losses.multimodal_cox_loss` para a regularização de balanceamento
        entre modalidades (mecanismo iii da Seção 6.1).
"""

from __future__ import annotations

import torch
from torch import nn

from models.fusion_attention import MODALITIES


class CrossAttentionBlock(nn.Module):
    """Atenção cruzada direcional (pre-norm) + MLP. Q de uma modalidade, K/V de outra."""

    def __init__(self, dim: int, n_heads: int, dropout: float, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        delta, attn_w = self.attn(
            qn, kvn, kvn,
            key_padding_mask=kv_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        q = q + delta
        q = q + self.mlp(self.norm_mlp(q))
        return q, attn_w


class CrossModalCoAttentionFusion(nn.Module):
    """Fusão multimodal por co-atenção par-a-par + cabeça de risco.

    Args:
        embed_dim: Dimensão dos tokens (igual nos três ramos).
        n_layers: Nº de rodadas de co-atenção par-a-par.
        n_heads: Nº de cabeças de atenção.
        dropout: Taxa de dropout.
        n_outputs: Saídas da cabeça (1 = risco Cox).
        genomics_query_only: Se True, a genômica só atua como Query (os demais
            ramos não a tomam como K/V) -- reduz o risco de a genômica, de baixa
            cardinalidade, ser diluída.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.2,
        n_outputs: int = 1,
        genomics_query_only: bool = False,
        aux_heads: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modalities = MODALITIES
        self.genomics_query_only = genomics_query_only
        self.aux_heads = aux_heads

        self.type_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, embed_dim)) for m in self.modalities}
        )
        self.in_norm = nn.ModuleDict({m: nn.LayerNorm(embed_dim) for m in self.modalities})
        for p in self.type_embed.values():
            nn.init.trunc_normal_(p, std=0.02)

        pairs = [(q, kv) for q in self.modalities for kv in self.modalities if q != kv]
        if genomics_query_only:
            pairs = [(q, kv) for (q, kv) in pairs if kv != "genomics"]
        self.pairs = pairs
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {f"{q}->{kv}": CrossAttentionBlock(embed_dim, n_heads, dropout) for (q, kv) in pairs}
            )
            for _ in range(n_layers)
        )

        # Pooling por modalidade (Query aprendível) -> um vetor por modalidade.
        self.pool_query = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, embed_dim)) for m in self.modalities}
        )
        self.pool_attn = nn.ModuleDict(
            {m: nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
             for m in self.modalities}
        )
        for p in self.pool_query.values():
            nn.init.trunc_normal_(p, std=0.02)

        # Leitura: token [FUSION] atende sobre os vetores por modalidade.
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.fusion_token, std=0.02)
        self.readout = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.readout_norm = nn.LayerNorm(embed_dim)

        self.risk_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_outputs),
        )

        # Cabeças de risco unimodais (regularização de balanceamento entre modalidades).
        if aux_heads:
            self.aux_risk = nn.ModuleDict(
                {
                    m: nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_outputs))
                    for m in self.modalities
                }
            )

    @staticmethod
    def _cross(block, q, kv, kv_present):
        """Aplica um bloco de co-atenção com máscara de presença POR AMOSTRA da modalidade K/V."""
        b, n_kv, _ = kv.shape
        kpm = None
        if kv_present is not None and bool((~kv_present).any()):
            kpm = torch.zeros(b, n_kv, dtype=torch.bool, device=kv.device)
            kpm[~kv_present] = True
            kpm[~kv_present, 0] = False  # evita linha 100% mascarada -> NaN no softmax
        out, attn_w = block(q, kv, kv_key_padding_mask=kpm)
        if kv_present is not None:
            out = torch.where(kv_present.view(b, 1, 1), out, q)  # sem K/V -> mantém Q
        return out, attn_w

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        present: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        active = [m for m in self.modalities if tokens.get(m) is not None]
        if not active:
            raise ValueError("Nenhuma modalidade fornecida à co-atenção.")
        b = tokens[active[0]].size(0)
        device = tokens[active[0]].device

        if present is None:
            present = torch.ones(b, len(self.modalities), dtype=torch.bool, device=device)
        pres = {
            m: (present[:, i] if i < present.shape[1]
                else torch.zeros(b, dtype=torch.bool, device=device))
            for i, m in enumerate(self.modalities)
        }

        # Normaliza + tipo de modalidade; sanitiza NaN/Inf de amostras sem a modalidade.
        h = {
            m: self.in_norm[m](torch.nan_to_num(tokens[m])) + self.type_embed[m]
            for m in active
        }

        for layer in self.layers:
            deltas: dict[str, list[torch.Tensor]] = {m: [] for m in active}
            for (q, kv) in self.pairs:
                if q in active and kv in active:
                    out, _ = self._cross(layer[f"{q}->{kv}"], h[q], h[kv], pres[kv])
                    deltas[q].append(out)
            h = {
                m: (torch.stack(deltas[m], 0).mean(0) if deltas[m] else h[m])
                for m in active
            }

        # Pool por modalidade -> (B, 1, D) cada; empilha na ordem canônica.
        pooled, order = [], []
        aux_risk: dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m in active:
                q = self.pool_query[m].expand(b, -1, -1)
                vec, _ = self.pool_attn[m](q, h[m], h[m], need_weights=False)
                pooled.append(vec)
                order.append(m)
                if self.aux_heads:
                    aux_risk[m] = self.aux_risk[m](vec.squeeze(1))  # (B, n_outputs)
        pooled = torch.cat(pooled, dim=1)  # (B, M_active, D)

        # Máscara de presença por amostra para a leitura final.
        pool_present = torch.stack([pres[m] for m in order], dim=1)  # (B, M_active)
        kpm = None
        if bool((~pool_present).any()):
            kpm = ~pool_present.clone()
            no_mod = ~pool_present.any(dim=1)
            kpm[no_mod] = False  # amostra sem nenhuma modalidade -> não mascara (raro)

        fus = self.fusion_token.expand(b, -1, -1)
        fused, gate = self.readout(
            fus, pooled, pooled, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=True,
        )
        fused = self.readout_norm(fused.squeeze(1))
        risk = self.risk_head(fused)

        modality_gate = {m: gate[:, 0, i] for i, m in enumerate(order)}
        out = {"risk": risk, "fused": fused, "modality_gate": modality_gate}
        if self.aux_heads:
            out["aux_risk"] = aux_risk
        return out


if __name__ == "__main__":
    fusion = CrossModalCoAttentionFusion(embed_dim=128)
    tok = {
        "radiomics": torch.randn(2, 8, 128),
        "histology": torch.randn(2, 8, 128),
        "genomics": torch.randn(2, 4, 128),
    }
    out = fusion(tok)
    print("Co-atenção -- risco:", out["risk"].shape, "| fundido:", out["fused"].shape)
    print("modality_gate:", {k: tuple(v.shape) for k, v in out["modality_gate"].items()})
    print("aux_risk:", {k: tuple(v.shape) for k, v in out["aux_risk"].items()})

    # Paciente 2 sem histologia (presença por amostra):
    present = torch.tensor([[True, True, True], [True, False, True]])
    out2 = fusion(tok, present=present)
    print("Co-atenção (amostra sem Ramo B) -- risco:", out2["risk"].shape)
