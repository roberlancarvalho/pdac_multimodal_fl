"""
Contabilização do orçamento de privacidade (ε) para DP-FedAvg.

`DifferentialPrivacyServerSideFixedClipping` do Flower adiciona ruído gaussiano
`N(0, (σ·C)²)` ao agregado a cada rodada, com clipping fixo em `C`. Tratando cada
rodada como um passo do mecanismo gaussiano subamostrado (taxa = fração de
clientes por rodada), o `RDPAccountant` da Opacus dá o (ε, δ) acumulado.

`epsilon_estimate` é uma **estimativa** (privacidade em nível de cliente/instituição,
não por-amostra); serve para reportar a ordem de grandeza e o efeito de σ.
"""

from __future__ import annotations


def epsilon_estimate(
    noise_multiplier: float,
    num_rounds: int,
    sample_rate: float = 1.0,
    delta: float = 1e-5,
) -> float | None:
    """(ε) acumulado após `num_rounds` rodadas de DP-FedAvg, ou None se indisponível."""
    if noise_multiplier <= 0 or num_rounds <= 0:
        return None
    try:
        from opacus.accountants import RDPAccountant

        acc = RDPAccountant()
        for _ in range(int(num_rounds)):
            acc.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        return float(acc.get_epsilon(delta=delta))
    except Exception:
        return None
