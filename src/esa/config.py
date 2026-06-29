from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GateMode = Literal["direct", "controlled"]


@dataclass
class ESAConfig:
    """Configuration for Entangled State Attention.

    Args:
        dim: Input and output hidden dimension.
        state_dim: Internal ESA state dimension. Defaults to ``dim``.
        gate_mode: ``"direct"`` uses E_t = A_t * E_{t-1} + V_t.
            ``"controlled"`` uses E_t = g_t * E_{t-1} + (1 - g_t) * V_t.
        use_norm: Whether to normalize the state before readout.
        dropout: Dropout probability applied after output projection.
        bias: Whether linear layers use bias.
    """

    dim: int
    state_dim: int | None = None
    gate_mode: GateMode = "controlled"
    use_norm: bool = True
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.state_dim is None:
            self.state_dim = self.dim
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.gate_mode not in {"direct", "controlled"}:
            raise ValueError("gate_mode must be either 'direct' or 'controlled'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
