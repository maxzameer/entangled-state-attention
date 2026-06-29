from __future__ import annotations

import torch
from torch import nn

from .config import ESAConfig
from .esa import EntangledStateAttention


class GatedFeedForward(nn.Module):
    """Compact gated feed-forward network.

    This is intentionally small and dependency-free for the first public ESA
    release. It can be replaced by a larger FFN in downstream models.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.up(x)) * torch.sigmoid(self.gate(x))))


class ESALayer(nn.Module):
    """Stackable residual neural network layer using Entangled State Attention.

    ESALayer is similar in role to a Transformer layer: it applies normalization,
    Entangled State Attention, residual connections, and a feed-forward network.
    Multiple ESALayer instances can be stacked sequentially.
    """

    def __init__(
        self,
        dim: int | ESAConfig,
        state_dim: int | None = None,
        ffn_mult: float = 4.0,
        gate_mode: str = "controlled",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(dim, ESAConfig):
            config = dim
            dim_value = config.dim
        else:
            config = ESAConfig(
                dim=dim,
                state_dim=state_dim,
                gate_mode=gate_mode,  # type: ignore[arg-type]
                dropout=dropout,
                bias=bias,
            )
            dim_value = dim

        self.norm1 = nn.LayerNorm(dim_value)
        self.esa = EntangledStateAttention(config)
        self.norm2 = nn.LayerNorm(dim_value)
        hidden_dim = max(1, int(dim_value * ffn_mult))
        self.ffn = GatedFeedForward(dim_value, hidden_dim, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.esa(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

