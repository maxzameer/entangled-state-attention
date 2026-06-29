from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .config import ESAConfig
from .scan import esa_controlled_scan, esa_direct_scan


class EntangledStateAttention(nn.Module):
    """Entangled State Attention layer.

    ESA replaces a full token-token causal attention matrix with a token-local
    projection and a causal state recurrence.

    Direct mode:
        A_t = sigmoid(x_t W_A)
        V_t = x_t W_V
        E_t = A_t * E_{t-1} + V_t

    Controlled mode:
        g_t = sigmoid(x_t W_g)
        V_t = x_t W_V
        E_t = g_t * E_{t-1} + (1 - g_t) * V_t

    Readout:
        q_t = x_t W_Q
        y_t = W_O(q_t * Norm(E_t))
    """

    def __init__(
        self,
        dim: int | ESAConfig,
        state_dim: int | None = None,
        gate_mode: Literal["direct", "controlled"] = "controlled",
        use_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(dim, ESAConfig):
            config = dim
        else:
            config = ESAConfig(
                dim=dim,
                state_dim=state_dim,
                gate_mode=gate_mode,
                use_norm=use_norm,
                dropout=dropout,
                bias=bias,
            )
        self.config = config
        self.dim = config.dim
        self.state_dim = int(config.state_dim)
        self.gate_mode = config.gate_mode

        self.gate_proj = nn.Linear(self.dim, self.state_dim, bias=config.bias)
        self.value_proj = nn.Linear(self.dim, self.state_dim, bias=config.bias)
        self.query_proj = nn.Linear(self.dim, self.state_dim, bias=config.bias)
        self.out_proj = nn.Linear(self.state_dim, self.dim, bias=config.bias)
        self.norm = nn.LayerNorm(self.state_dim) if config.use_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        return_state: bool = False,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ESA over a causal sequence.

        Args:
            x: Input tensor with shape ``[batch, seq, dim]``.
            initial_state: Optional initial state with shape ``[batch, state_dim]``.
            return_state: Return final state along with output.
            return_states: Return full state sequence along with output.

        Returns:
            Output tensor of shape ``[batch, seq, dim]``. Optionally also returns
            the final state and/or full state sequence.
        """
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq, dim]")
        if x.size(-1) != self.dim:
            raise ValueError(f"last dimension must be {self.dim}, got {x.size(-1)}")

        gate = torch.sigmoid(self.gate_proj(x))
        value = self.value_proj(x)

        if self.gate_mode == "direct":
            states = esa_direct_scan(gate, value, initial_state)
        else:
            states = esa_controlled_scan(gate, value, initial_state)

        query = self.query_proj(x)
        read = query * self.norm(states)
        out = self.dropout(self.out_proj(read))

        if return_state and return_states:
            return out, states[:, -1], states
        if return_state:
            return out, states[:, -1]
        if return_states:
            return out, states
        return out


class ControlledEntangledStateAttention(EntangledStateAttention):
    """Convenience wrapper for the controlled ESA variant."""

    def __init__(
        self,
        dim: int,
        state_dim: int | None = None,
        use_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            dim=dim,
            state_dim=state_dim,
            gate_mode="controlled",
            use_norm=use_norm,
            dropout=dropout,
            bias=bias,
        )
