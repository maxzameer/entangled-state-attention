from __future__ import annotations

import torch


def esa_direct_scan(a: torch.Tensor, v: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
    """Compute direct ESA recurrence with a simple causal loop.

    E_t = A_t * E_{t-1} + V_t

    Args:
        a: Gate tensor of shape ``[batch, seq, state_dim]``.
        v: Value tensor of shape ``[batch, seq, state_dim]``.
        initial_state: Optional tensor of shape ``[batch, state_dim]``.

    Returns:
        State sequence of shape ``[batch, seq, state_dim]``.
    """
    if a.shape != v.shape:
        raise ValueError(f"a and v must have the same shape, got {a.shape} and {v.shape}")
    if a.dim() != 3:
        raise ValueError("a and v must have shape [batch, seq, state_dim]")

    batch, seq_len, state_dim = a.shape
    state = _make_initial_state(v, initial_state, batch, state_dim)
    states = []
    for t in range(seq_len):
        state = a[:, t] * state + v[:, t]
        states.append(state)
    return torch.stack(states, dim=1)


def esa_controlled_scan(g: torch.Tensor, v: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
    """Compute controlled ESA recurrence with a simple causal loop.

    E_t = g_t * E_{t-1} + (1 - g_t) * V_t

    Args:
        g: Keep gate tensor of shape ``[batch, seq, state_dim]``.
        v: Candidate value tensor of shape ``[batch, seq, state_dim]``.
        initial_state: Optional tensor of shape ``[batch, state_dim]``.

    Returns:
        State sequence of shape ``[batch, seq, state_dim]``.
    """
    if g.shape != v.shape:
        raise ValueError(f"g and v must have the same shape, got {g.shape} and {v.shape}")
    if g.dim() != 3:
        raise ValueError("g and v must have shape [batch, seq, state_dim]")

    batch, seq_len, state_dim = g.shape
    state = _make_initial_state(v, initial_state, batch, state_dim)
    states = []
    for t in range(seq_len):
        state = g[:, t] * state + (1.0 - g[:, t]) * v[:, t]
        states.append(state)
    return torch.stack(states, dim=1)


def _make_initial_state(
    reference: torch.Tensor,
    initial_state: torch.Tensor | None,
    batch: int,
    state_dim: int,
) -> torch.Tensor:
    if initial_state is None:
        return reference.new_zeros(batch, state_dim)
    if initial_state.shape != (batch, state_dim):
        raise ValueError(
            f"initial_state must have shape {(batch, state_dim)}, got {tuple(initial_state.shape)}"
        )
    return initial_state
