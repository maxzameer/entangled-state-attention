from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block import ESALayer


class TinyESALanguageModel(nn.Module):
    """Tiny causal language model using ESA layers.

    This model is for examples, smoke tests, and small experiments. It is not
    intended to be a full production training stack.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        dim: int = 256,
        n_layer: int = 4,
        state_dim: int | None = None,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        gate_mode: str = "controlled",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ESALayer(
                    dim=dim,
                    state_dim=state_dim,
                    ffn_mult=ffn_mult,
                    gate_mode=gate_mode,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.dim() != 2:
            raise ValueError("idx must have shape [batch, seq]")
        batch, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch * seq_len, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
