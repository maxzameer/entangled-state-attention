from __future__ import annotations

import torch


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a PyTorch module."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
