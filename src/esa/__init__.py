"""Entangled State Attention (ESA).

A lightweight PyTorch library for causal sequence modeling with scan-friendly
state updates.
"""

from .config import ESAConfig
from .esa import ControlledEntangledStateAttention, EntangledStateAttention
from .block import ESALayer
from .model import TinyESALanguageModel

__version__ = "0.1.0"

# Short public alias for the core attention layer.
ESA = EntangledStateAttention

__all__ = [
    "ESAConfig",
    "EntangledStateAttention",
    "ControlledEntangledStateAttention",
    "ESA",
    "ESALayer",
    "TinyESALanguageModel",
]
