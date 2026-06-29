# Entangled State Attention

Entangled State Attention (ESA) is a PyTorch library for causal sequence modeling using a scan-friendly recurrent state update instead of a full token-token causal attention matrix.

This repository is the first clean library release of the ESA mechanism described in:

**Entangled State Attention: Associative State Scanning for Causal Sequence Modeling**  
DOI: https://doi.org/10.5281/zenodo.20973958

Status: early research library, `v0.1.0`.

## Naming

Install package:

```bash
pip install entangled-state-attention
```

Python import:

```python
from esa import ESA, EntangledStateAttention, ESALayer
```

Recommended short usage:

```python
from esa import ESA, ESALayer
```

- `ESA` is a short alias for `EntangledStateAttention`.
- `EntangledStateAttention` is the core ESA layer.
- `ESALayer` is a stackable residual layer using ESA.

## Core idea

For an input sequence `x`, ESA computes token-local gates and values, then builds a causal state sequence.

Direct ESA:

```math
A_t = \sigma(x_t W_A)
```

```math
V_t = x_t W_V
```

```math
E_t = A_t \odot E_{t-1} + V_t
```

Readout:

```math
q_t = x_t W_Q
```

```math
y_t = W_O(q_t \odot \mathrm{Norm}(E_t))
```

Controlled ESA:

```math
g_t = \sigma(x_t W_g)
```

```math
E_t = g_t \odot E_{t-1} + (1 - g_t) \odot V_t
```

The controlled form couples old-state retention with new-content writing: when the keep gate is high, the write amount is low.

## Direct vs controlled mode

ESA supports two recurrence modes:

```python
ESA(dim=256, gate_mode="controlled")
```

```python
ESA(dim=256, gate_mode="direct")
```

Use `gate_mode="controlled"` for most experiments and downstream models. It is the safer default because the same gate controls both how much old state is retained and how much new content is written. This creates a clear tradeoff:

```text
high keep gate  -> more old state, less new writing
low keep gate   -> less old state, more new writing
```

Use `gate_mode="direct"` when you want to reproduce or study the original direct ESA recurrence. In direct mode, the old-state gate controls retention, but the new value is added directly. This is useful for ablations, research comparison, and testing the original formulation, but it is less constrained than controlled mode.

Recommended default:

```python
from esa import ESA

layer = ESA(dim=256, gate_mode="controlled")
```

## Installation

From source:

```bash
pip install -e .
```

With development tools:

```bash
pip install -e ".[dev]"
```

After PyPI release:

```bash
pip install entangled-state-attention
```

## Minimal usage

```python
import torch
from esa import ESA

x = torch.randn(2, 128, 256)  # batch, sequence, hidden dimension

esa = ESA(
    dim=256,
    state_dim=256,
    gate_mode="controlled",
)

y = esa(x)

print(y.shape)  # torch.Size([2, 128, 256])
```

The full class name is also available:

```python
from esa import EntangledStateAttention

esa = EntangledStateAttention(dim=256, state_dim=256)
```

## ESA layer

`ESALayer` is the stackable layer form. It contains normalization, Entangled State Attention, residual connections, and a feed-forward network.

```python
import torch
from esa import ESALayer

x = torch.randn(2, 128, 256)

layer = ESALayer(dim=256, state_dim=256)

y = layer(x)
```

You can stack ESA layers:

```python
import torch.nn as nn
from esa import ESALayer

model = nn.Sequential(
    ESALayer(dim=256, state_dim=256),
    ESALayer(dim=256, state_dim=256),
    ESALayer(dim=256, state_dim=256),
)
```

## Tiny language model example

```python
from esa import TinyESALanguageModel

model = TinyESALanguageModel(
    vocab_size=5000,
    block_size=256,
    dim=256,
    n_layer=4,
)
```

This tiny model is included for smoke tests and simple experiments. It is not meant to be the official ESA-SLM release.

Future ESA-based language models should install this library with `pip` and live in separate model repositories.

## Run examples

```bash
python examples/minimal_esa.py
python examples/tiny_lm_demo.py
```

## Run tests

```bash
python -m pytest
```

Included tests:

- ESA output shape
- ESA layer shape
- tiny LM forward/loss
- causal behavior
- direct scan equivalence
- controlled scan equivalence
- public alias imports

## Project scope

Version `0.1.0` intentionally includes only the standalone ESA mechanism:

- `ESA`
- `EntangledStateAttention`
- `ControlledEntangledStateAttention`
- `ESALayer`
- `TinyESALanguageModel`
- examples
- tests

It does not include SOUP, Entangled Router, ObserverTransformer, or other architecture projects. Those should remain separate projects.

## Citation

```bibtex
@misc{hussain2026entangledstateattention,
  title        = {Entangled State Attention: Associative State Scanning for Causal Sequence Modeling},
  author       = {Hussain, Zameer},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20973958},
  url          = {https://doi.org/10.5281/zenodo.20973958}
}
```

## License

Apache License 2.0.
