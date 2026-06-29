import torch

from esa import EntangledStateAttention


x = torch.randn(2, 128, 256)  # batch, sequence, hidden dimension

esa = EntangledStateAttention(
    dim=256,
    state_dim=256,
    gate_mode="controlled",
)

y = esa(x)

print("input:", x.shape)
print("output:", y.shape)
