import torch

from esa import EntangledStateAttention


def test_esa_is_causal() -> None:
    torch.manual_seed(0)
    esa = EntangledStateAttention(dim=16, state_dim=16, dropout=0.0)
    esa.eval()

    x1 = torch.randn(2, 10, 16)
    x2 = x1.clone()
    x2[:, 6:, :] = torch.randn_like(x2[:, 6:, :])  # change future tokens only

    y1 = esa(x1)
    y2 = esa(x2)

    # Outputs before the changed future region should be identical.
    assert torch.allclose(y1[:, :6, :], y2[:, :6, :], atol=1e-6)
