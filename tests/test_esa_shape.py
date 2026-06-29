import torch

from esa import ESA, EntangledStateAttention, ESALayer, TinyESALanguageModel


def test_esa_output_shape() -> None:
    x = torch.randn(2, 16, 32)
    esa = EntangledStateAttention(dim=32, state_dim=48)
    y = esa(x)
    assert y.shape == x.shape


def test_esa_return_state_shape() -> None:
    x = torch.randn(2, 16, 32)
    esa = EntangledStateAttention(dim=32, state_dim=48)
    y, final_state = esa(x, return_state=True)
    assert y.shape == x.shape
    assert final_state.shape == (2, 48)


def test_esa_layer_shape() -> None:
    x = torch.randn(2, 16, 32)
    layer = ESALayer(dim=32, state_dim=32)
    y = layer(x)
    assert y.shape == x.shape


def test_tiny_lm_shape_and_loss() -> None:
    model = TinyESALanguageModel(vocab_size=20, block_size=8, dim=32, n_layer=2)
    idx = torch.randint(0, 20, (4, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 8, 20)
    assert loss is not None


def test_short_alias() -> None:
    x = torch.randn(2, 8, 32)
    esa = ESA(dim=32, state_dim=32)
    layer = ESALayer(dim=32, state_dim=32)
    assert esa(x).shape == x.shape
    assert layer(x).shape == x.shape
