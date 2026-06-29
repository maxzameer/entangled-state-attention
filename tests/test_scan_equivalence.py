import torch

from esa.scan import esa_controlled_scan, esa_direct_scan


def test_direct_scan_matches_manual_loop() -> None:
    torch.manual_seed(0)
    a = torch.sigmoid(torch.randn(2, 8, 16))
    v = torch.randn(2, 8, 16)
    states = esa_direct_scan(a, v)

    state = torch.zeros(2, 16)
    manual = []
    for t in range(8):
        state = a[:, t] * state + v[:, t]
        manual.append(state)
    manual_states = torch.stack(manual, dim=1)

    assert torch.allclose(states, manual_states)


def test_controlled_scan_matches_manual_loop() -> None:
    torch.manual_seed(0)
    g = torch.sigmoid(torch.randn(2, 8, 16))
    v = torch.randn(2, 8, 16)
    states = esa_controlled_scan(g, v)

    state = torch.zeros(2, 16)
    manual = []
    for t in range(8):
        state = g[:, t] * state + (1.0 - g[:, t]) * v[:, t]
        manual.append(state)
    manual_states = torch.stack(manual, dim=1)

    assert torch.allclose(states, manual_states)
