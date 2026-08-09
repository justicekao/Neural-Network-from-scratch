"""
tests/test_models.py

Sanity checks for the MonodCNN architecture. Run with `pytest` from the
project root. NEEDS YOUR INPUT: nothing.
"""

import torch
from monod_pinn.models import MonodCNN


def test_output_shape():
    model = MonodCNN(in_channels=1)
    x = torch.randn(4, 1, 100)  # batch=4, channels=1, T=100
    out = model(x)
    assert out.shape == (4, 3)


def test_output_is_positive():
    """mu_max, Ks, Y are all physically non-negative — softplus should enforce this."""
    model = MonodCNN(in_channels=1)
    x = torch.randn(8, 1, 50)
    out = model(x)
    assert torch.all(out > 0)


def test_handles_variable_length_input():
    """AdaptiveAvgPool1d means the network shouldn't care about T."""
    model = MonodCNN(in_channels=1)
    for T in (30, 75, 200):
        x = torch.randn(2, 1, T)
        out = model(x)
        assert out.shape == (2, 3)
