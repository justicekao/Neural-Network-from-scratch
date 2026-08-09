"""
models.py

The 1D convolutional neural network that reads a population time series
(and optionally a substrate time series) and outputs Monod parameters
[mu_max, Ks, Y].

NEEDS YOUR INPUT: the defaults (channels, kernel sizes, sequence length
assumption) are reasonable starting points but not tuned to your data.
Once you know your real time-series length and noise level, revisit
`conv_channels` and `kernel_size` below.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MonodCNN(nn.Module):
    """
    Input:  (batch, in_channels, T) time series.
              in_channels = 1 if you only measure population X(t).
              in_channels = 2 if you also measure substrate S(t).
    Output: (batch, 3) predicted [mu_max, Ks, Y], passed through softplus
            so they come out strictly positive (all three Monod parameters
            are physically non-negative).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # NEEDS YOUR INPUT: in_channels must match how many time series
        # you feed per sample (see dataset.py). Default assumes you only
        # have population counts, not substrate measurements.
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers.append(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # Global average pooling makes the network robust to varying input
        # length T (you don't need every curve to have identical length).
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(c_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [mu_max, Ks, Y]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, T)
        Returns:
            params: (batch, 3), strictly positive via softplus.
        """
        z = self.conv(x)
        z = self.pool(z).squeeze(-1)     # (batch, c_in)
        raw = self.head(z)               # (batch, 3)
        params = nn.functional.softplus(raw)  # keep mu_max, Ks, Y > 0
        return params
