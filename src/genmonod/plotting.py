"""
plotting.py

Builds a matplotlib figure comparing observed data (markers) against a
fitted trajectory (lines), for one dataset. NEEDS YOUR INPUT: nothing,
adjust styling if you like.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from .data_io import Dataset
from .config import SystemConfig


def plot_fit(cfg: SystemConfig, ds: Dataset, trajectory: np.ndarray):
    """
    Args:
        cfg: the (resolved) SystemConfig used to produce `trajectory`.
        ds: the Dataset being compared against.
        trajectory: (len(ds.t), S+M+T) simulated array from fitting.FitResult.

    Returns:
        matplotlib Figure.
    """
    names = cfg.strain_names + cfg.metabolite_names + cfg.toxin_names
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for j, name in enumerate(names):
        if not np.all(np.isnan(ds.Y[:, j])):
            ax.plot(ds.t, ds.Y[:, j], "o", color=colors[j], markersize=4, label=f"{name} (observed)")
        ax.plot(ds.t, trajectory[:, j], "-", color=colors[j], linewidth=2, label=f"{name} (fit)")

    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("value (log scale)")
    ax.set_title(ds.name)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig
