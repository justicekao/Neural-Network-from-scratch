"""
utils.py

Small helpers used across the project: reproducibility seeding and
checkpoint save/load. NEEDS YOUR INPUT: nothing, this file should work
as-is.
"""

from __future__ import annotations
import random
import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Make runs reproducible across numpy/torch/python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: str, **extra) -> None:
    """Save model weights plus any extra metadata (epoch, optimizer state, ...)."""
    state = {"model_state_dict": model.state_dict(), **extra}
    torch.save(state, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location="cpu") -> dict:
    """Load model weights in-place; returns the full checkpoint dict."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    return state
