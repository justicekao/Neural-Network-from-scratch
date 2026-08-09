"""
dataset.py

Two dataset classes:
  - SyntheticMonodDataset: generates random-parameter Monod curves on the
    fly (or once and cached) — use this to pretrain/sanity-check the
    network before you touch real data.
  - RealCurveDataset: loads your actual experimental strain-population
    time series from CSV. NEEDS YOUR INPUT — see the class docstring for
    the exact format expected and what to change.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .physics import simulate_monod


class SyntheticMonodDataset(Dataset):
    """
    Generates synthetic Monod growth curves with randomly sampled
    parameters, for pretraining or unit testing.

    NEEDS YOUR INPUT: the sampling ranges below (`mu_max_range`, etc.)
    are arbitrary placeholders. Set them to plausible ranges for your
    organism/strain if you want the synthetic pretraining distribution to
    resemble your real data.
    """

    def __init__(
        self,
        n_curves: int = 1000,
        t_max: float = 24.0,
        n_timepoints: int = 100,
        mu_max_range: tuple[float, float] = (0.1, 1.5),
        Ks_range: tuple[float, float] = (0.05, 2.0),
        Y_range: tuple[float, float] = (0.2, 0.8),
        X0_range: tuple[float, float] = (0.01, 0.1),
        S0_range: tuple[float, float] = (1.0, 10.0),
        noise_std: float = 0.02,
        seed: int | None = 0,
    ):
        rng = np.random.default_rng(seed)
        self.t = torch.linspace(0, t_max, n_timepoints)

        mu_max = rng.uniform(*mu_max_range, size=n_curves)
        Ks = rng.uniform(*Ks_range, size=n_curves)
        Y = rng.uniform(*Y_range, size=n_curves)
        X0 = rng.uniform(*X0_range, size=n_curves)
        S0 = rng.uniform(*S0_range, size=n_curves)

        params = torch.tensor(np.stack([mu_max, Ks, Y], axis=1), dtype=torch.float32)
        y0 = torch.tensor(np.stack([X0, S0], axis=1), dtype=torch.float32)

        with torch.no_grad():
            traj = simulate_monod(params, y0, self.t)  # (n_curves, T, 2)

        # add measurement noise to the population channel to mimic real data
        noise = torch.randn_like(traj[..., 0]) * noise_std * traj[..., 0].std()
        traj_noisy = traj.clone()
        traj_noisy[..., 0] = traj[..., 0] + noise

        self.params = params
        self.y0 = y0
        self.traj_clean = traj
        self.traj_noisy = traj_noisy

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        # input to the CNN: (channels=1, T) — just the noisy population curve
        x = self.traj_noisy[idx, :, 0].unsqueeze(0)  # (1, T)
        return {
            "x": x,
            "y0": self.y0[idx],
            "t": self.t,
            "observed_traj": self.traj_noisy[idx],
            "true_params": self.params[idx],
        }


class RealCurveDataset(Dataset):
    """
    Loads real experimental strain-population time series from a CSV file.

    # NEEDS YOUR INPUT — expected CSV format (edit this class if yours
    # differs; see data/README.md for a written-out spec):
    #
    #   curve_id, t, X, S, X0, S0
    #   run1,     0, 0.02, 5.0, 0.02, 5.0
    #   run1,     1, 0.05, 4.8, 0.02, 5.0
    #   ...
    #   run2,     0, 0.01, 8.0, 0.01, 8.0
    #   ...
    #
    # - curve_id: groups rows belonging to the same experimental run.
    # - t: time (any consistent unit, e.g. hours), must be uniformly spaced
    #      WITHIN a curve_id (required by the RK4 integrator in physics.py).
    # - X: measured population/biomass at time t. Required.
    # - S: measured substrate concentration at time t. Optional — fill
    #      with NaN or 0 if you don't measure it; set `has_substrate=False`.
    # - X0, S0: initial conditions (usually just the t=0 row's X, S,
    #      repeated on every row for convenience) — required so physics.py
    #      knows where to start simulating from.
    #
    # If your real CSV has different column names, change COLUMN NAMES
    # just below, or rename your columns before loading.
    """

    # NEEDS YOUR INPUT: adjust to your actual column names
    COL_ID, COL_T, COL_X, COL_S, COL_X0, COL_S0 = (
        "curve_id", "t", "X", "S", "X0", "S0",
    )

    def __init__(self, csv_path: str, has_substrate: bool = False):
        df = pd.read_csv(csv_path)
        self.has_substrate = has_substrate
        self.curves = []

        for curve_id, group in df.groupby(self.COL_ID):
            group = group.sort_values(self.COL_T)
            t = torch.tensor(group[self.COL_T].values, dtype=torch.float32)
            X = torch.tensor(group[self.COL_X].values, dtype=torch.float32)
            if has_substrate:
                S = torch.tensor(group[self.COL_S].values, dtype=torch.float32)
            else:
                S = torch.zeros_like(X)  # placeholder, not used in loss by default

            X0 = float(group[self.COL_X0].iloc[0])
            S0 = float(group[self.COL_S0].iloc[0]) if has_substrate else 1.0

            self.curves.append({
                "t": t,
                "traj": torch.stack([X, S], dim=-1),
                "y0": torch.tensor([X0, S0], dtype=torch.float32),
            })

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        c = self.curves[idx]
        x = c["traj"][:, 0].unsqueeze(0)  # (1, T) population channel only
        return {
            "x": x,
            "y0": c["y0"],
            "t": c["t"],
            "observed_traj": c["traj"],
            "true_params": None,  # unknown for real data
        }
