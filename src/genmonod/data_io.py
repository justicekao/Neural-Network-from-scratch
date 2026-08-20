"""
data_io.py

Loads experimental time-series data and ties it to a SystemConfig via an
EXPLICIT column mapping — you say which CSV column is which strain /
metabolite / toxin, rather than relying on the column NAME containing a
recognizable keyword.

# WHAT WAS FIXED vs the original MATLAB version: it guessed which
# columns were strains/metabolites/toxins by searching column names for
# substrings like "Recipient", "Donor", "Nutrient", "Toxin" (with
# multiple follow-up patches to that regex as new column names came up).
# That's fragile — a column named anything unexpected silently falls
# through. Here you map columns to roles explicitly, once, and it's
# saved with the dataset.

NEEDS YOUR INPUT: nothing to run this — the visual app builds the
mapping interactively. If you're scripting instead, see
`Dataset.from_csv` below for the column_map format.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import SystemConfig


@dataclass
class Dataset:
    """
    One experiment's time series, already aligned to a SystemConfig's
    strain/metabolite/toxin ordering.

    t: (T_points,) time values.
    Y: (T_points, S+M+T) observed values, columns ordered
       [strains..., metabolites..., toxins...] matching cfg's ordering.
       Use np.nan for any (state, timepoint) you don't have a measurement
       for — those are simply skipped in the fit residual.
    y0: (S+M+T,) initial condition. Defaults to Y[0] if not given
        explicitly (useful when t=0 wasn't actually measured). For any
        entry where the true initial value is unknown, this is just a
        placeholder — see y0_free_mask.
    y0_free_mask: (S+M+T,) bool array. Where True, that entry of y0 is
        UNKNOWN and will be fit as a free parameter (bounded, see
        fitting.py's Y0_BOUNDS) rather than trusted as given. Defaults
        to wherever the original data had no measurement at t=0 — this
        is what makes it possible to fit growth on an unmeasured/hidden
        metabolite or toxin pool, instead of being stuck with it fixed
        at an arbitrary placeholder value.
    name: a label for this dataset, e.g. "Tube Run 1".

    # WHAT WAS FIXED: earlier versions of this package fixed any
    # unmeasured initial condition at a small constant (1e-3) — fine for
    # a measured state (real y0 from data) but silently wrong for a
    # truly unknown one (e.g. an unmeasured metabolite/toxin pool you
    # need to have a growth mechanism at all): a wrong fixed guess can't
    # be corrected by the fit no matter how good the rest of the model
    # is. Now those entries are fit, not assumed.
    """
    t: np.ndarray
    Y: np.ndarray
    y0: np.ndarray
    y0_free_mask: np.ndarray = None
    name: str = "dataset"

    def __post_init__(self):
        if self.y0_free_mask is None:
            self.y0_free_mask = np.zeros_like(self.y0, dtype=bool)

    @classmethod
    def from_csv(
        cls,
        path_or_buffer,
        time_col: str,
        column_map: dict[str, str],
        cfg: SystemConfig,
        name: str = "dataset",
    ) -> "Dataset":
        """
        Args:
            time_col: name of the time column in the CSV.
            column_map: maps cfg state names -> CSV column names, e.g.
                {"Strain_1": "OD_recipient", "Strain_2": "OD_donor",
                 "Metabolite_1": "Glucose_mM"}
                Any cfg state name NOT present in column_map is treated
                as entirely unmeasured (filled with NaN in Y), and its
                initial condition will be FIT rather than assumed (see
                Dataset.y0_free_mask above).
            cfg: the SystemConfig this dataset will be fit against —
                 used only to get the expected state ordering/names.
        """
        df = pd.read_csv(path_or_buffer)
        t = df[time_col].to_numpy(dtype=float)

        all_names = cfg.strain_names + cfg.metabolite_names + cfg.toxin_names
        Y = np.full((len(df), len(all_names)), np.nan)
        for j, state_name in enumerate(all_names):
            csv_col = column_map.get(state_name)
            if csv_col is not None and csv_col in df.columns:
                Y[:, j] = df[csv_col].to_numpy(dtype=float)

        y0_free_mask = np.isnan(Y[0])
        y0 = Y[0].copy()
        # placeholder only for display/standalone-simulate purposes —
        # fitting.py overrides any y0_free_mask=True entry with a fitted
        # value instead of trusting this
        y0 = np.where(y0_free_mask, 1.0, y0)

        return cls(t=t, Y=Y, y0=y0, y0_free_mask=y0_free_mask, name=name)
