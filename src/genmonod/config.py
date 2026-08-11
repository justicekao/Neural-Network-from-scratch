"""
config.py

Defines the "generalized Monod system": how many strains/metabolites/
toxins you have, which optional processes (mutation, translocation,
toxins) are switched on, and — for every parameter matrix — which
entries are FIXED at a known value and which are FREE to be fit.

This is the file that encodes "all optional inputs": call
`default_config(...)` with just a strain count and everything else
defaults to off/free. Then use `set_fixed(...)` to pin down anything you
already know.

NEEDS YOUR INPUT: nothing to run the defaults. You'll mostly interact
with this through the visual app (app.py), not by editing this file —
but if you're scripting fits directly, this is the object you build.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MatrixSpec:
    """
    One parameter matrix (e.g. growth rate r, shape strains x metabolites).

    `values` holds the CURRENT value for every entry:
      - np.nan  -> this entry is FREE, fit will estimate it
      - a float -> this entry is FIXED at that value during fitting

    `lower`/`upper` give the fit bounds applied to FREE entries (ignored
    for fixed ones). Same shape as `values`, or a single (lo, hi) pair
    applied to every free entry.
    """
    values: np.ndarray
    lower: float = 0.0
    upper: float = 10.0

    @property
    def shape(self):
        return self.values.shape

    def is_free(self, i: int, j: int | None = None) -> bool:
        idx = (i,) if j is None else (i, j)
        return bool(np.isnan(self.values[idx]))

    def set_fixed(self, value: float, i: int, j: int | None = None) -> None:
        idx = (i,) if j is None else (i, j)
        self.values[idx] = value

    def set_free(self, i: int, j: int | None = None) -> None:
        idx = (i,) if j is None else (i, j)
        self.values[idx] = np.nan


def _free_matrix(shape: tuple[int, ...]) -> np.ndarray:
    """All-NaN matrix = every entry free by default."""
    return np.full(shape, np.nan, dtype=float)


@dataclass
class SystemConfig:
    """
    The full specification of one generalized Monod system.

    NEEDS YOUR INPUT: build these with `default_config()` below rather
    than constructing directly, unless you're comfortable wiring up every
    MatrixSpec by hand.
    """
    n_strains: int
    n_metabolites: int = 0
    n_toxins: int = 0
    include_mutation: bool = False
    include_translocation: bool = False

    strain_names: list[str] = field(default_factory=list)
    metabolite_names: list[str] = field(default_factory=list)
    toxin_names: list[str] = field(default_factory=list)

    # growth term: dN_i grows via metabolite k with rate r[i,k], saturation k_half[i,k]
    growth_rate: MatrixSpec | None = None          # (S, M)
    growth_half_sat: MatrixSpec | None = None       # (S, M)
    consumption: MatrixSpec | None = None           # (S, M) - metabolite consumed per unit growth

    # toxin term: dN_i killed by toxin l with rate P[i,l], saturation K[i,l]
    toxin_kill_rate: MatrixSpec | None = None       # (S, T)
    toxin_half_sat: MatrixSpec | None = None        # (S, T)
    secretion: MatrixSpec | None = None             # (S, T) - toxin secreted per unit kill activity

    # per-strain terms
    mortality: MatrixSpec | None = None             # (S,)

    # strain-strain terms (both default OFF; only used if include_* is True)
    mutation: MatrixSpec | None = None               # (S, S) linear per-capita rate, entry [i,j] = j -> i rate
    translocation: MatrixSpec | None = None           # (S, S) contact-dependent rate, entry [i,j] = i converts j

    # environment terms
    metabolite_supply: MatrixSpec | None = None      # (M,)
    metabolite_dilution: MatrixSpec | None = None     # (M,)
    toxin_supply: MatrixSpec | None = None            # (T,)
    toxin_decay: MatrixSpec | None = None             # (T,)

    def shape_signature(self) -> tuple:
        """
        A hashable summary of this config's "shape" (dimensions + which
        optional terms are on) — used by fit_store.py / amortized_model.py
        to group past fits that can share an initial-guess network.
        """
        return (
            self.n_strains, self.n_metabolites, self.n_toxins,
            self.include_mutation, self.include_translocation,
        )


def default_config(
    n_strains: int,
    n_metabolites: int = 1,
    n_toxins: int = 0,
    include_mutation: bool = False,
    include_translocation: bool = False,
) -> SystemConfig:
    """
    Build a SystemConfig where every applicable matrix entry is FREE
    (fit will estimate it) with sensible default bounds. This is the
    normal entry point — every "constraint" is optional, so calling this
    with just a strain count gives you a fully-general, fully-free
    system to fit.

    NEEDS YOUR INPUT: nothing to call this. Use `set_fixed` afterward (or
    the visual app) to pin down anything you already know.
    """
    S, M, T = n_strains, n_metabolites, n_toxins

    cfg = SystemConfig(
        n_strains=S, n_metabolites=M, n_toxins=T,
        include_mutation=include_mutation,
        include_translocation=include_translocation,
        strain_names=[f"Strain_{i+1}" for i in range(S)],
        metabolite_names=[f"Metabolite_{k+1}" for k in range(M)],
        toxin_names=[f"Toxin_{l+1}" for l in range(T)],
    )

    if M > 0:
        cfg.growth_rate = MatrixSpec(_free_matrix((S, M)), 0.0, 2.0)
        cfg.growth_half_sat = MatrixSpec(_free_matrix((S, M)), 1e-3, 10.0)
        cfg.consumption = MatrixSpec(_free_matrix((S, M)), 0.0, 5.0)
        cfg.metabolite_supply = MatrixSpec(_free_matrix((M,)), 0.0, 5.0)
        cfg.metabolite_dilution = MatrixSpec(_free_matrix((M,)), 0.0, 2.0)

    if T > 0:
        cfg.toxin_kill_rate = MatrixSpec(_free_matrix((S, T)), 0.0, 2.0)
        cfg.toxin_half_sat = MatrixSpec(_free_matrix((S, T)), 1e-3, 10.0)
        cfg.secretion = MatrixSpec(_free_matrix((S, T)), 0.0, 2.0)
        cfg.toxin_supply = MatrixSpec(_free_matrix((T,)), 0.0, 2.0)
        cfg.toxin_decay = MatrixSpec(_free_matrix((T,)), 0.0, 2.0)

    cfg.mortality = MatrixSpec(_free_matrix((S,)), 0.0, 1.0)

    if include_mutation:
        m = _free_matrix((S, S))
        np.fill_diagonal(m, 0.0)  # diagonal is never free: no self-mutation
        cfg.mutation = MatrixSpec(m, 0.0, 1.0)

    if include_translocation:
        tl = _free_matrix((S, S))
        np.fill_diagonal(tl, 0.0)  # diagonal is never free: no self-translocation
        cfg.translocation = MatrixSpec(tl, 0.0, 1.0)

    return cfg
