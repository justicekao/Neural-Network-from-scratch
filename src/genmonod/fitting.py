"""
fitting.py

Packs every FREE entry (see config.py's MatrixSpec) across all of a
SystemConfig's matrices into one flat vector, runs
scipy.optimize.least_squares against one or more Datasets, and unpacks
the result back into the matrices.

NEEDS YOUR INPUT: nothing to run a fit. `physics_weight`/solver options
have sane defaults; the main thing worth tuning per-project is
`init_guess` — see `fit()` below — since a good starting guess matters a
lot for these nonlinear least-squares problems.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .config import SystemConfig, MatrixSpec
from .physics import simulate
from .data_io import Dataset

# Every matrix attribute name on SystemConfig that fitting should consider.
# NEEDS YOUR INPUT: only if you add a brand new matrix to config.py — add
# its attribute name here so fitting.py knows to pack/unpack it too.
_MATRIX_ATTRS = [
    "r", "K", "c",
    "mortality", "mutation", "translocation",
    "metabolite_supply", "metabolite_dilution",
    "toxin_supply", "toxin_decay",
]

# Bounds used when fitting an UNKNOWN initial condition (Dataset.y0_free_mask
# entries). These are arbitrary units when the state was never measured in
# real units — what matters is the SHAPE of the resulting curve, not the
# absolute scale, since other free parameters (e.g. half-saturation, supply)
# can compensate for a rescaling. NEEDS YOUR INPUT: widen this if a fit keeps
# pushing a hidden initial condition to one of these bounds.
Y0_BOUNDS = (0.01, 100.0)


def _active_matrices(cfg: SystemConfig) -> list[tuple[str, MatrixSpec]]:
    """Matrices that actually exist for this config (some are None if e.g. n_toxins=0)."""
    out = []
    for attr in _MATRIX_ATTRS:
        spec = getattr(cfg, attr)
        if spec is not None:
            out.append((attr, spec))
    return out


def pack_free_params(cfg: SystemConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        x0: initial guess vector for every free entry (midpoint of its bounds)
        lb, ub: matching bound arrays for least_squares
    The order is deterministic (attribute order in _MATRIX_ATTRS, then
    row-major within each matrix, then any conjugative_transfer entries,
    then any production entries — rate then half_sat — in list order)
    so pack/unpack always agree.

    NOTE: this covers config matrix entries only. If you're fitting
    Datasets that have unmeasured (hidden) initial conditions, use
    `pack_free_params_with_y0` instead — that's what `fit()` uses
    internally.
    """
    x0, lb, ub = [], [], []
    for _, spec in _active_matrices(cfg):
        free_mask = np.isnan(spec.values)
        n_free = int(free_mask.sum())
        mid = (spec.lower + spec.upper) / 2.0
        x0.extend([mid] * n_free)
        lb.extend([spec.lower] * n_free)
        ub.extend([spec.upper] * n_free)
    for entry in (cfg.conjugative_transfer or []):
        if np.isnan(entry["rate"]):
            x0.append((entry["lower"] + entry["upper"]) / 2.0)
            lb.append(entry["lower"])
            ub.append(entry["upper"])
    for entry in (cfg.production or []):
        if np.isnan(entry["rate"]):
            x0.append((entry["rate_lower"] + entry["rate_upper"]) / 2.0)
            lb.append(entry["rate_lower"])
            ub.append(entry["rate_upper"])
        if np.isnan(entry["half_sat"]):
            x0.append((entry["half_sat_lower"] + entry["half_sat_upper"]) / 2.0)
            lb.append(entry["half_sat_lower"])
            ub.append(entry["half_sat_upper"])
    return np.array(x0), np.array(lb), np.array(ub)


def pack_free_params_with_y0(
    cfg: SystemConfig, datasets: list[Dataset]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Like pack_free_params, but appends one free entry per
    Dataset.y0_free_mask=True slot, across all datasets in order.

    Returns:
        x0, lb, ub: combined vector/bounds (config entries first, then
            each dataset's hidden y0 entries in order)
        n_cfg: how many of the leading entries belong to cfg (so you can
            split x back into "config part" and "y0 part")
    """
    x0_cfg, lb_cfg, ub_cfg = pack_free_params(cfg)
    n_cfg = len(x0_cfg)

    x0_y0, lb_y0, ub_y0 = [], [], []
    y0_lo, y0_hi = Y0_BOUNDS
    y0_mid = (y0_lo + y0_hi) / 2.0
    for ds in datasets:
        n_free = int(ds.y0_free_mask.sum())
        x0_y0.extend([y0_mid] * n_free)
        lb_y0.extend([y0_lo] * n_free)
        ub_y0.extend([y0_hi] * n_free)

    x0 = np.concatenate([x0_cfg, x0_y0])
    lb = np.concatenate([lb_cfg, lb_y0])
    ub = np.concatenate([ub_cfg, ub_y0])
    return x0, lb, ub, n_cfg


def unpack_into_config(cfg: SystemConfig, x: np.ndarray) -> SystemConfig:
    """
    Returns a NEW SystemConfig (same fixed entries, free entries filled
    in from x) — the original cfg's free/fixed pattern is left untouched
    so you can re-run fits without losing which entries you pinned.
    """
    import copy
    resolved = copy.deepcopy(cfg)
    idx = 0
    for attr, spec in _active_matrices(cfg):
        resolved_spec = getattr(resolved, attr)
        free_mask = np.isnan(spec.values)
        n_free = int(free_mask.sum())
        resolved_spec.values[free_mask] = x[idx: idx + n_free]
        idx += n_free
    for i, entry in enumerate(cfg.conjugative_transfer or []):
        if np.isnan(entry["rate"]):
            resolved.conjugative_transfer[i]["rate"] = float(x[idx])
            idx += 1
    for i, entry in enumerate(cfg.production or []):
        if np.isnan(entry["rate"]):
            resolved.production[i]["rate"] = float(x[idx])
            idx += 1
        if np.isnan(entry["half_sat"]):
            resolved.production[i]["half_sat"] = float(x[idx])
            idx += 1
    return resolved


def unpack_y0s(datasets: list[Dataset], x_y0: np.ndarray) -> list[np.ndarray]:
    """Splits the y0 portion of a combined parameter vector back into one
    resolved y0 array per dataset (fixed entries unchanged, free entries
    filled in from x_y0)."""
    y0s = []
    idx = 0
    for ds in datasets:
        y0 = ds.y0.copy()
        n_free = int(ds.y0_free_mask.sum())
        y0[ds.y0_free_mask] = x_y0[idx: idx + n_free]
        idx += n_free
        y0s.append(y0)
    return y0s


@dataclass
class FitResult:
    config: SystemConfig       # fully resolved (no more free/NaN entries)
    success: bool
    cost: float
    x: np.ndarray               # raw fitted free-parameter vector (config entries + hidden y0 entries)
    datasets: list[Dataset]
    trajectories: list[np.ndarray]  # simulated trajectory per dataset, aligned to dataset.t
    y0s: list[np.ndarray]        # resolved initial condition actually used per dataset (fixed + fitted-hidden)


def _residuals(x: np.ndarray, cfg: SystemConfig, datasets: list[Dataset], n_cfg: int) -> np.ndarray:
    resolved = unpack_into_config(cfg, x[:n_cfg])
    y0s = unpack_y0s(datasets, x[n_cfg:])
    all_res = []
    for ds, y0 in zip(datasets, y0s):
        try:
            sim = simulate(resolved, y0, ds.t)
            if not np.all(np.isfinite(sim)):
                raise RuntimeError("non-finite trajectory")
        except RuntimeError:
            # integration blew up (or produced inf/nan) for this parameter
            # set — penalize heavily rather than crashing the whole fit or
            # silently poisoning it with non-finite residuals
            all_res.append(np.full(ds.Y.size, 10.0))
            continue

        # compare in log10 space, normalized per-column by that column's
        # observed data range, and skip any (timepoint, state) with no
        # observation (NaN in ds.Y) — including columns that are ENTIRELY
        # unmeasured (e.g. a hidden metabolite), which would otherwise emit
        # a harmless-but-noisy "all-NaN slice" warning from nanmax/nanmin
        obs = ds.Y
        log_sim = np.log10(np.maximum(sim, 1e-8))
        log_obs = np.log10(np.maximum(obs, 1e-8))
        all_nan_col = np.all(np.isnan(obs), axis=0)
        with np.errstate(invalid="ignore"):
            col_range = np.where(
                all_nan_col, 1.0,
                np.nanmax(np.where(np.isnan(log_obs), -np.inf, log_obs), axis=0)
                - np.nanmin(np.where(np.isnan(log_obs), np.inf, log_obs), axis=0),
            )
        col_range = np.where(col_range < 1e-2, 1.0, col_range)

        err = (log_sim - log_obs) / col_range
        err = np.where(np.isnan(obs), 0.0, err)  # unmeasured entries contribute nothing
        all_res.append(err.ravel())
    return np.concatenate(all_res)


def fit(
    cfg: SystemConfig,
    datasets: Dataset | list[Dataset],
    init_guess: np.ndarray | None = None,
    max_nfev: int = 300,
) -> FitResult:
    """
    Fit cfg's free parameters against one or more datasets JOINTLY (all
    datasets share the same fitted matrices — pass a single Dataset for
    an isolated per-experiment fit instead).

    Args:
        init_guess: optional starting point for the free-parameter
            vector (e.g. from amortized_model.py's guesser). If omitted,
            starts at the midpoint of each parameter's bounds. A good
            init_guess matters a lot here — it's the single biggest
            lever on both fit quality and fit speed.

    NEEDS YOUR INPUT: `max_nfev` trades off fit thoroughness against
    runtime — each nonlinear-least-squares iteration needs roughly
    (n_free_params + 1) ODE simulations just to estimate the Jacobian by
    finite differences, so systems with 20+ free parameters can take
    real time (seconds to low minutes) even at this default. If a fit
    keeps hitting max_nfev without converging (`result.success` is
    False), raise this value; if fits are taking too long, either lower
    it or fix more matrix entries (fewer free parameters = faster). If
    fits are landing at their bounds a lot, widen the relevant
    MatrixSpec's lower/upper (or Y0_BOUNDS, for hidden initial
    conditions) before re-fitting.
    """
    if isinstance(datasets, Dataset):
        datasets = [datasets]

    x0, lb, ub, n_cfg = pack_free_params_with_y0(cfg, datasets)
    if init_guess is not None and len(init_guess) == len(x0):
        x0 = np.clip(init_guess, lb, ub)

    result = least_squares(
        _residuals, x0, bounds=(lb, ub), args=(cfg, datasets, n_cfg),
        max_nfev=max_nfev, method="trf",
    )

    resolved = unpack_into_config(cfg, result.x[:n_cfg])
    y0s = unpack_y0s(datasets, result.x[n_cfg:])
    trajectories = [simulate(resolved, y0, ds.t) for ds, y0 in zip(datasets, y0s)]

    return FitResult(
        config=resolved, success=result.success, cost=result.cost,
        x=result.x, datasets=datasets, trajectories=trajectories, y0s=y0s,
    )
