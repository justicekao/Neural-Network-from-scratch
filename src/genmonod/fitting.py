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
    "growth_rate", "growth_half_sat", "consumption",
    "toxin_kill_rate", "toxin_half_sat", "secretion",
    "mortality", "mutation", "translocation",
    "metabolite_supply", "metabolite_dilution",
    "toxin_supply", "toxin_decay",
]


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
    row-major within each matrix) so pack/unpack always agree.
    """
    x0, lb, ub = [], [], []
    for _, spec in _active_matrices(cfg):
        free_mask = np.isnan(spec.values)
        n_free = int(free_mask.sum())
        mid = (spec.lower + spec.upper) / 2.0
        x0.extend([mid] * n_free)
        lb.extend([spec.lower] * n_free)
        ub.extend([spec.upper] * n_free)
    return np.array(x0), np.array(lb), np.array(ub)


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
    return resolved


@dataclass
class FitResult:
    config: SystemConfig       # fully resolved (no more free/NaN entries)
    success: bool
    cost: float
    x: np.ndarray               # raw fitted free-parameter vector
    datasets: list[Dataset]
    trajectories: list[np.ndarray]  # simulated trajectory per dataset, aligned to dataset.t


def _residuals(x: np.ndarray, cfg: SystemConfig, datasets: list[Dataset]) -> np.ndarray:
    resolved = unpack_into_config(cfg, x)
    all_res = []
    for ds in datasets:
        try:
            sim = simulate(resolved, ds.y0, ds.t)
        except RuntimeError:
            # integration blew up for this parameter set — penalize heavily
            # rather than crashing the whole fit
            all_res.append(np.full(ds.Y.size, 10.0))
            continue

        # compare in log10 space, normalized per-column by that column's
        # observed data range, and skip any (timepoint, state) with no
        # observation (NaN in ds.Y)
        obs = ds.Y
        log_sim = np.log10(np.maximum(sim, 1e-8))
        log_obs = np.log10(np.maximum(obs, 1e-8))
        col_range = np.nanmax(log_obs, axis=0) - np.nanmin(log_obs, axis=0)
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
    MatrixSpec's lower/upper before re-fitting.
    """
    if isinstance(datasets, Dataset):
        datasets = [datasets]

    x0, lb, ub = pack_free_params(cfg)
    if init_guess is not None and len(init_guess) == len(x0):
        x0 = np.clip(init_guess, lb, ub)

    result = least_squares(
        _residuals, x0, bounds=(lb, ub), args=(cfg, datasets),
        max_nfev=max_nfev, method="trf",
    )

    resolved = unpack_into_config(cfg, result.x)
    trajectories = [simulate(resolved, ds.y0, ds.t) for ds in datasets]

    return FitResult(
        config=resolved, success=result.success, cost=result.cost,
        x=result.x, datasets=datasets, trajectories=trajectories,
    )
