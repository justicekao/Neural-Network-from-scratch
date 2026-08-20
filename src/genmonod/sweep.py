"""
sweep.py

Two things:
  1. A generic way to SET one parameter by name (works for matrix
     entries like r/K/c/mortality/mutation/translocation, AND for
     conjugative_transfer/production list entries) -- `set_param`. Use
     this plus `physics.simulate` directly for "give me a trajectory
     from these specific parameter values":

        set_param(cfg, "r", (0, 0), 0.5)
        set_param(cfg, "mortality", 0, 0.02)
        traj = simulate(cfg, y0, t)

  2. A 2D PARAMETER SWEEP: run many trajectories varying TWO parameters
     across a grid, for exploring how a system's behavior depends on
     them -- `run_sweep`, plus `heatmap` (classify each grid cell by
     some criteria, e.g. red/black for "did X happen") and
     `trajectory_grid` (show the actual time-series for each cell).

This is for EXPLORING a system's behavior across parameter space, not
fitting one to data -- nothing here touches Dataset or least_squares.

NEEDS YOUR INPUT: nothing to use the defaults, but `store_trajectories`
and grid size matter for memory/runtime — see run_sweep's docstring.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from .config import SystemConfig
from .physics import simulate

# matrix-valued attributes settable via set_param's (i,) or (i,j) index form
_MATRIX_ATTRS = ["r", "K", "c", "mortality", "mutation", "translocation",
                  "metabolite_supply", "metabolite_dilution", "toxin_supply", "toxin_decay"]
# list-of-dict attributes settable via set_param's (entry_index, field_name) index form
_LIST_ATTRS = ["conjugative_transfer", "production"]


def set_param(cfg: SystemConfig, attr: str, index, value: float) -> None:
    """
    Sets ONE parameter to a specific value, in place. Works for:
      - matrix attrs (r, K, c, mortality, mutation, translocation,
        metabolite_supply, metabolite_dilution, toxin_supply,
        toxin_decay): index is an int (1D matrices like mortality) or a
        (row, col) tuple (2D matrices like r/K/c).
      - conjugative_transfer / production (lists of dicts): index is
        (entry_position, field_name), e.g. (0, "rate") for the first
        registered entry's rate.

    NEEDS YOUR INPUT: nothing for existing attrs. If you add a new
    matrix to config.py, add its name to _MATRIX_ATTRS above.
    """
    if attr in _MATRIX_ATTRS:
        spec = getattr(cfg, attr)
        if spec is None:
            raise ValueError(f"cfg.{attr} is not active in this config (check dimensions/toggles)")
        spec.values[index] = value
    elif attr in _LIST_ATTRS:
        entries = getattr(cfg, attr)
        entry_idx, field = index
        entries[entry_idx][field] = value
    else:
        raise ValueError(f"'{attr}' is not a settable parameter (expected one of {_MATRIX_ATTRS + _LIST_ATTRS})")


@dataclass
class ParamAxis:
    """One axis of a sweep: which parameter, what values to try over it."""
    attr: str
    index: object          # int, (row, col), or (entry_index, field_name) -- see set_param
    values: np.ndarray
    label: str | None = None  # defaults to "attr[index]" if not given

    def display_label(self) -> str:
        return self.label or f"{self.attr}[{self.index}]"


@dataclass
class SweepResult:
    x_axis: ParamAxis
    y_axis: ParamAxis
    t: np.ndarray
    state_names: list[str]
    trajectories: np.ndarray | None  # (len(x), len(y), len(t), n_states) or None if not stored
    failed: np.ndarray               # (len(x), len(y)) bool -- True where integration failed (see run_sweep)


def run_sweep(
    cfg: SystemConfig, y0: np.ndarray, t: np.ndarray,
    x_axis: ParamAxis, y_axis: ParamAxis,
    store_trajectories: bool = True,
) -> SweepResult:
    """
    Runs one simulation per (x, y) combination in the grid, varying
    `x_axis`/`y_axis`'s parameter while everything else in `cfg` stays
    at whatever you already set (fix everything you're not sweeping to
    a specific value first — a NaN/free entry has no meaning here, this
    is simulation, not fitting).

    Args:
        store_trajectories: keep every full trajectory (needed for
            `trajectory_grid`, and for any heatmap criteria that needs
            the full time series). Set False for large grids where you
            only need a heatmap criteria that can be computed from a
            SUMMARY instead — for a 50x50 grid with 200 timepoints and
            5 states that's 50*50*200*5*8 bytes = 20MB, fine; a much
            larger grid or longer/more-state system could add up.
            When False, `heatmap`'s criteria_fn only receives the FINAL
            state, not the full trajectory (see heatmap's docstring).

    A cell where integration fails (see physics.simulate) is recorded in
    `.failed` rather than raising — one bad parameter combination
    shouldn't crash an entire sweep.
    """
    import copy

    nx, ny = len(x_axis.values), len(y_axis.values)
    n_states = cfg.n_strains + cfg.n_metabolites + cfg.n_toxins
    state_names = cfg.strain_names + cfg.combined_substrate_names()

    trajectories = np.full((nx, ny, len(t), n_states), np.nan) if store_trajectories else None
    final_states = np.full((nx, ny, n_states), np.nan)
    failed = np.zeros((nx, ny), dtype=bool)

    for i, xv in enumerate(x_axis.values):
        for j, yv in enumerate(y_axis.values):
            trial_cfg = copy.deepcopy(cfg)
            set_param(trial_cfg, x_axis.attr, x_axis.index, xv)
            set_param(trial_cfg, y_axis.attr, y_axis.index, yv)
            try:
                traj = simulate(trial_cfg, y0, t)
                if not np.all(np.isfinite(traj)):
                    raise RuntimeError("non-finite trajectory")
            except RuntimeError:
                failed[i, j] = True
                continue
            if store_trajectories:
                trajectories[i, j] = traj
            final_states[i, j] = traj[-1]

    result = SweepResult(x_axis=x_axis, y_axis=y_axis, t=t, state_names=state_names,
                          trajectories=trajectories, failed=failed)
    result.final_states = final_states  # attached for use by heatmap when trajectories aren't stored
    return result


def heatmap(
    sweep: SweepResult,
    criteria_fn: Callable,
    ax=None,
    cmap_true: str = "#c0392b", cmap_false: str = "#1a1a1a",
    title: str | None = None,
):
    """
    Colors each (x, y) grid cell by `criteria_fn`, red/black by default
    for a boolean criteria (matching "red vs black" — pass different
    colors via cmap_true/cmap_false for anything else) — or pass a
    criteria_fn returning a float for a continuous colormap instead.

    Args:
        criteria_fn: called as `criteria_fn(traj, t, state_names)` if
            sweep was run with store_trajectories=True (traj is the full
            (len(t), n_states) array for that cell), or as
            `criteria_fn(final_state, state_names)` if trajectories
            weren't stored (final_state is just the (n_states,) array at
            the last timepoint) — write your function to match how you
            ran the sweep. Return True/False for red/black, or a float
            for a continuous colormap.
            Example: `lambda traj, t, names: traj[-1, names.index("Transconjugant")] > 1e4`

    Failed cells (see run_sweep) are shown in white/hatched regardless
    of criteria_fn, since there's no trajectory to evaluate there.

    Returns: the matplotlib Figure (or reuses `ax`'s figure if given).
    """
    nx, ny = len(sweep.x_axis.values), len(sweep.y_axis.values)
    grid = np.full((ny, nx), np.nan)  # imshow wants (rows=y, cols=x)
    is_bool = True

    for i in range(nx):
        for j in range(ny):
            if sweep.failed[i, j]:
                continue
            if sweep.trajectories is not None:
                val = criteria_fn(sweep.trajectories[i, j], sweep.t, sweep.state_names)
            else:
                val = criteria_fn(sweep.final_states[i, j], sweep.state_names)
            if not isinstance(val, (bool, np.bool_)):
                is_bool = False
            grid[j, i] = float(val)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if is_bool:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([cmap_false, cmap_true])
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                        extent=[sweep.x_axis.values[0], sweep.x_axis.values[-1],
                                sweep.y_axis.values[0], sweep.y_axis.values[-1]])
    else:
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                        extent=[sweep.x_axis.values[0], sweep.x_axis.values[-1],
                                sweep.y_axis.values[0], sweep.y_axis.values[-1]])
        fig.colorbar(im, ax=ax)

    # mark failed cells with an X so they're visually distinct from a
    # real (if extreme) criteria result
    fail_i, fail_j = np.where(sweep.failed)
    for i, j in zip(fail_i, fail_j):
        ax.scatter(sweep.x_axis.values[i], sweep.y_axis.values[j], marker="x", color="white", s=40, zorder=5)

    ax.set_xlabel(sweep.x_axis.display_label())
    ax.set_ylabel(sweep.y_axis.display_label())
    ax.set_title(title or "Parameter sweep")
    fig.tight_layout()
    return fig


def trajectory_grid(
    sweep: SweepResult,
    state_indices: list[int] | None = None,
    figsize_per_cell: float = 2.0,
    log_y: bool = True,
):
    """
    Plots a grid of small time-series subplots, one per (x, y) cell —
    requires the sweep to have been run with store_trajectories=True.
    Best for modest grids (roughly up to ~6x6); a heatmap (above) scales
    to much larger grids since it only needs one number per cell.

    Args:
        state_indices: which states to plot in each subplot (default:
            all of them). Use `sweep.state_names` to find indices.

    Returns: the matplotlib Figure.
    """
    if sweep.trajectories is None:
        raise ValueError("trajectory_grid needs a sweep run with store_trajectories=True")

    nx, ny = len(sweep.x_axis.values), len(sweep.y_axis.values)
    state_indices = state_indices if state_indices is not None else list(range(len(sweep.state_names)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(state_indices)))

    fig, axes = plt.subplots(ny, nx, figsize=(figsize_per_cell * nx, figsize_per_cell * ny),
                              sharex=True, sharey=True, squeeze=False)
    for i, xv in enumerate(sweep.x_axis.values):
        for j, yv in enumerate(sweep.y_axis.values):
            ax = axes[ny - 1 - j][i]  # row 0 at top -> put smallest y at the bottom, like a normal plot
            if sweep.failed[i, j]:
                ax.text(0.5, 0.5, "failed", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="red")
            else:
                for si, c in zip(state_indices, colors):
                    ax.plot(sweep.t, sweep.trajectories[i, j, :, si], color=c, linewidth=1)
                if log_y:
                    ax.set_yscale("log")
            if j == 0:
                ax.set_xlabel(f"{xv:.3g}", fontsize=8)
            if i == 0:
                ax.set_ylabel(f"{yv:.3g}", fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"{sweep.x_axis.display_label()} (columns)  x  {sweep.y_axis.display_label()} (rows)", fontsize=10)
    handles = [plt.Line2D([0], [0], color=c, label=sweep.state_names[si]) for si, c in zip(state_indices, colors)]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
