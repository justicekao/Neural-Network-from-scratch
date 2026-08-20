"""
strain_library.py

The piece that lets you build up from small tractable fits to larger
systems. Every fit you save here records its parameter values keyed by
the actual STRAIN NAME (and metabolite/toxin name), not by config shape
— so if "Z1331" shows up in a 2-strain fit today and a 9-strain
community config next month, values learned about Z1331 the first time
can automatically be carried forward and FIXED in the new config,
leaving only genuinely new unknowns (like cross-strain interactions you
have no small-scale data for) free to fit.

This is separate from fit_store.py / amortized_model.py, which are
about SPEED (a better starting guess for a fit of the exact same shape).
This is about SCALE: shrinking how much a bigger system's fit actually
has to search, by not re-discovering things you already know.

NEEDS YOUR INPUT: nothing to use the defaults. The one real caveat:
this only works well if you reuse the SAME name for the same real
strain/metabolite/toxin across configs (e.g. always calling it "Z1331",
not "Strain_1" in one config and "Recipient" in another) — generic
default names like "Metabolite_1" will get treated as literally the
same substance across configs even if they're not, so rename anything
you want tracked meaningfully.
"""

from __future__ import annotations
import copy
import json
import os
import numpy as np

from .config import SystemConfig
from .fitting import FitResult, _MATRIX_ATTRS

DEFAULT_LIBRARY_PATH = "genmonod_strain_library.jsonl"

# which name-list(s) index the rows/columns of each matrix attribute —
# used to turn a (row, col) index pair into a real strain/metabolite/toxin
# NAME, which is the actual lookup key (not the row/col index, which isn't
# stable across configs of different sizes). "combined_substrate_names" is
# a METHOD (metabolites then toxins together — see config.py's r/K/c) —
# resolved specially below, not a plain attribute.
# NEEDS YOUR INPUT: only if you add a new matrix attribute in config.py —
# add its name-axis mapping here too so the library can key it correctly.
_AXES = {
    "r": ("strain_names", "combined_substrate_names"),
    "K": ("strain_names", "combined_substrate_names"),
    "c": ("strain_names", "combined_substrate_names"),
    "mortality": ("strain_names",),
    "mutation": ("strain_names", "strain_names"),
    "translocation": ("strain_names", "strain_names"),
    "metabolite_supply": ("metabolite_names",),
    "metabolite_dilution": ("metabolite_names",),
    "toxin_supply": ("toxin_names",),
    "toxin_decay": ("toxin_names",),
}


def _resolve_axis_names(cfg: SystemConfig, axis: str) -> list[str]:
    val = getattr(cfg, axis)
    return val() if callable(val) else val


def record_strain_params(result: FitResult, store_path: str = DEFAULT_LIBRARY_PATH) -> int:
    """
    Appends one record per resolved matrix entry to the library, keyed by
    the real strain/metabolite/toxin name(s) involved. Call this after a
    fit you trust (e.g. a small, well-identified system) to make its
    values available for auto-filling larger configs later.

    Returns the number of records written.
    """
    cfg = result.config
    n_written = 0
    with open(store_path, "a") as f:
        for attr in _MATRIX_ATTRS:
            spec = getattr(cfg, attr)
            if spec is None:
                continue
            axes = _AXES[attr]
            names_per_axis = [_resolve_axis_names(cfg, axis) for axis in axes]

            for idx in np.ndindex(spec.shape):
                if len(idx) == 2 and axes[0] == axes[1] and idx[0] == idx[1]:
                    continue  # skip diagonal of S x S matrices (never a real parameter)
                subject = [names_per_axis[a][idx[a]] for a in range(len(idx))]
                record = {
                    "kind": attr,
                    "subject": subject,
                    "value": float(spec.values[idx]),
                    "fit_cost": result.cost,
                }
                f.write(json.dumps(record) + "\n")
                n_written += 1
    return n_written


def _load_library(store_path: str) -> list[dict]:
    if not os.path.exists(store_path):
        return []
    with open(store_path) as f:
        return [json.loads(line) for line in f]


def apply_library(
    cfg: SystemConfig, store_path: str = DEFAULT_LIBRARY_PATH, aggregate: str = "median",
) -> tuple[SystemConfig, list[tuple]]:
    """
    Returns a NEW SystemConfig (cfg is left untouched) where every
    currently-FREE entry that has a matching record in the library
    (matched by real strain/metabolite/toxin name, not position) is
    FIXED at the aggregated value from past fits.

    Args:
        aggregate: "median" or "mean" across however many past records
            match a given entry.

    Returns:
        (new_cfg, filled): filled is a list of (attr, subject_names,
        value) for everything that got auto-filled, so you can see
        exactly what came from the library vs what's still genuinely
        unknown in this new, larger system.
    """
    records = _load_library(store_path)
    resolved = copy.deepcopy(cfg)
    filled = []

    for attr in _MATRIX_ATTRS:
        spec = getattr(resolved, attr)
        if spec is None:
            continue
        axes = _AXES[attr]
        names_per_axis = [_resolve_axis_names(resolved, axis) for axis in axes]
        matching = [r for r in records if r["kind"] == attr]
        if not matching:
            continue

        for idx in np.ndindex(spec.shape):
            if not np.isnan(spec.values[idx]):
                continue  # already fixed by the user — library never overrides an explicit choice
            if len(idx) == 2 and axes[0] == axes[1] and idx[0] == idx[1]:
                continue

            subject = [names_per_axis[a][idx[a]] for a in range(len(idx))]
            values = [r["value"] for r in matching if r["subject"] == subject]
            if not values:
                continue

            value = float(np.median(values)) if aggregate == "median" else float(np.mean(values))
            spec.values[idx] = value
            filled.append((attr, tuple(subject), value))

    return resolved, filled
