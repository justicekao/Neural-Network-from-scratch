"""
fit_store.py

A simple append-only local store of past fits: config shape, the data
that was fit, and the resulting parameter vector. This is the training
data for amortized_model.py's "learns from collected fits" guesser.

NEEDS YOUR INPUT: nothing to run. Change `default_store_path` if you
want fits saved somewhere other than ./genmonod_fit_store.jsonl (the app
lets you set this too).
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os
import numpy as np

from .config import SystemConfig
from .fitting import FitResult, pack_free_params
from .data_io import Dataset

DEFAULT_STORE_PATH = "genmonod_fit_store.jsonl"


def _data_summary(ds: Dataset) -> list[float]:
    """
    A small fixed-length numeric summary of one dataset's shape, used as
    a feature vector for the amortized guesser. NEEDS YOUR INPUT:
    nothing, but feel free to add more summary statistics here if the
    guesser isn't learning well — e.g. per-column growth rate estimated
    by a simple linear fit to log-space early timepoints.
    """
    Y = ds.Y
    with np.errstate(invalid="ignore"):
        start = np.nanmin(Y, axis=0)
        end = np.nanmax(Y, axis=0)
        mean = np.nanmean(Y, axis=0)
    duration = float(ds.t[-1] - ds.t[0]) if len(ds.t) > 1 else 0.0
    # pad/flatten to a consistent-ish summary: [duration, then per-column start/end/mean]
    summary = [duration] + list(start) + list(end) + list(mean)
    return [float(v) if np.isfinite(v) else 0.0 for v in summary]


def record_fit(result: FitResult, store_path: str = DEFAULT_STORE_PATH) -> None:
    """Append this fit's config shape, data summary, and fitted x-vector to the store."""
    cfg = result.config
    record = {
        "shape_signature": list(cfg.shape_signature()),
        "x": result.x.tolist(),
        "cost": result.cost,
        "data_summaries": [_data_summary(ds) for ds in result.datasets],
    }
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    with open(store_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_matching_records(shape_signature: tuple, store_path: str = DEFAULT_STORE_PATH) -> list[dict]:
    """Load all past fit records whose config shape matches (same S, M, T, mutation/translocation flags)."""
    if not os.path.exists(store_path):
        return []
    matches = []
    with open(store_path) as f:
        for line in f:
            rec = json.loads(line)
            if tuple(rec["shape_signature"]) == tuple(shape_signature):
                matches.append(rec)
    return matches
