"""
amortized_model.py

The "learns from collected fits" piece. A small MLP, trained per config
"shape" (same strain/metabolite/toxin counts and mutation/translocation
flags), that maps a dataset's summary statistics -> a predicted
free-parameter vector. Its only job is to give `fitting.fit()` a better
starting point than the bounds-midpoint default, so:
  - fits converge faster
  - fits are less likely to land in a bad local optimum
and both of those get better as more fits accumulate in the store for
that shape.

This is deliberately simple (not the full physics-informed CNN from the
other package) because the free-parameter vector's length and meaning
change with every config shape — a fixed architecture like a CNN over
raw time series doesn't transfer across shapes the way it can for a
single fixed Monod model. If you want the heavier PICNN-style approach
for a single fixed shape you use a lot, that's a good extension point;
see the note at the bottom of this file.

NEEDS YOUR INPUT: nothing to use the defaults. `min_examples_to_train`
is the main knob — with very few stored fits for a shape, training a
network is more likely to hurt than help, so guessing is skipped until
you have enough examples.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .fit_store import load_matching_records


class _GuesserNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def guess_initial_params(
    shape_signature: tuple,
    data_summary: list[float],
    n_free_params: int,
    store_path: str,
    min_examples_to_train: int = 8,
    epochs: int = 200,
) -> np.ndarray | None:
    """
    Returns a predicted initial-guess vector of length n_free_params, or
    None if there isn't enough stored history yet for this config shape
    (in which case fitting.fit() just falls back to its default guess).

    NEEDS YOUR INPUT: nothing to call this from the app — it's wired up
    for you in app.py. Call it directly only if you're scripting fits
    yourself outside the app.
    """
    records = load_matching_records(shape_signature, store_path)
    records = [r for r in records if len(r["x"]) == n_free_params]
    if len(records) < min_examples_to_train:
        return None

    X = np.array([r["data_summaries"][0] for r in records], dtype=np.float32)
    # data_summaries can vary in length across records if datasets had
    # different numbers of columns measured; guard against that by only
    # using records whose summary length matches the majority
    lengths = [len(x) for x in X]
    common_len = max(set(lengths), key=lengths.count)
    mask = [len(x) == common_len for x in X]
    X = np.array([x for x, m in zip(X, mask) if m], dtype=np.float32)
    y = np.array([r["x"] for r, m in zip(records, mask) if m], dtype=np.float32)
    if len(X) < min_examples_to_train:
        return None

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = _GuesserNet(in_dim=X.shape[1], out_dim=n_free_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    query = torch.tensor(np.array(data_summary, dtype=np.float32)).unsqueeze(0)
    if query.shape[1] != X.shape[1]:
        return None  # this dataset's summary shape doesn't match the training set

    model.eval()
    with torch.no_grad():
        pred = model(query).squeeze(0).numpy()
    return pred


# Extension point: once you have a config shape you use constantly (say,
# a fixed 2-strain + 1-metabolite + 1-toxin system), you could swap this
# per-shape MLP for something closer to the CNN-over-raw-time-series
# approach from the other package (monod_pinn), trained specifically for
# that one shape, for a stronger and more physics-informed guess.
