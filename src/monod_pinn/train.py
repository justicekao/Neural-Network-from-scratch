"""
train.py

The training loop. Import `train_model` from your own scripts, or just run
`scripts/run_training.py` from the command line (see README).

NEEDS YOUR INPUT: the default hyperparameters (lr, batch_size, epochs) are
reasonable starting points, not tuned values — expect to adjust them once
you're training on real data.
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import MonodCNN
from .losses import physics_informed_loss
from .utils import set_seed, save_checkpoint


def collate_fn(batch: list[dict]):
    """
    Custom collate function because `true_params` may be None (real data
    with unknown ground truth) and because curves can vary in length T.

    NEEDS YOUR INPUT: this assumes all samples in a single batch share the
    same T (number of time points) and the same t grid — true for
    SyntheticMonodDataset, and true for RealCurveDataset IF all your real
    curves were sampled on the same schedule. If your real curves have
    different lengths, set batch_size=1 when training on RealCurveDataset,
    or add padding/masking logic here.
    """
    x = torch.stack([b["x"] for b in batch])
    y0 = torch.stack([b["y0"] for b in batch])
    t = batch[0]["t"]  # assumed shared across the batch, see docstring above
    observed_traj = torch.stack([b["observed_traj"] for b in batch])
    if batch[0]["true_params"] is not None:
        true_params = torch.stack([b["true_params"] for b in batch])
    else:
        true_params = None
    return {"x": x, "y0": y0, "t": t, "observed_traj": observed_traj, "true_params": true_params}


def train_model(
    dataset,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    physics_weight: float = 1.0,
    param_weight: float = 0.1,
    in_channels: int = 1,
    device: str | None = None,
    checkpoint_path: str | None = None,
    seed: int = 0,
) -> MonodCNN:
    """
    Trains a MonodCNN on the given dataset and returns the trained model.

    Args:
        dataset: a SyntheticMonodDataset or RealCurveDataset (or your own,
            as long as it returns dicts shaped like theirs).
        param_weight: set > 0 only if your dataset provides true_params
            (i.e. synthetic data). Ignored automatically for real data.
        checkpoint_path: if given, saves the trained model here at the end.
    """
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = MonodCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            x = batch["x"].to(device)
            y0 = batch["y0"].to(device)
            t = batch["t"].to(device)
            observed_traj = batch["observed_traj"].to(device)
            true_params = batch["true_params"].to(device) if batch["true_params"] is not None else None

            optimizer.zero_grad()
            predicted_params = model(x)
            loss, parts = physics_informed_loss(
                predicted_params, observed_traj, y0, t,
                true_params=true_params,
                physics_weight=physics_weight,
                param_weight=param_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.shape[0]

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        print(f"epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    if checkpoint_path:
        save_checkpoint(model, checkpoint_path, epoch=epochs, history=history)
        print(f"saved checkpoint to {checkpoint_path}")

    return model
