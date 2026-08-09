"""
examples/quickstart_example.py

The fastest way to see the whole pipeline work: generates a small synthetic
dataset, trains for a few epochs, and plots one fitted curve. Takes under a
minute on CPU.

Run with:
    python examples/quickstart_example.py

NEEDS YOUR INPUT: nothing, this is meant to run as-is as a sanity check
that your install works before you touch real data.
"""

import os

from monod_pinn.dataset import SyntheticMonodDataset
from monod_pinn.train import train_model
from monod_pinn.evaluate import predict_and_plot

# 1. Make a small synthetic dataset
dataset = SyntheticMonodDataset(n_curves=200, n_timepoints=60, seed=1)

# 2. Train a small model quickly (few epochs, just to prove the pipeline works)
model = train_model(
    dataset,
    epochs=15,
    batch_size=16,
    lr=1e-3,
    physics_weight=1.0,
    param_weight=0.1,  # we DO have ground truth here since it's synthetic
    checkpoint_path=None,
)

# 3. Evaluate on one example curve and save a plot
os.makedirs("outputs", exist_ok=True)
sample = dataset[0]
predicted = predict_and_plot(model, sample, save_path="outputs/quickstart_fit.png")

true_params = sample["true_params"]
print("True params  [mu_max, Ks, Y]:", true_params.tolist())
print("Predicted    [mu_max, Ks, Y]:", predicted.tolist())
print("Saved plot to outputs/quickstart_fit.png")
