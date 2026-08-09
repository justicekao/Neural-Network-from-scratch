"""
evaluate.py

Plotting/evaluation helpers: run a trained model on a curve and compare
the simulated trajectory (using predicted parameters) against the observed
data. NEEDS YOUR INPUT: nothing required to run, but feel free to adjust
plot styling.
"""

from __future__ import annotations
import torch
import matplotlib.pyplot as plt

from .physics import simulate_monod


@torch.no_grad()
def predict_and_plot(model, sample: dict, device: str = "cpu", save_path: str | None = None):
    """
    Args:
        model: trained MonodCNN.
        sample: one item from a dataset (dict with x, y0, t, observed_traj).
        save_path: if given, saves the figure there instead of just showing it.

    Returns:
        predicted_params: (3,) tensor [mu_max, Ks, Y].
    """
    model.eval()
    x = sample["x"].unsqueeze(0).to(device)       # (1, 1, T)
    y0 = sample["y0"].unsqueeze(0).to(device)      # (1, 2)
    t = sample["t"].to(device)

    predicted_params = model(x)                    # (1, 3)
    simulated = simulate_monod(predicted_params, y0, t)[0]  # (T, 2)

    observed = sample["observed_traj"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t.cpu(), observed[:, 0].cpu(), "o", label="observed X(t)", markersize=3)
    ax.plot(t.cpu(), simulated[:, 0].cpu(), "-", label="fitted Monod curve")
    ax.set_xlabel("time")
    ax.set_ylabel("population X")
    mu, Ks, Y = predicted_params[0].cpu().tolist()
    ax.set_title(f"mu_max={mu:.3f}, Ks={Ks:.3f}, Y={Y:.3f}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return predicted_params[0].cpu()
