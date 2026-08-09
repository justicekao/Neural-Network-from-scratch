"""
losses.py

Combines two loss terms:
  1. Data loss: how well the simulated trajectory (using the CNN's
     predicted parameters) matches the observed population curve.
  2. Parameter loss (optional): if you have ground-truth parameters
     (only true for synthetic training data), penalize the raw
     parameter error directly. This is set to zero-weight by default
     for real data where you don't know the truth.

NEEDS YOUR INPUT: `physics_weight` and `param_weight` are the most
important hyperparameters in the whole project. Start with the defaults,
then adjust based on results:
  - If trajectories fit well but parameters look unreasonable given prior
    knowledge, increase param_weight (if you have synthetic labels) or add
    parameter bounds/regularization.
  - If training is unstable, lower physics_weight.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .physics import simulate_monod


def physics_informed_loss(
    predicted_params: torch.Tensor,
    observed_traj: torch.Tensor,
    y0: torch.Tensor,
    t: torch.Tensor,
    true_params: torch.Tensor | None = None,
    physics_weight: float = 1.0,
    param_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Args:
        predicted_params: (batch, 3) CNN output [mu_max, Ks, Y].
        observed_traj: (batch, T, 2) observed [X, S] (S can be zeros/NaN-free
            placeholder if you don't measure substrate — see dataset.py).
        y0: (batch, 2) initial condition used to simulate forward.
        t: (T,) shared time points.
        true_params: (batch, 3) ground-truth parameters, only available for
            synthetic data. Pass None for real data.
        physics_weight: weight on the trajectory-matching term.
        param_weight: weight on the direct parameter-matching term
            (only used if true_params is provided).

    Returns:
        total_loss: scalar tensor to call .backward() on.
        parts: dict of the individual (unweighted) loss components, useful
               for logging/plotting during training.
    """
    mse = nn.MSELoss()

    simulated = simulate_monod(predicted_params, y0, t)  # (batch, T, 2)

    # Trajectory / data loss — only compare the X (population) channel by
    # default, since that's what you said you always have. If you also
    # measure substrate S(t), change this to compare both channels.
    # NEEDS YOUR INPUT: switch to `mse(simulated, observed_traj)` if you
    # have real substrate measurements to fit against.
    traj_loss = mse(simulated[..., 0], observed_traj[..., 0])

    parts = {"trajectory_loss": traj_loss.detach()}
    total = physics_weight * traj_loss

    if true_params is not None and param_weight > 0:
        param_loss = mse(predicted_params, true_params)
        parts["parameter_loss"] = param_loss.detach()
        total = total + param_weight * param_loss

    parts["total_loss"] = total.detach()
    return total, parts
