"""
physics.py

Defines the Monod growth-kinetics ODEs and a differentiable integrator.

NEEDS YOUR INPUT: if your "specific Monod model" isn't the plain two-state
model below (e.g. you have a lag phase, a death-rate term, substrate
inhibition, or multiple substrates), edit `monod_rhs` here. Everything else
in the package (CNN, losses, training) treats this file as the single
source of truth for the physics, so changes here propagate everywhere.

Why a custom RK4 instead of scipy.integrate.solve_ivp?
Because the physics-informed loss needs gradients to flow from the
simulated trajectory back into the CNN's predicted parameters (mu_max, Ks,
Y). scipy's solvers are not differentiable by PyTorch's autograd, so we
implement a small fixed-step RK4 integrator directly in torch. This is
plenty accurate for smooth Monod curves; if you need adaptive step sizes,
look into the `torchdiffeq` package instead.
"""

from __future__ import annotations
import torch


def monod_rhs(t: torch.Tensor, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Right-hand side of the Monod ODE system.

    Args:
        t: current time (unused here since the system is autonomous, but
           kept in the signature for compatibility with generic ODE solvers
           and in case you add a time-varying term, e.g. feeding/dilution).
        y: tensor of shape (..., 2) holding [X, S] (population, substrate).
        params: tensor of shape (..., 3) holding [mu_max, Ks, Y].

    Returns:
        dy/dt, same shape as y.

    # NEEDS YOUR INPUT if your model differs from plain single-substrate
    # Monod kinetics. Example extensions:
    #   - Add a death/decay term:      dX/dt -= k_d * X
    #   - Add substrate inhibition:    mu = mu_max * S / (Ks + S + S**2/Ki)
    #   - Add a lag phase:             gate mu by a time-dependent factor
    """
    X = y[..., 0]
    S = y[..., 1]
    mu_max = params[..., 0]
    Ks = params[..., 1]
    Y = params[..., 2]

    # small epsilon guards against divide-by-zero if S -> 0
    eps = 1e-8
    mu = mu_max * S / (Ks + S + eps)

    dX = mu * X
    dS = -(1.0 / (Y + eps)) * mu * X

    return torch.stack([dX, dS], dim=-1)


def rk4_step(t: torch.Tensor, y: torch.Tensor, dt: float, params: torch.Tensor) -> torch.Tensor:
    """One classic 4th-order Runge-Kutta step. NEEDS YOUR INPUT: nothing."""
    k1 = monod_rhs(t, y, params)
    k2 = monod_rhs(t + dt / 2, y + dt / 2 * k1, params)
    k3 = monod_rhs(t + dt / 2, y + dt / 2 * k2, params)
    k4 = monod_rhs(t + dt, y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_monod(
    params: torch.Tensor,
    y0: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Simulate the Monod ODE system forward in time for a batch of parameter
    sets, fully differentiably (safe to backprop through).

    Args:
        params: (batch, 3) tensor of [mu_max, Ks, Y] per sample.
        y0: (batch, 2) tensor of initial [X0, S0] per sample.
        t: (T,) 1D tensor of evaluation time points, assumed uniformly
           spaced and shared across the batch (typical for a fixed
           sampling protocol).

    Returns:
        traj: (batch, T, 2) tensor of simulated [X, S] at each time in t.

    NEEDS YOUR INPUT: nothing, unless your time points are not uniformly
    spaced per curve, in which case switch to a per-step dt inside the loop.
    """
    dt = float(t[1] - t[0])
    batch = params.shape[0]
    T = t.shape[0]

    traj = torch.empty(batch, T, 2, device=params.device, dtype=params.dtype)
    y = y0.clone()
    traj[:, 0, :] = y
    for i in range(1, T):
        y = rk4_step(t[i - 1], y, dt, params)
        traj[:, i, :] = y
    return traj
