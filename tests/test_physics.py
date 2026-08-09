"""
tests/test_physics.py

Sanity checks for the Monod ODE integrator. Run with `pytest` from the
project root. NEEDS YOUR INPUT: nothing, but add more tests here as you
extend physics.py (e.g. if you add a lag phase or death term).
"""

import torch
from monod_pinn.physics import simulate_monod


def test_population_grows_when_substrate_available():
    """With positive mu_max and ample substrate, population X should increase."""
    params = torch.tensor([[0.5, 0.5, 0.5]])  # [mu_max, Ks, Y]
    y0 = torch.tensor([[0.1, 10.0]])            # [X0, S0]
    t = torch.linspace(0, 10, 50)

    traj = simulate_monod(params, y0, t)
    X = traj[0, :, 0]

    assert X[-1] > X[0], "population should grow over time with ample substrate"


def test_substrate_is_consumed():
    """Substrate S should monotonically decrease as biomass grows."""
    params = torch.tensor([[0.5, 0.5, 0.5]])
    y0 = torch.tensor([[0.1, 10.0]])
    t = torch.linspace(0, 10, 50)

    traj = simulate_monod(params, y0, t)
    S = traj[0, :, 1]

    assert S[-1] < S[0], "substrate should be consumed over time"


def test_gradients_flow_to_parameters():
    """The physics-informed loss requires autograd through the integrator."""
    params = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    y0 = torch.tensor([[0.1, 10.0]])
    t = torch.linspace(0, 10, 20)

    traj = simulate_monod(params, y0, t)
    loss = traj[..., 0].sum()
    loss.backward()

    assert params.grad is not None
    assert torch.all(torch.isfinite(params.grad))
