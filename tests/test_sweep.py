"""
tests/test_sweep.py

Run with `pytest` from the project root. NEEDS YOUR INPUT: nothing.
"""

import numpy as np
from genmonod.config import default_config
from genmonod.physics import simulate
from genmonod.sweep import set_param, ParamAxis, run_sweep, heatmap, trajectory_grid


def test_set_param_matrix_entry():
    cfg = default_config(1, 1, 0)
    set_param(cfg, "r", (0, 0), 0.7)
    set_param(cfg, "mortality", 0, 0.03)
    assert cfg.r.values[0, 0] == 0.7
    assert cfg.mortality.values[0] == 0.03


def test_set_param_list_entry():
    from genmonod.config import add_conjugative_transfer
    cfg = default_config(3, 1, 0)
    cfg.strain_names = ["Recipient", "Donor", "Transconjugant"]
    add_conjugative_transfer(cfg, "Transconjugant", "Donor", "Recipient")
    set_param(cfg, "conjugative_transfer", (0, "rate"), 1e-9)
    assert cfg.conjugative_transfer[0]["rate"] == 1e-9


def test_sweep_reproduces_known_survival_boundary():
    """A 1-strain system should survive (grow) exactly where growth rate
    exceeds mortality, and go extinct otherwise -- verify the sweep
    correctly reproduces this simple, known boundary."""
    cfg = default_config(1, 1, 0)
    cfg.K.values[:] = 2.0
    cfg.c.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0

    y0 = np.array([1.0, 100.0])  # abundant metabolite so growth ~ saturates near r
    t = np.linspace(0, 10, 20)

    x_axis = ParamAxis(attr="r", index=(0, 0), values=np.array([0.1, 0.9]))
    y_axis = ParamAxis(attr="mortality", index=0, values=np.array([0.05, 0.5]))

    result = run_sweep(cfg, y0, t, x_axis, y_axis)
    assert not result.failed.any()
    assert result.trajectories.shape == (2, 2, 20, 2)

    def survives(traj, t, names):
        return traj[-1, 0] > traj[0, 0]

    # low r (0.1), high mortality (0.5) -> should NOT survive
    assert not survives(result.trajectories[0, 1], t, result.state_names)
    # high r (0.9), low mortality (0.05) -> should survive
    assert survives(result.trajectories[1, 0], t, result.state_names)


def test_sweep_failed_cells_do_not_crash():
    """A pathological parameter combination should be caught and recorded
    in .failed, not raise and abort the whole sweep."""
    cfg = default_config(1, 1, 0)
    cfg.K.values[:] = 2.0
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0

    y0 = np.array([1.0, 10.0])
    t = np.linspace(0, 10, 10)
    x_axis = ParamAxis(attr="r", index=(0, 0), values=np.array([0.3, 0.5]))
    y_axis = ParamAxis(attr="K", index=(0, 0), values=np.array([1.0, 2.0]))

    result = run_sweep(cfg, y0, t, x_axis, y_axis)
    assert result.failed.shape == (2, 2)  # no crash, ran to completion


def test_heatmap_and_trajectory_grid_produce_figures():
    """Smoke test: both plotting functions should return a real figure
    without raising, for a small sweep."""
    cfg = default_config(1, 1, 0)
    cfg.K.values[:] = 2.0
    cfg.c.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    y0 = np.array([1.0, 10.0])
    t = np.linspace(0, 10, 15)
    x_axis = ParamAxis(attr="r", index=(0, 0), values=np.linspace(0.1, 0.9, 3))
    y_axis = ParamAxis(attr="mortality", index=0, values=np.linspace(0.05, 0.5, 3))
    result = run_sweep(cfg, y0, t, x_axis, y_axis)

    fig1 = heatmap(result, lambda traj, t, names: traj[-1, 0] > traj[0, 0])
    assert fig1 is not None
    fig2 = trajectory_grid(result)
    assert fig2 is not None
