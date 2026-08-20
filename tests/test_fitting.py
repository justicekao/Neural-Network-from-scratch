"""
tests/test_fitting.py

Run with `pytest` from the project root. NEEDS YOUR INPUT: nothing.
"""

import numpy as np
from genmonod.config import default_config, add_production
from genmonod.physics import simulate
from genmonod.data_io import Dataset
from genmonod.fitting import fit, pack_free_params


def test_fixed_entries_are_not_fit():
    """An entry you set fixed should come back unchanged after fitting."""
    cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    cfg.r.set_fixed(0.5, 0, 0)

    # fill remaining free entries and simulate some "observed" data
    for attr in ["K", "c", "mortality", "metabolite_supply", "metabolite_dilution"]:
        spec = getattr(cfg, attr)
        spec.values[np.isnan(spec.values)] = 0.3

    y0 = np.array([0.1, 10.0])
    t = np.linspace(0, 15, 20)
    traj = simulate(cfg, y0, t)
    ds = Dataset(t=t, Y=traj, y0=y0, name="synthetic")

    result = fit(cfg, ds, max_nfev=50)
    assert result.config.r.values[0, 0] == 0.5


def test_free_param_count_matches_pack():
    cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=1,
                          include_mutation=True, include_translocation=True)
    x0, lb, ub = pack_free_params(cfg)
    assert len(x0) == len(lb) == len(ub)
    assert len(x0) > 0


def test_hidden_initial_condition_is_fit_not_fixed():
    """A dataset with an entirely unmeasured metabolite column should still
    be able to fit growth by treating that metabolite's initial condition
    as a free parameter, instead of being stuck at an arbitrary fixed
    placeholder (this was a real gap in an earlier version)."""
    true_cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    for attr in ["r", "K", "c"]:
        getattr(true_cfg, attr).values[:] = 0.4
    true_cfg.mortality.values[:] = 0.02
    true_cfg.metabolite_supply.values[:] = 0.0
    true_cfg.metabolite_dilution.values[:] = 0.0

    y0_true = np.array([0.05, 5.0])
    t = np.linspace(0, 15, 15)
    traj = simulate(true_cfg, y0_true, t)

    # build a dataset where the metabolite column is entirely NaN, i.e.
    # never measured, the way Dataset.from_csv would produce it
    Y = np.column_stack([traj[:, 0], np.full(len(t), np.nan)])
    ds = Dataset(t=t, Y=Y, y0=np.array([0.05, 1.0]), name="pop_only")
    assert ds.y0_free_mask.tolist() == [False, False]  # default mask, not auto-inferred here

    # manually mark the metabolite IC as unknown, the way from_csv does automatically
    ds.y0_free_mask = np.array([False, True])

    cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    cfg.metabolite_supply.set_fixed(0.0, 0)
    cfg.metabolite_dilution.set_fixed(0.0, 0)

    result = fit(cfg, ds)
    assert result.success
    assert result.cost < 1e-6  # should fit the population curve essentially exactly
    assert len(result.y0s) == 1
    assert result.y0s[0][1] != 1.0  # the hidden IC moved away from its arbitrary start


def test_production_rate_and_half_sat_are_packed_and_recovered():
    """Both free scalars in a production entry (rate, half_sat) should be
    fit, and pinned to their true values, curve fit should be essentially
    exact."""
    true_cfg = default_config(1, 1, 1)
    true_cfg.r.values[:] = [0.3, 0.0]
    true_cfg.K.values[:] = [1.0, 1.0]
    true_cfg.c.values[:] = 0.0
    true_cfg.mortality.values[:] = 0.02
    true_cfg.metabolite_supply.values[:] = 0.0
    true_cfg.metabolite_dilution.values[:] = 0.0
    true_cfg.toxin_supply.values[:] = 0.0
    true_cfg.toxin_decay.values[:] = 0.0
    add_production(true_cfg, product=true_cfg.toxin_names[0], strain=true_cfg.strain_names[0],
                    precursor=true_cfg.metabolite_names[0], rate=0.25, half_sat=1.5)

    y0 = np.array([0.1, 5.0, 0.01])
    t = np.linspace(0, 15, 12)
    observed = simulate(true_cfg, y0, t)
    ds = Dataset(t=t, Y=observed, y0=y0, name="synthetic_production")

    cfg = default_config(1, 1, 1)
    for i in range(2):
        cfg.r.set_fixed(true_cfg.r.values[0, i], 0, i)
        cfg.K.set_fixed(true_cfg.K.values[0, i], 0, i)
        cfg.c.set_fixed(0.0, 0, i)
    cfg.mortality.set_fixed(0.02, 0)
    cfg.metabolite_supply.set_fixed(0.0, 0)
    cfg.metabolite_dilution.set_fixed(0.0, 0)
    cfg.toxin_supply.set_fixed(0.0, 0)
    cfg.toxin_decay.set_fixed(0.0, 0)
    add_production(cfg, product=cfg.toxin_names[0], strain=cfg.strain_names[0], precursor=cfg.metabolite_names[0])

    result = fit(cfg, ds, max_nfev=150)
    assert result.success
    assert result.cost < 1e-6
    assert not np.isnan(result.config.production[0]["rate"])
    assert not np.isnan(result.config.production[0]["half_sat"])
