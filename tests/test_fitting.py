"""
tests/test_fitting.py

Run with `pytest` from the project root. NEEDS YOUR INPUT: nothing.
"""

import numpy as np
from genmonod.config import default_config
from genmonod.physics import simulate
from genmonod.data_io import Dataset
from genmonod.fitting import fit, pack_free_params


def test_fixed_entries_are_not_fit():
    """An entry you set fixed should come back unchanged after fitting."""
    cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    cfg.growth_rate.set_fixed(0.5, 0, 0)

    # fill remaining free entries and simulate some "observed" data
    for attr in ["growth_half_sat", "consumption", "mortality",
                 "metabolite_supply", "metabolite_dilution"]:
        spec = getattr(cfg, attr)
        spec.values[np.isnan(spec.values)] = 0.3

    y0 = np.array([0.1, 10.0])
    t = np.linspace(0, 15, 20)
    traj = simulate(cfg, y0, t)
    ds = Dataset(t=t, Y=traj, y0=y0, name="synthetic")

    result = fit(cfg, ds, max_nfev=50)
    assert result.config.growth_rate.values[0, 0] == 0.5


def test_free_param_count_matches_pack():
    cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=1,
                          include_mutation=True, include_translocation=True)
    x0, lb, ub = pack_free_params(cfg)
    assert len(x0) == len(lb) == len(ub)
    assert len(x0) > 0
