"""
tests/test_physics.py

Run with `pytest` from the project root. NEEDS YOUR INPUT: nothing, but
add tests here if you extend physics.py with a new process term.
"""

import numpy as np
from genmonod.config import default_config
from genmonod.physics import simulate


def _fill_free(cfg, value=0.3, mortality=0.02):
    """Helper: fill every free (NaN) entry with a constant, for tests that
    don't care about fitting, just about the ODE being sane. Mortality
    defaults much lower than the other rates so growth isn't canceled out
    by construction (a rate exactly equal to mortality gives ~0 net
    growth, which isn't useful for a "does it grow" test)."""
    for attr in ["growth_rate", "growth_half_sat", "consumption",
                 "toxin_kill_rate", "toxin_half_sat", "secretion",
                 "mutation", "translocation",
                 "metabolite_supply", "metabolite_dilution",
                 "toxin_supply", "toxin_decay"]:
        spec = getattr(cfg, attr)
        if spec is not None:
            spec.values[np.isnan(spec.values)] = value
    if cfg.mortality is not None:
        cfg.mortality.values[np.isnan(cfg.mortality.values)] = mortality
    return cfg


def test_single_strain_grows_with_metabolite():
    cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    cfg = _fill_free(cfg, 0.3)
    y0 = np.array([0.1, 10.0])
    t = np.linspace(0, 20, 50)
    traj = simulate(cfg, y0, t)
    assert traj[-1, 0] > traj[0, 0]


def test_toxin_suppresses_growth():
    cfg_no_tox = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    cfg_no_tox = _fill_free(cfg_no_tox, 0.3)

    cfg_tox = default_config(n_strains=1, n_metabolites=1, n_toxins=1)
    cfg_tox = _fill_free(cfg_tox, 0.3)

    t = np.linspace(0, 20, 50)
    traj_no_tox = simulate(cfg_no_tox, np.array([0.1, 10.0]), t)
    traj_tox = simulate(cfg_tox, np.array([0.1, 10.0, 5.0]), t)

    assert traj_tox[-1, 0] < traj_no_tox[-1, 0]


def test_mutation_transfers_population():
    """With mutation on and strain 1 -> strain 2 rate high, strain 2 should
    end up nonzero even if it starts at (near) zero."""
    cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=0, include_mutation=True)
    cfg = _fill_free(cfg, 0.2)
    cfg.mutation.values[:, :] = 0.0
    cfg.mutation.values[1, 0] = 0.5  # strain 0 -> strain 1 at rate 0.5

    y0 = np.array([1.0, 1e-6, 10.0])
    t = np.linspace(0, 10, 30)
    traj = simulate(cfg, y0, t)
    assert traj[-1, 1] > y0[1]


def test_translocation_matrix_is_general_not_hardcoded():
    """Regression test for the original MATLAB bug: translocation should
    work for ANY strain pair, not just a hardcoded (3,1) special case."""
    cfg = default_config(n_strains=4, n_metabolites=1, n_toxins=0, include_translocation=True)
    cfg = _fill_free(cfg, 0.2)
    cfg.translocation.values[:, :] = 0.0
    cfg.translocation.values[3, 0] = 0.4  # strain 3 converts strain 0 on contact

    y0 = np.array([1.0, 1.0, 1.0, 1e-6, 10.0])
    t = np.linspace(0, 10, 30)
    traj = simulate(cfg, y0, t)
    assert traj[-1, 3] > y0[3]
