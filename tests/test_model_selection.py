"""
tests/test_model_selection.py

Run with `pytest` from the project root. NEEDS YOUR INPUT: nothing.
"""

import numpy as np
from genmonod.config import default_config
from genmonod.physics import simulate
from genmonod.data_io import Dataset
from genmonod.model_selection import compare_structures, build_structure_grid


def test_build_structure_grid_covers_all_combinations():
    grid = build_structure_grid(n_strains=2, include_toxin=(False, True), metabolic_overlap=("shared", "separate"))
    assert len(grid) == 4
    names = {c.name for c in grid}
    assert "shared-metabolite" in names
    assert "shared-metabolite+toxin" in names
    assert "separate-metabolite" in names
    assert "separate-metabolite+toxin" in names


def test_compare_structures_recovers_true_structure():
    """Generate data from a NO-toxin, shared-metabolite system, then check
    that compare_structures ranks that structure at or near the top --
    i.e. it should not prefer the toxin variant when there's no toxin
    signal in the data at all."""
    true_cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=0)
    true_cfg.r.values[:] = [[0.5], [0.3]]
    true_cfg.K.values[:] = 0.5
    true_cfg.c.values[:] = 0.0
    true_cfg.mortality.values[:] = 0.05
    true_cfg.metabolite_supply.values[:] = 0.0
    true_cfg.metabolite_dilution.values[:] = 0.0

    y0 = np.array([0.1, 0.1, 5.0])
    t = np.linspace(0, 15, 20)
    traj = simulate(true_cfg, y0, t)
    # add realistic measurement noise -- with PERFECTLY noiseless synthetic
    # data, RSS can be driven to ~0 by either structure, and AIC's log(RSS)
    # term becomes unstable/dominated by floating-point noise as RSS -> 0,
    # letting the more complex (toxin) structure spuriously "win" by
    # exploiting its extra free parameters to shave residual below machine
    # precision rather than genuinely explaining more signal. Real data
    # always has a noise floor; this test needs one too to be meaningful.
    rng = np.random.default_rng(0)
    traj = traj * (1 + rng.normal(0, 0.03, traj.shape))

    def builder(cfg):
        # must match EACH candidate's own shape -- a toxin candidate needs
        # an extra (entirely unmeasured/NaN) column, exactly like a real
        # Dataset.from_csv would produce for an unmapped state
        n_extra = cfg.n_toxins
        if n_extra == 0:
            return Dataset(t=t, Y=traj, y0=y0, name="synthetic_no_toxin")
        Y_padded = np.column_stack([traj, np.full((len(t), n_extra), np.nan)])
        y0_padded = np.concatenate([y0, np.ones(n_extra)])
        return Dataset(t=t, Y=Y_padded, y0=y0_padded, name="synthetic_no_toxin")

    results = compare_structures(
        builder, n_strains=2, include_toxin=(False, True), metabolic_overlap=("shared",),
        max_nfev=100,
    )
    assert len(results) == 2
    best = results[0]
    assert best.name == "shared-metabolite"  # the no-toxin structure should win
    assert results[0].aicc <= results[1].aicc


def test_reliability_flag_fires_for_small_data():
    """A dataset with very few points relative to free parameters should
    be flagged as unreliable, not silently ranked as if trustworthy."""
    y0 = np.array([0.1, 5.0])
    t = np.linspace(0, 5, 2)  # only 2 timepoints -> very few observations
    true_cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
    true_cfg.r.values[:] = 0.3
    true_cfg.K.values[:] = 0.5
    true_cfg.c.values[:] = 0.0
    true_cfg.mortality.values[:] = 0.05
    true_cfg.metabolite_supply.values[:] = 0.0
    true_cfg.metabolite_dilution.values[:] = 0.0
    traj = simulate(true_cfg, y0, t)

    def builder(cfg):
        return Dataset(t=t, Y=traj, y0=y0, name="tiny")

    results = compare_structures(builder, n_strains=1, include_toxin=(False, True), metabolic_overlap=("shared",), max_nfev=50)
    assert any(not r.reliable for r in results)
