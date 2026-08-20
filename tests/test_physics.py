"""
tests/test_physics.py

Run with `pytest` from the project root. NEEDS YOUR INPUT: nothing, but
add tests here if you extend physics.py with a new process term.
"""

import numpy as np
from genmonod.config import default_config, set_subsets, add_production
from genmonod.physics import simulate, system_rhs


def _fill_free(cfg, value=0.3, mortality=0.02):
    """Helper: fill every free (NaN) entry with a constant, for tests that
    don't care about fitting, just about the ODE being sane. Mortality
    defaults much lower than the other rates so growth isn't canceled out
    by construction (a rate exactly equal to mortality gives ~0 net
    growth, which isn't useful for a "does it grow" test)."""
    for attr in ["r", "K", "c", "mutation", "translocation",
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


def test_singleton_subset_matches_simple_single_substrate_monod():
    """The default (every substrate its own singleton subset) must exactly
    reproduce plain single-substrate Monod growth r*x/(K+x) -- this is the
    backward-compatibility contract for the shared-denominator rewrite."""
    cfg = default_config(1, 1, 0)
    cfg.r.values[:] = 0.5
    cfg.K.values[:] = 2.0
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.05
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0

    y0 = np.array([1.0, 10.0])
    t = np.array([0.0, 0.01])
    traj = simulate(cfg, y0, t)
    numeric_rate = (np.log(traj[1, 0]) - np.log(traj[0, 0])) / (t[1] - t[0])
    expected_rate = 0.5 * 10 / (2 + 10) - 0.05
    assert abs(numeric_rate - expected_rate) < 1e-3


def test_shared_subset_uses_one_denominator():
    """Two metabolites grouped into the SAME subset for a strain must use
    ONE shared saturation denominator (the paper's eq 9), not two
    independent ones -- this is what "metabolic overlap" means precisely
    in this model."""
    cfg = default_config(1, 2, 0)
    cfg.r.values[:] = [0.8, 0.6]
    cfg.K.values[:] = [2.0, 3.0]
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    set_subsets(cfg, cfg.strain_names[0], [[cfg.metabolite_names[0], cfg.metabolite_names[1]]])

    y_log = np.log([1.0, 5.0, 7.0])
    dydt = system_rhs(0, y_log, cfg)

    x1_K1, x2_K2 = 5.0 / 2.0, 7.0 / 3.0
    expected = (0.8 * x1_K1 + 0.6 * x2_K2) / (1 + x1_K1 + x2_K2)
    assert abs(dydt[0] - expected) < 1e-6


def test_toxin_independent_supply_is_additive():
    """A toxin in its own subset (the default) acts as a separate,
    ADDITIVE term with its own denominator -- the paper's eq 11 /
    "independent supply" case, which the paper focuses on."""
    cfg = default_config(1, 1, 1)
    cfg.r.values[:] = [0.8, -0.5]
    cfg.K.values[:] = [2.0, 1.5]
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    cfg.toxin_supply.values[:] = 0.0
    cfg.toxin_decay.values[:] = 0.0

    y_log = np.log([1.0, 5.0, 3.0])
    dydt = system_rhs(0, y_log, cfg)

    metabolic_term = 0.8 * (5.0 / 2.0) / (1 + 5.0 / 2.0)
    toxin_term = -0.5 * (3.0 / 1.5) / (1 + 3.0 / 1.5)
    assert abs(dydt[0] - (metabolic_term + toxin_term)) < 1e-6


def test_toxin_competing_supply_shares_denominator():
    """A toxin placed in the SAME subset as a metabolite competes with it
    for uptake (the paper's "option 1") -- ONE shared denominator instead
    of two independent ones."""
    cfg = default_config(1, 1, 1)
    cfg.r.values[:] = [0.8, -0.5]
    cfg.K.values[:] = [2.0, 1.5]
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    cfg.toxin_supply.values[:] = 0.0
    cfg.toxin_decay.values[:] = 0.0
    set_subsets(cfg, cfg.strain_names[0], [[cfg.metabolite_names[0], cfg.toxin_names[0]]])

    y_log = np.log([1.0, 5.0, 3.0])
    dydt = system_rhs(0, y_log, cfg)

    x1_K1, x2_K2 = 5.0 / 2.0, 3.0 / 1.5
    expected = (0.8 * x1_K1 + (-0.5) * x2_K2) / (1 + x1_K1 + x2_K2)
    assert abs(dydt[0] - expected) < 1e-6


def test_production_matches_paper_form():
    """A strain producing a toxin from a metabolite precursor (the paper's
    "ς" term) should match rate*x/(K+x)*N, using its OWN production-
    specific Monod constant, independent of the growth K."""
    cfg = default_config(1, 1, 1)
    cfg.r.values[:] = 0.0
    cfg.K.values[:] = 1.0
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    cfg.toxin_supply.values[:] = 0.0
    cfg.toxin_decay.values[:] = 0.0
    add_production(cfg, product=cfg.toxin_names[0], strain=cfg.strain_names[0],
                    precursor=cfg.metabolite_names[0], rate=0.4, half_sat=2.0)

    N, C, Tox = 1.0, 5.0, 3.0
    y_log = np.log([N, C, Tox])
    dydt = system_rhs(0, y_log, cfg)

    expected_dTox = 0.4 * (C / 2.0) / (1 + C / 2.0) * N
    assert abs(dydt[2] - expected_dTox / Tox) < 1e-6


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


def test_conjugative_transfer_bootstraps_third_population_from_zero():
    """translocation alone can't create a genuinely third population
    (donor + recipient -> transconjugant) since it requires the product
    type to already be present to catalyze its own growth. This is what
    add_conjugative_transfer is for — verify a near-zero transconjugant
    population can actually grow from donor x recipient contact."""
    from genmonod.config import add_conjugative_transfer

    cfg = default_config(n_strains=3, n_metabolites=1, n_toxins=0)
    cfg.strain_names = ["Recipient", "Donor", "Transconjugant"]
    cfg = _fill_free(cfg, 0.2)
    add_conjugative_transfer(cfg, product="Transconjugant", donor="Donor", recipient="Recipient", rate=0.05)

    y0 = np.array([1.0, 1.0, 1e-8, 10.0])
    t = np.linspace(0, 10, 20)
    traj = simulate(cfg, y0, t)
    assert traj[-1, 2] > 1e-6
    # donor and recipient should be essentially unaffected by the transfer itself
    assert traj[-1, 0] > 0.5 and traj[-1, 1] > 0.5


def test_dosing_event_adds_at_exact_time():
    from genmonod.config import add_dosing_event

    cfg = default_config(1, 1, 0)
    cfg.r.values[:] = 0.0
    cfg.K.values[:] = 1.0
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    add_dosing_event(cfg, time=5.0, target=cfg.metabolite_names[0], kind="add", amount=10.0)

    y0 = np.array([1.0, 2.0])
    t = np.array([0, 4, 5, 6, 10])
    traj = simulate(cfg, y0, t)
    assert abs(traj[1, 1] - 2.0) < 0.1   # t=4, before dosing: unchanged
    assert abs(traj[3, 1] - 12.0) < 0.2  # t=6, after dosing: jumped by +10


def test_passage_event_dilutes_every_state():
    """This is the key capability that was missing before: a passage event
    dilutes STRAIN populations too, not just a metabolite -- a continuous
    dilution ODE term can't represent a discrete, repeated culture
    dilution/passage step."""
    from genmonod.config import add_passage_event

    cfg = default_config(1, 1, 0)
    cfg.r.values[:] = 0.0
    cfg.K.values[:] = 1.0
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    add_passage_event(cfg, time=5.0, dilution_factor=0.1)

    y0 = np.array([100.0, 50.0])
    t = np.array([0, 4, 5, 6, 10])
    traj = simulate(cfg, y0, t)
    assert abs(traj[1, 0] - 100.0) < 0.1   # before dilution: unchanged
    assert abs(traj[3, 0] - 10.0) < 0.1    # strain population diluted 10x
    assert abs(traj[3, 1] - 5.0) < 0.1     # metabolite ALSO diluted 10x


def test_passage_and_dosing_at_same_timestamp():
    """Regression test: two schedule events at the identical timestamp must
    not create a zero-length integration segment (this failed before the
    fix -- same-instant events now get grouped and applied together,
    dilution before dosing, matching how fresh medium carries substrate
    into a diluted culture)."""
    from genmonod.config import add_passage_event, add_dosing_event

    cfg = default_config(1, 1, 0)
    cfg.r.values[:] = 0.0
    cfg.K.values[:] = 1.0
    cfg.c.values[:] = 0.0
    cfg.mortality.values[:] = 0.0
    cfg.metabolite_supply.values[:] = 0.0
    cfg.metabolite_dilution.values[:] = 0.0
    add_passage_event(cfg, time=5.0, dilution_factor=0.1)
    add_dosing_event(cfg, time=5.0, target=cfg.metabolite_names[0], kind="add", amount=20.0)

    y0 = np.array([100.0, 50.0])
    t = np.array([0, 4, 5, 6, 10])
    traj = simulate(cfg, y0, t)
    assert abs(traj[3, 0] - 10.0) < 0.1
    assert abs(traj[3, 1] - 25.0) < 0.2  # (50 * 0.1) + 20 = 25


def test_empty_schedule_is_a_true_noop():
    """A config with an empty schedule must behave identically to one
    without schedule support at all -- backward compatibility check."""
    cfg = default_config(1, 1, 0)
    cfg = _fill_free(cfg, 0.3)
    y0 = np.array([0.1, 10.0])
    t = np.linspace(0, 15, 20)
    assert cfg.schedule == []
    traj = simulate(cfg, y0, t)
    assert traj[-1, 0] > traj[0, 0]
