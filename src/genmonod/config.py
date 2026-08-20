"""
config.py

Defines the "generalized Monod system": how many strains/metabolites/
toxins you have, which optional processes (mutation, translocation,
toxins) are switched on, and — for every parameter matrix — which
entries are FIXED at a known value and which are FREE to be fit.

This is the file that encodes "all optional inputs": call
`default_config(...)` with just a strain count and everything else
defaults to off/free. Then use `set_fixed(...)` to pin down anything you
already know.

NEEDS YOUR INPUT: nothing to run the defaults. You'll mostly interact
with this through the visual app (app.py), not by editing this file —
but if you're scripting fits directly, this is the object you build.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MatrixSpec:
    """
    One parameter matrix (e.g. growth rate r, shape strains x metabolites).

    `values` holds the CURRENT value for every entry:
      - np.nan  -> this entry is FREE, fit will estimate it
      - a float -> this entry is FIXED at that value during fitting

    `lower`/`upper` give the fit bounds applied to FREE entries (ignored
    for fixed ones). Same shape as `values`, or a single (lo, hi) pair
    applied to every free entry.
    """
    values: np.ndarray
    lower: float = 0.0
    upper: float = 10.0

    @property
    def shape(self):
        return self.values.shape

    def is_free(self, i: int, j: int | None = None) -> bool:
        idx = (i,) if j is None else (i, j)
        return bool(np.isnan(self.values[idx]))

    def set_fixed(self, value: float, i: int, j: int | None = None) -> None:
        idx = (i,) if j is None else (i, j)
        self.values[idx] = value

    def set_free(self, i: int, j: int | None = None) -> None:
        idx = (i,) if j is None else (i, j)
        self.values[idx] = np.nan


def _free_matrix(shape: tuple[int, ...]) -> np.ndarray:
    """All-NaN matrix = every entry free by default."""
    return np.full(shape, np.nan, dtype=float)


@dataclass
class SystemConfig:
    """
    The full specification of one generalized Monod system.

    NEEDS YOUR INPUT: build these with `default_config()` below rather
    than constructing directly, unless you're comfortable wiring up every
    MatrixSpec by hand.
    """
    n_strains: int
    n_metabolites: int = 0
    n_toxins: int = 0
    include_mutation: bool = False
    include_translocation: bool = False

    strain_names: list[str] = field(default_factory=list)
    metabolite_names: list[str] = field(default_factory=list)
    toxin_names: list[str] = field(default_factory=list)

    # GROWTH / CONSUMPTION / UPTAKE, following the shared-denominator
    # multi-substrate Monod form derived from a proteome-allocation model
    # (Piskovsky, Schnepp-Pesch & Foster; extended to multi-strain,
    # multi-substrate, and toxins in "Modelling Microbe Population
    # Dynamics with Monod Equations Grounded in Statistical Mechanics").
    # This REPLACES an earlier version of this package that summed
    # independent single-substrate saturation curves — see `subsets`
    # below for why that was a special case of this, not a different model.
    #
    # r, K, c are all shape (S, M+T): ONE combined substrate axis
    # spanning metabolites AND toxins together, in that order
    # (metabolite_names then toxin_names). A toxin is mathematically
    # just a substrate with r<0 (it reduces growth instead of adding to
    # it) — same equations, no separate toxin-specific term needed.
    r: MatrixSpec | None = None   # growth rate constant per (strain, substrate); typically >0 for metabolites, <0 for toxins, but the SIGN is set/fit by you, never assumed
    K: MatrixSpec | None = None   # Monod constant per (strain, substrate) — SHARED between growth and consumption/uptake for that pair, per the paper
    c: MatrixSpec | None = None   # consumption/uptake rate per (strain, substrate) — depletes that substrate's pool; for a toxin this is "uptake" (e.g. binding/internalization), not literal consumption

    # per-strain terms
    mortality: MatrixSpec | None = None             # (S,)

    # strain-strain terms (both default OFF; only used if include_* is True)
    mutation: MatrixSpec | None = None               # (S, S) linear per-capita rate, entry [i,j] = j -> i rate
    translocation: MatrixSpec | None = None           # (S, S) contact-dependent rate, entry [i,j] = i converts j

    # environment terms — CONTINUOUS supply/dilution. Metabolites and
    # toxins keep separate supply/dilution mechanisms (metabolite_supply/
    # dilution vs toxin_supply/decay) since those are about how the
    # ENVIRONMENT behaves, unlike r/K/c above which are about how STRAINS
    # respond to a substrate and are genuinely unified between the two.
    metabolite_supply: MatrixSpec | None = None      # (M,)
    metabolite_dilution: MatrixSpec | None = None     # (M,)
    toxin_supply: MatrixSpec | None = None            # (T,)
    toxin_decay: MatrixSpec | None = None             # (T,)

    # SUBSETS: which substrates (metabolites and/or toxins) SHARE a
    # saturation denominator for a given strain — i.e. compete with each
    # other for that strain's limited uptake/protein capacity. This is
    # what "metabolic overlap" precisely means in this model: two
    # metabolites in the SAME subset compete (one shared denominator,
    # more of one leaves less "room" for the other); two metabolites in
    # DIFFERENT subsets are fully independent (separate denominators,
    # exactly recovering the old independent-summed-Monod behavior).
    # A toxin included in the same subset as metabolites represents
    # "toxin competes with metabolites for uptake"; a toxin in its own
    # subset (the DEFAULT — every substrate starts in its own singleton
    # subset) represents "toxin has an independent supply" — the case
    # the source paper focuses on. See `set_subsets` below.
    # dict: strain_name -> list of lists of substrate names.
    subsets: dict = field(default_factory=dict)

    # PRODUCTION: a strain synthesizing a substrate (metabolite or toxin)
    # from a PRECURSOR substrate — the paper's "ς" term. Different from
    # consumption/uptake (`c` above): production is an OUTPUT the strain
    # makes, at a rate that saturates with the PRECURSOR's own
    # concentration (via its own Monod constant, not the precursor's
    # growth-relevant K). List of dicts — see `add_production` below.
    production: list = field(default_factory=list)

    # conjugative transfer: a genuinely THIRD population (e.g. a
    # transconjugant) created via contact between a donor and a
    # recipient, WITHOUT depleting either of them — this is a different
    # process from `translocation` above, which only handles one
    # population converting into another (2 populations trading mass).
    # A donor/recipient/transconjugant system needs this instead: see
    # `add_conjugative_transfer` below. List of dicts, each:
    # {"product": name, "donor": name, "recipient": name, "rate": float
    # or nan, "lower": float, "upper": float}
    conjugative_transfer: list[dict] | None = None

    # HOW METABOLITE (or toxin) IS INTRODUCED: the continuous
    # metabolite_supply/metabolite_dilution terms above model a
    # steady, ongoing process (e.g. a true chemostat's continuous
    # feed). Real protocols are often DISCRETE instead — a single
    # one-time addition, or repeated dosing/passage events at specific
    # times — which a continuous ODE term can't represent on its own.
    # `schedule` holds DosingEvent / PassageEvent entries (see below)
    # for that. Continuous terms and a discrete schedule can be used
    # together (e.g. a small continuous baseline plus periodic
    # top-ups) or the schedule can be the ONLY input if you fix
    # metabolite_supply/dilution at 0.
    schedule: list = field(default_factory=list)

    def combined_substrate_names(self) -> list[str]:
        """Metabolites then toxins, in the order r/K/c's columns use."""
        return self.metabolite_names + self.toxin_names

    def shape_signature(self) -> tuple:
        """
        A hashable summary of this config's "shape" (dimensions + which
        optional terms are on) — used by fit_store.py / amortized_model.py
        to group past fits that can share an initial-guess network.
        """
        return (
            self.n_strains, self.n_metabolites, self.n_toxins,
            self.include_mutation, self.include_translocation,
            len(self.conjugative_transfer) if self.conjugative_transfer else 0,
            len(self.schedule), len(self.production),
        )


@dataclass
class DosingEvent:
    """
    A one-time change to a SPECIFIC metabolite or toxin's concentration
    at a specific time — e.g. "10 units of sugar were added at t=24h".
    Does not affect anything else (strains, other metabolites/toxins).

    Args:
        time: when this happens (same time units as your data).
        target: must match a name in cfg.metabolite_names or cfg.toxin_names.
        kind: "add" (increase by `amount`) or "set" (jump to exactly `amount`).
        amount: fixed and known — see add_dosing_event below for why this
            isn't a fittable free parameter in the common case.
    """
    time: float
    target: str
    kind: str  # "add" or "set"
    amount: float


@dataclass
class PassageEvent:
    """
    A dilution/passage event: EVERY state (every strain, metabolite, and
    toxin) is multiplied by `dilution_factor` at this time — e.g.
    "the culture was diluted 10-fold into fresh medium at t=24h" is
    `dilution_factor=0.1`. This is what a repeated-passage protocol
    needs that a continuous ODE dilution term can't represent: it acts
    on the STRAIN populations too, not just the metabolite.

    Combine with a DosingEvent at the same `time` to represent "diluted
    AND fresh substrate was added" (a chemostat-style passage step) —
    register both; order between same-timestamp events doesn't matter
    since they're independent multiplicative/additive operations applied
    at that instant.
    """
    time: float
    dilution_factor: float


def add_dosing_event(cfg: "SystemConfig", time: float, target: str, kind: str, amount: float) -> None:
    """
    Registers a one-time addition/reset of a specific metabolite or
    toxin's concentration. `target` must already exist in
    cfg.metabolite_names or cfg.toxin_names.

    NEEDS YOUR INPUT: `amount` is fixed at whatever you specify — it is
    NOT currently fit as a free parameter, since in the common case you
    designed the protocol yourself and know exactly how much you added.
    If you genuinely don't know the amount and want it estimated from
    data, that's a natural extension (similar to how
    add_conjugative_transfer's rate can be left free) that isn't built
    yet — open an issue / extend fitting.py's pack/unpack if you need it.
    """
    if kind not in ("add", "set"):
        raise ValueError(f"kind must be 'add' or 'set', got {kind!r}")
    if target not in cfg.metabolite_names and target not in cfg.toxin_names:
        raise ValueError(f"'{target}' is not in cfg.metabolite_names or cfg.toxin_names")
    cfg.schedule.append(DosingEvent(time=time, target=target, kind=kind, amount=amount))


def add_passage_event(cfg: "SystemConfig", time: float, dilution_factor: float) -> None:
    """
    Registers a passage/dilution event affecting EVERY state (strains,
    metabolites, toxins) at once — see PassageEvent above.

    NEEDS YOUR INPUT: `dilution_factor` is fixed (e.g. 0.1 for a
    10-fold dilution) — not fit as a free parameter, for the same
    reason as add_dosing_event's `amount`.
    """
    cfg.schedule.append(PassageEvent(time=time, dilution_factor=dilution_factor))


def default_config(
    n_strains: int,
    n_metabolites: int = 1,
    n_toxins: int = 0,
    include_mutation: bool = False,
    include_translocation: bool = False,
) -> SystemConfig:
    """
    Build a SystemConfig where every applicable matrix entry is FREE
    (fit will estimate it) with sensible default bounds. This is the
    normal entry point — every "constraint" is optional, so calling this
    with just a strain count gives you a fully-general, fully-free
    system to fit.

    By default every substrate (metabolite or toxin) is placed in its
    OWN singleton subset per strain — i.e. every substrate saturates
    independently of every other, and toxins have an independent supply
    from metabolites. This is the same behavior as a simple summed-
    Monod model; call `set_subsets` to introduce metabolic overlap
    (shared uptake capacity) or toxin/metabolite competition.

    NEEDS YOUR INPUT: nothing to call this. Use `set_fixed` afterward (or
    the visual app) to pin down anything you already know.
    """
    S, M, T = n_strains, n_metabolites, n_toxins

    cfg = SystemConfig(
        n_strains=S, n_metabolites=M, n_toxins=T,
        include_mutation=include_mutation,
        include_translocation=include_translocation,
        strain_names=[f"Strain_{i+1}" for i in range(S)],
        metabolite_names=[f"Metabolite_{k+1}" for k in range(M)],
        toxin_names=[f"Toxin_{l+1}" for l in range(T)],
    )

    n_sub = M + T
    if n_sub > 0:
        # r allows NEGATIVE values: a metabolite typically has r>0 (adds to
        # growth), a toxin typically has r<0 (subtracts from growth) — same
        # matrix, same equation, sign carries the meaning, per the paper.
        cfg.r = MatrixSpec(_free_matrix((S, n_sub)), -2.0, 2.0)
        cfg.K = MatrixSpec(_free_matrix((S, n_sub)), 1e-3, 10.0)
        cfg.c = MatrixSpec(_free_matrix((S, n_sub)), 0.0, 5.0)

    if M > 0:
        cfg.metabolite_supply = MatrixSpec(_free_matrix((M,)), 0.0, 5.0)
        cfg.metabolite_dilution = MatrixSpec(_free_matrix((M,)), 0.0, 2.0)
    if T > 0:
        cfg.toxin_supply = MatrixSpec(_free_matrix((T,)), 0.0, 2.0)
        cfg.toxin_decay = MatrixSpec(_free_matrix((T,)), 0.0, 2.0)

    cfg.mortality = MatrixSpec(_free_matrix((S,)), 0.0, 1.0)

    if include_mutation:
        m = _free_matrix((S, S))
        np.fill_diagonal(m, 0.0)  # diagonal is never free: no self-mutation
        cfg.mutation = MatrixSpec(m, 0.0, 1.0)

    if include_translocation:
        tl = _free_matrix((S, S))
        np.fill_diagonal(tl, 0.0)  # diagonal is never free: no self-translocation
        cfg.translocation = MatrixSpec(tl, 0.0, 1.0)

    # default: every substrate its own singleton subset, for every strain
    # (fully independent saturation, matching a plain summed-Monod model)
    all_names = cfg.combined_substrate_names()
    cfg.subsets = {s: [[n] for n in all_names] for s in cfg.strain_names}

    return cfg


def set_subsets(cfg: SystemConfig, strain: str, groups: list[list[str]]) -> None:
    """
    Declares which substrates SHARE a saturation denominator (compete for
    uptake capacity) for one strain — this is what "metabolic overlap"
    (or toxin/metabolite competition) means precisely in this model.

    Args:
        strain: must be in cfg.strain_names.
        groups: each inner list is a set of metabolite/toxin NAMES that
            compete with each other for this strain. Any substrate NOT
            mentioned in any group keeps its own independent singleton
            subset (unaffected). Putting a toxin in the same group as
            metabolites represents "toxin competes with metabolites for
            uptake" (the paper's option 1); leaving it in its own
            singleton (the default) represents "independent supply"
            (the paper's option 2, its main focus).

    Example — two metabolites competing for the same uptake machinery:
        set_subsets(cfg, "StrainA", [["Glucose", "Fructose"]])
    """
    if strain not in cfg.strain_names:
        raise ValueError(f"'{strain}' is not in cfg.strain_names")
    all_names = cfg.combined_substrate_names()
    grouped = set()
    for g in groups:
        for n in g:
            if n not in all_names:
                raise ValueError(f"'{n}' is not a metabolite or toxin name in this config")
            if n in grouped:
                raise ValueError(f"'{n}' appears in more than one group — a substrate can only be in one subset per strain")
            grouped.add(n)
    remaining = [[n] for n in all_names if n not in grouped]
    cfg.subsets[strain] = [list(g) for g in groups] + remaining


def add_production(
    cfg: SystemConfig, product: str, strain: str, precursor: str,
    rate: float | None = None, half_sat: float | None = None,
    rate_bounds: tuple[float, float] = (0.0, 2.0),
    half_sat_bounds: tuple[float, float] = (1e-3, 10.0),
) -> None:
    """
    Registers that `strain` PRODUCES `product` (a metabolite or toxin)
    using `precursor` (another metabolite or toxin) as input — the
    paper's "ς" term. This is different from growth/consumption (`r`/`c`
    above): production is an output the strain synthesizes, saturating
    with the PRECURSOR's own concentration via its OWN Monod constant
    (not the precursor's growth-relevant K — production and growth can
    saturate at different precursor levels).

    If multiple precursors are registered for the same (product, strain)
    and some of them share a subset (see set_subsets) for that strain,
    they'll share ONE saturation denominator when producing this
    product too — mirroring how growth/consumption share denominators
    within a subset.

    Args:
        rate: the paper's d_{product,precursor,strain}. None = free.
        half_sat: the paper's K_{product,precursor,strain} (production-
            specific — distinct from precursor's growth K). None = free.

    NEEDS YOUR INPUT: nothing for the common case. Call once per real
    production pathway (e.g. once per precursor if a product draws from
    several).
    """
    all_names = cfg.combined_substrate_names()
    if product not in all_names:
        raise ValueError(f"'{product}' is not a metabolite or toxin name in this config")
    if precursor not in all_names:
        raise ValueError(f"'{precursor}' is not a metabolite or toxin name in this config")
    if strain not in cfg.strain_names:
        raise ValueError(f"'{strain}' is not in cfg.strain_names")
    cfg.production.append({
        "product": product, "strain": strain, "precursor": precursor,
        "rate": np.nan if rate is None else float(rate),
        "half_sat": np.nan if half_sat is None else float(half_sat),
        "rate_lower": rate_bounds[0], "rate_upper": rate_bounds[1],
        "half_sat_lower": half_sat_bounds[0], "half_sat_upper": half_sat_bounds[1],
    })


def add_conjugative_transfer(
    cfg: SystemConfig, product: str, donor: str, recipient: str,
    rate: float | None = None, lower: float = 0.0, upper: float = 2.0,
) -> None:
    """
    Registers a conjugative-transfer process: contact between `donor`
    and `recipient` creates new individuals of `product` (a genuinely
    different strain — e.g. a transconjugant), at rate `rate * N_donor *
    N_recipient`. Neither donor nor recipient is depleted by this — the
    standard simplifying assumption for these systems is that transfer
    events are rare relative to the donor/recipient population sizes,
    so their own growth/death dynamics dominate.

    This is DIFFERENT from `translocation` in the base config: that term
    only handles one population converting into another (two
    populations, one trading mass for the other). Donor + recipient ->
    transconjugant is a three-population process and needs this instead.

    Args:
        product, donor, recipient: must match names already in
            cfg.strain_names.
        rate: pass a number to fix it, or leave None to fit it freely
            within [lower, upper].

    NEEDS YOUR INPUT: nothing to call this for the common case. Call it
    once per real conjugation process in your system (usually just one
    donor/recipient/transconjugant triple, but nothing stops you from
    registering more).
    """
    for name in (product, donor, recipient):
        if name not in cfg.strain_names:
            raise ValueError(f"'{name}' is not in cfg.strain_names: {cfg.strain_names}")
    entry = {
        "product": product, "donor": donor, "recipient": recipient,
        "rate": np.nan if rate is None else float(rate),
        "lower": lower, "upper": upper,
    }
    if cfg.conjugative_transfer is None:
        cfg.conjugative_transfer = []
    cfg.conjugative_transfer.append(entry)
