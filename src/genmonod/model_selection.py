"""
model_selection.py

Automatically builds and fits several structural variants of a system
(toxin present or not, metabolic overlap or not, mutation/translocation
on or off) against your data, and ranks them so you don't have to guess
or manually run each combination yourself — this is the package version
of the manual toxin-vs-no-toxin comparison from earlier analysis, now
reusable for any strain set and any data.

Ranking uses AICc (small-sample-corrected AIC), not plain AIC. This
matters a lot for real biological datasets: AIC's complexity penalty
(2k) alone gets unreliable once free parameters (k) approach the number
of observations (n) — exactly the regime small experiments are usually
in — and AICc corrects for that. When k is too close to n for AICc to
even be computable, that's surfaced explicitly rather than hidden.

NEEDS YOUR INPUT: nothing to use the defaults. The main things you might
want to change are which axes to search (`include_toxin`,
`metabolic_overlap`, `include_mutation`, `include_translocation`) and
`consumption_bounds` if your data's population scale needs different
bounds than the default (see the note on `fix_consumption_zero` below).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .config import default_config, add_conjugative_transfer, SystemConfig
from .fitting import fit, FitResult
from .data_io import Dataset


@dataclass
class StructureCandidate:
    name: str
    description: str
    cfg: SystemConfig


@dataclass
class StructureResult:
    name: str
    description: str
    result: FitResult
    n_obs: int
    n_free: int
    aic: float
    aicc: float
    reliable: bool  # False when n_obs is too close to n_free for AICc to mean much


def _aic_aicc(rss: float, n: int, k: int) -> tuple[float, float, bool]:
    if rss <= 0 or n <= 0:
        return np.inf, np.inf, False
    aic = n * np.log(rss / n) + 2 * k
    denom = n - k - 1
    if denom <= 0:
        # AICc's correction term divides by (n-k-1); at or below zero it's
        # undefined. Rather than silently return a number, flag it.
        return aic, np.inf, False
    aicc = aic + (2 * k * (k + 1)) / denom
    return aic, aicc, (n >= 2 * k)  # rule-of-thumb reliability flag, not a hard cutoff


def build_structure_grid(
    n_strains: int,
    strain_names: list[str] | None = None,
    include_toxin: tuple = (False, True),
    metabolic_overlap: tuple = ("shared", "separate"),
    include_mutation: tuple = (False,),
    include_translocation: tuple = (False,),
    fix_consumption_zero: bool = True,
    consumption_bounds: tuple[float, float] | None = None,
) -> list[StructureCandidate]:
    """
    Builds every combination of the requested structural axes as separate
    SystemConfigs. Each combination gets a config where the corresponding
    matrices are either present (free) or absent (n=0 / fixed at 0).

    `metabolic_overlap` here means whether multiple STRAINS draw from the
    SAME resource pool ("shared", one metabolite state) or each has its
    own private one ("separate", one metabolite state per strain) — this
    is a different question from config.set_subsets (which is about
    whether a SINGLE strain's own multiple substrates compete with each
    other for its uptake capacity). If you want to search subset-based
    within-strain competition too, build those candidates manually with
    set_subsets — not auto-searched here, to keep the default grid a
    manageable size.

    Args:
        fix_consumption_zero: the earlier joint fits found that freely-
            fit consumption/uptake at real cfu/mL population scale
            (~1e8-1e9) makes the ODE catastrophically stiff (a substrate
            depletes in microseconds against an hours-long experiment).
            Default True keeps BOTH metabolite consumption AND toxin
            uptake at 0 (substrate dynamics driven only by their own
            supply/dilution, decoupled from population size) for
            numerical safety. Set False and pass `consumption_bounds`
            (tightly scaled to your population size, e.g. (0, 1e-7) for
            ~1e8-scale populations) if resource depletion/competition is
            actually the thing you're trying to test — see the NSERC
            competition fit for a worked example of when that matters.
    """
    candidates = []
    for tox in include_toxin:
        for overlap in metabolic_overlap:
            for mut in include_mutation:
                for transloc in include_translocation:
                    n_metab = 1 if overlap == "shared" else n_strains
                    n_tox = 1 if tox else 0
                    cfg = default_config(
                        n_strains=n_strains, n_metabolites=n_metab, n_toxins=n_tox,
                        include_mutation=mut, include_translocation=transloc,
                    )
                    if strain_names:
                        cfg.strain_names = list(strain_names)

                    if overlap == "separate":
                        for i in range(n_strains):
                            for k in range(n_metab):
                                if i != k:
                                    cfg.r.set_fixed(0.0, i, k)
                                    cfg.K.set_fixed(1.0, i, k)  # irrelevant, rate=0

                    n_sub = n_metab + n_tox
                    if fix_consumption_zero:
                        for i in range(n_strains):
                            for k in range(n_sub):
                                cfg.c.set_fixed(0.0, i, k)
                    elif consumption_bounds:
                        cfg.c.lower, cfg.c.upper = consumption_bounds

                    tags = [f"{overlap}-metabolite"]
                    if tox:
                        tags.append("toxin")
                    if mut:
                        tags.append("mutation")
                    if transloc:
                        tags.append("translocation")
                    name = "+".join(tags)
                    desc = (f"{'one shared' if overlap == 'shared' else 'separate per-strain'} metabolite pool, "
                            f"toxin {'ON' if tox else 'OFF'}, mutation {'ON' if mut else 'OFF'}, "
                            f"translocation {'ON' if transloc else 'OFF'}")
                    candidates.append(StructureCandidate(name=name, description=desc, cfg=cfg))
    return candidates


def compare_structures(
    dataset_builder,
    n_strains: int,
    strain_names: list[str] | None = None,
    include_toxin: tuple = (False, True),
    metabolic_overlap: tuple = ("shared", "separate"),
    include_mutation: tuple = (False,),
    include_translocation: tuple = (False,),
    fix_consumption_zero: bool = True,
    consumption_bounds: tuple[float, float] | None = None,
    conjugative_transfer: list[tuple[str, str, str]] | None = None,
    max_nfev: int = 150,
) -> list[StructureResult]:
    """
    Fits every requested structural combination and ranks them by AICc,
    best (lowest) first.

    Args:
        dataset_builder: a function `cfg -> Dataset | list[Dataset]`, NOT
            a fixed dataset. This matters: "shared metabolite" vs
            "separate metabolite" candidates have a DIFFERENT number of
            metabolite columns (1 vs n_strains), so the same Dataset
            object can't be reused across every candidate — each
            candidate needs its own matching one, built from its own
            resolved cfg. If you already have raw data loaded and just
            need to wrap it, something like
            `lambda cfg: Dataset.from_csv(path, time_col, column_map, cfg)`
            is usually enough, since column_map only needs strain names.
        conjugative_transfer: optional list of (product, donor, recipient)
            name triples, registered on EVERY candidate config. Use this
            for a donor/recipient/transconjugant-type system where the
            transfer process itself isn't really "optional" the way
            toxin/overlap are — it's part of the base biology, so it
            shouldn't be one of the things being toggled on/off.

    Returns:
        Results sorted best-first by AICc. A result with `.reliable=False`
        had too few observations relative to its free parameter count for
        AICc to be a solid basis for comparison — still returned (never
        silently dropped), but treat its ranking with real caution rather
        than as a confident conclusion.
    """
    candidates = build_structure_grid(
        n_strains, strain_names, include_toxin, metabolic_overlap,
        include_mutation, include_translocation, fix_consumption_zero, consumption_bounds,
    )

    results = []
    for cand in candidates:
        cfg = cand.cfg
        if conjugative_transfer:
            for product, donor, recipient in conjugative_transfer:
                add_conjugative_transfer(cfg, product, donor, recipient, lower=0.0, upper=5e-9)
        try:
            datasets = dataset_builder(cfg)
            if isinstance(datasets, Dataset):
                datasets = [datasets]
            n_obs = sum(int(np.sum(~np.isnan(d.Y))) for d in datasets)

            r = fit(cfg, datasets, max_nfev=max_nfev)
            k = len(r.x)
            rss = 2 * r.cost
            aic, aicc, reliable = _aic_aicc(rss, n_obs, k)
            results.append(StructureResult(cand.name, cand.description, r, n_obs, k, aic, aicc, reliable))
        except Exception as e:
            print(f"structure '{cand.name}' failed to fit: {e}")

    results.sort(key=lambda x: x.aicc)
    return results


def summarize(results: list[StructureResult]) -> str:
    """A quick, readable text summary — best first, with the reliability caveat surfaced."""
    lines = [f"{'structure':40s} {'AICc':>10s} {'AIC':>10s} {'n_free':>7s} {'n_obs':>6s}  reliable?"]
    for r in results:
        lines.append(f"{r.name:40s} {r.aicc:10.2f} {r.aic:10.2f} {r.n_free:7d} {r.n_obs:6d}  {'yes' if r.reliable else 'NO -- too few obs for n_free'}")
    return "\n".join(lines)
