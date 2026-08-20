"""
physics.py

The generalized Monod ODE system. Growth and consumption/uptake follow
the SHARED-DENOMINATOR multi-substrate Monod form derived from a
proteome-allocation model (Piskovsky, Schnepp-Pesch & Foster; extended
to multi-strain, multi-substrate systems with toxins in "Modelling
Microbe Population Dynamics with Monod Equations Grounded in
Statistical Mechanics", Aug 2026) — NOT independently-summed
single-substrate saturation curves. Substrates that share a "subset"
(config.set_subsets) compete for one strain's limited uptake capacity
via one shared denominator; substrates in different subsets are fully
independent. A toxin is mathematically just a substrate with a
negative rate constant — same equations as a metabolite, no separate
mechanism needed, whether it shares a subset with metabolites
(competing for uptake) or has its own (independent supply, the source
paper's focus).

Every OTHER process (mortality, mutation, translocation, conjugative
transfer) is a SEPARATE additive term, each driven by its own matrix
from config.py — this is what makes it "generalized": turning a
process off just means its matrix/list is absent, not a different set
of equations.

State is tracked in log-space (log N, log C, log Tox) purely so that
populations/concentrations can never go negative during numerical
integration, and so scipy's stiff solvers behave well over many orders
of magnitude.

# WHAT WAS FIXED vs the original MATLAB version (for reference):
#   - the strain-strain transfer term ("Z_transconjugation") only ever
#     used entry (3,1) and hardcoded "strain 1/2/3 = recipient/donor/
#     transconjugant". Here, translocation uses the FULL S x S matrix
#     for arbitrary strain count and topology.
#   - "mutation" did not exist as a term at all. It's added here as its
#     own linear (non-contact-dependent) process, separate from
#     translocation (which stays contact/mass-action dependent).
#   - dead parameters (`g`, `d_outflow`) that were parsed but never used
#     in the ODE are removed entirely.
#
# WHAT WAS FIXED, continued: `translocation` alone can't produce a THIRD
# population from donor-recipient contact (see config.add_conjugative_transfer
# for the fix) — found while fitting this package's own generalized model
# against real donor/recipient/transconjugant data and noticing transconjugant
# populations could never bootstrap from near-zero.
#
# WHAT CHANGED (Aug 2026): growth/consumption/toxin-kill were previously
# each substrate's saturation curve summed independently (equivalent to
# every substrate always being in its own subset). That's still exactly
# what you get by default here (every substrate defaults to a singleton
# subset) — the shared-denominator form is a strict generalization, not
# a breaking change in behavior, unless you explicitly group substrates
# with `set_subsets`. What IS a breaking API change: `growth_rate`,
# `growth_half_sat`, `toxin_kill_rate`, `toxin_half_sat`, `consumption`,
# and `secretion` are replaced by three unified matrices — `r`, `K`, `c`
# — spanning metabolites and toxins together (see config.py). `secretion`
# specifically is replaced by the more general `production` mechanism,
# which properly saturates on the PRECURSOR's own concentration via a
# production-specific Monod constant, rather than being tied to kill
# activity.
#
# IDENTIFIABILITY NOTE (mutation and translocation both): with only
# population-count data (not strain-tagged measurements), a pairwise
# strain-strain process is only identifiable up to its NET direction —
# mutation[i,j] and mutation[j,i] (or translocation[i,j] and [j,i]) trade
# off against each other in the fit unless you already know one
# direction is zero and fix it as such. See examples/quickstart_example.py
# for a worked example of this. This is also why the original MATLAB
# tool tracked donor/recipient/transconjugant as three SEPARATE
# populations rather than a pairwise conversion — that's what makes a
# transfer direction and rate uniquely recoverable from data.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from .config import SystemConfig

_EPS = 1e-8


def system_rhs(t: float, y: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    Right-hand side of the full system, in log-state.

    Args:
        y: flat array [log N (S,), log C (M,), log Tox (T,)]
        cfg: SystemConfig with all matrix VALUES already resolved to
             concrete floats (see fitting.py — during fitting, free NaN
             entries get replaced with a candidate value before this is
             called; this function itself never sees NaN).

    NEEDS YOUR INPUT: only if you're adding a brand new physical process
    beyond growth/toxin/mortality/mutation/translocation. Each existing
    process is a clearly separated block below — add a new block the
    same way rather than editing the existing ones, to keep processes
    independent as you asked.
    """
    S, M, T = cfg.n_strains, cfg.n_metabolites, cfg.n_toxins
    logN = y[:S]
    logC = y[S:S + M]
    logTox = y[S + M:S + M + T]

    N = np.exp(logN)
    C = np.exp(logC) if M > 0 else np.zeros(0)
    Tox = np.exp(logTox) if T > 0 else np.zeros(0)
    substrate_x = np.concatenate([C, Tox])          # (M+T,) combined metabolite+toxin concentrations
    substrate_names = cfg.combined_substrate_names()
    n_sub = M + T

    growth_percapita = np.zeros(S)             # (S,) per-capita growth, all subsets summed
    uptake = np.zeros((S, n_sub))               # (S, M+T) per-capita consumption/uptake rate, by (strain, substrate)

    # --- growth + consumption/uptake: shared-denominator multi-substrate
    # Monod form, per strain, per SUBSET of substrates that compete for
    # that strain's uptake capacity (see config.set_subsets). A substrate
    # in its own singleton subset (the default) reduces to the familiar
    # single-substrate Monod term -- this is a strict generalization, not
    # a different model, of independently-summed saturation curves.
    if n_sub > 0:
        name_to_idx = {n: i for i, n in enumerate(substrate_names)}
        for k, strain in enumerate(cfg.strain_names):
            groups = cfg.subsets.get(strain) or [[n] for n in substrate_names]
            for group in groups:
                idxs = [name_to_idx[n] for n in group if n in name_to_idx]
                if not idxs:
                    continue
                x_g = substrate_x[idxs]
                K_g = cfg.K.values[k, idxs]
                sat_g = x_g / (K_g + _EPS)                     # x/K per substrate in this subset
                denom = 1.0 + sat_g.sum()                       # ONE shared denominator for the whole subset
                r_g = cfg.r.values[k, idxs]
                growth_percapita[k] += (r_g * sat_g).sum() / denom
                c_g = cfg.c.values[k, idxs]
                uptake[k, idxs] = c_g * sat_g / denom

    mortality = cfg.mortality.values                                # (S,)

    # --- population dynamics: sum of independent processes ---
    dN = N * (growth_percapita - mortality)

    if cfg.include_mutation and cfg.mutation is not None:
        mut = cfg.mutation.values                                    # (S, S), entry [i,j] = per-capita rate j -> i
        inflow = mut @ N                                             # (S,)
        outflow = N * mut.sum(axis=0)                                # (S,)
        dN = dN + inflow - outflow

    if cfg.include_translocation and cfg.translocation is not None:
        tl = cfg.translocation.values                                 # (S, S), entry [i,j] = contact rate: i converts j
        # dN_i += N_i * sum_j tl[i,j]*N_j   (i gains, acting as catalyst)
        # dN_i -= N_i * sum_k tl[k,i]*N_k   (i is converted away by contact with k)
        gain = N * (tl @ N)
        loss = N * (tl.T @ N)
        dN = dN + gain - loss

    if cfg.conjugative_transfer:
        # a genuinely THIRD population created from donor-recipient
        # contact, e.g. a transconjugant — see config.add_conjugative_transfer
        # for why this needs to be separate from `translocation` above.
        strain_idx = {name: i for i, name in enumerate(cfg.strain_names)}
        for entry in cfg.conjugative_transfer:
            p, d, r = strain_idx[entry["product"]], strain_idx[entry["donor"]], strain_idx[entry["recipient"]]
            source = entry["rate"] * N[d] * N[r]
            dN[p] = dN[p] + source
            # donor and recipient are NOT depleted — see docstring in config.py

    dydt = np.zeros_like(y)
    dydt[:S] = dN / (N + _EPS)  # d(logN)/dt = dN/dt / N

    # --- substrate (metabolite + toxin) dynamics ---
    if n_sub > 0:
        consumed_total = uptake.T @ N                                # (M+T,) total uptake, summed over strains weighted by abundance

        # --- production: a strain synthesizing `product` from `precursor`
        # (the paper's "ς" term) — its OWN saturation, using the
        # production-specific Monod constant, NOT growth's K. Precursors
        # registered for the same (product, strain) that share a subset
        # (per that strain's growth/consumption grouping) share ONE
        # denominator here too, mirroring growth's structure.
        produced_total = np.zeros(n_sub)
        if cfg.production:
            groups_by_product_strain = {}
            for entry in cfg.production:
                groups_by_product_strain.setdefault((entry["product"], entry["strain"]), []).append(entry)
            for (product, strain), entries in groups_by_product_strain.items():
                k = cfg.strain_names.index(strain)
                p_idx = name_to_idx[product]
                strain_groups = cfg.subsets.get(strain) or [[n] for n in substrate_names]
                precursor_entry = {e["precursor"]: e for e in entries}
                for group in strain_groups:
                    relevant = [n for n in group if n in precursor_entry]
                    if not relevant:
                        continue
                    x_l = np.array([substrate_x[name_to_idx[n]] for n in relevant])
                    K_l = np.array([precursor_entry[n]["half_sat"] for n in relevant])
                    d_l = np.array([precursor_entry[n]["rate"] for n in relevant])
                    sat_l = x_l / (K_l + _EPS)
                    denom_l = 1.0 + sat_l.sum()
                    produced_total[p_idx] += ((d_l * sat_l).sum() / denom_l) * N[k]

        dSubstrate = np.zeros(n_sub)
        if M > 0:
            dSubstrate[:M] += cfg.metabolite_supply.values - cfg.metabolite_dilution.values * C
        if T > 0:
            dSubstrate[M:] += cfg.toxin_supply.values - cfg.toxin_decay.values * Tox
        dSubstrate += produced_total - consumed_total

        dydt[S:S + n_sub] = dSubstrate / (substrate_x + _EPS)

    return dydt


def _integrate_segment(cfg, y0_log, t_start, t_end, t_eval_in_segment, method, rtol, atol):
    """One continuous-ODE integration between two schedule events (or the
    start/end of the whole simulation if there's no schedule). Always
    integrates through to t_end so the caller has the exact state to apply
    the next event to, even if t_end wasn't one of the user's requested
    output times."""
    # t_eval must be sorted and within [t_start, t_end]; make sure t_end
    # itself is included so we always know the state at the segment boundary
    t_eval_full = t_eval_in_segment
    need_end = len(t_eval_full) == 0 or t_eval_full[-1] < t_end
    if need_end:
        t_eval_full = np.concatenate([t_eval_full, [t_end]])
    with np.errstate(over="ignore", invalid="ignore"):
        sol = solve_ivp(
            system_rhs, (t_start, t_end), y0_log, t_eval=t_eval_full,
            args=(cfg,), method=method, rtol=rtol, atol=atol,
        )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    y_log_at_end = sol.y[:, -1]
    # drop the synthetic t_end row from the returned points if the caller
    # didn't actually ask for it
    y_out = sol.y[:, :-1] if need_end else sol.y
    return y_out, y_log_at_end


def _apply_event(cfg: SystemConfig, y_linear: np.ndarray, event) -> np.ndarray:
    """Applies one DosingEvent or PassageEvent to a LINEAR-space state
    vector, returning the updated state (does not mutate in place)."""
    from .config import DosingEvent, PassageEvent  # local import avoids a cycle at module load time

    y = y_linear.copy()
    S, M, T = cfg.n_strains, cfg.n_metabolites, cfg.n_toxins
    if isinstance(event, PassageEvent):
        y = y * event.dilution_factor
    elif isinstance(event, DosingEvent):
        if event.target in cfg.metabolite_names:
            idx = S + cfg.metabolite_names.index(event.target)
        else:
            idx = S + M + cfg.toxin_names.index(event.target)
        y[idx] = y[idx] + event.amount if event.kind == "add" else event.amount
    else:
        raise TypeError(f"unknown schedule event type: {type(event)}")
    return np.maximum(y, _EPS)  # keep strictly positive for the next segment's log-state


def simulate(
    cfg: SystemConfig, y0: np.ndarray, t_eval: np.ndarray,
    method: str = "LSODA", rtol: float = 1e-4, atol: float = 1e-6,
) -> np.ndarray:
    """
    Simulate the system forward from y0 (linear-space initial conditions,
    length S+M+T) at the requested time points.

    Returns:
        traj: (len(t_eval), S+M+T) array in LINEAR space (already
              exponentiated back out of log-space for you).

    NEEDS YOUR INPUT: the default was RK45 early on (faster per call on
    small well-behaved synthetic examples), but real fits on real
    cfu/mL-scale data repeatedly hit a failure mode where a small
    fraction of parameter combinations the optimizer explores mid-search
    are pathologically stiff for RK45 — it doesn't fail cleanly, it just
    grinds through an enormous number of tiny steps, and a SINGLE
    evaluation like that can blow a fit's entire time budget. LSODA
    auto-detects stiffness and switches to an implicit method, which
    cost several real debugging cycles to track down and is now the
    default because robustness matters more than a modest constant-
    factor speedup once you're fitting real data, not toy examples. If
    you're fitting small, well-scaled synthetic systems and want the
    speed back, pass `method="RK45"` explicitly.

    If cfg.schedule has entries (see config.DosingEvent / PassageEvent),
    integration is split into segments at each event time, with the
    event's discrete change applied at the boundary between segments —
    otherwise this is a single continuous integration, unchanged from
    before schedule support existed (so a config with an empty schedule
    behaves identically to earlier versions of this function).
    """
    y0_log = np.log(np.maximum(y0, _EPS))
    from .config import PassageEvent  # local import avoids a cycle at module load time

    events = sorted(cfg.schedule, key=lambda e: e.time) if cfg.schedule else []
    # only events strictly inside the requested time window affect the
    # output; one at or before t_eval[0] is applied to the initial state
    # instead of splitting off a zero-length segment
    t_start, t_end = t_eval[0], t_eval[-1]
    leading = [e for e in events if e.time <= t_start]
    mid = [e for e in events if t_start < e.time < t_end]

    if not events:
        # unchanged fast path -- identical to the pre-schedule implementation
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(
                system_rhs, (t_start, t_end), y0_log, t_eval=t_eval,
                args=(cfg,), method=method, rtol=rtol, atol=atol,
            )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return np.exp(sol.y.T)

    y_linear = np.exp(y0_log)
    for e in leading:
        y_linear = _apply_event(cfg, y_linear, e)
    y_log_current = np.log(y_linear)

    # Group same-instant events into one boundary each (two events at the
    # identical timestamp must NOT become two segments -- that creates a
    # zero-length integration span). Within a shared timestamp, apply
    # PassageEvents before DosingEvents: physically, a passage dilutes the
    # existing culture and the incoming fresh medium is what carries any
    # substrate being added, so "diluted, then fresh substrate arrives" is
    # the correct order, not the reverse.
    unique_times = sorted(set(e.time for e in mid))
    events_by_time = {t: sorted((e for e in mid if e.time == t), key=lambda e: 0 if isinstance(e, PassageEvent) else 1)
                       for t in unique_times}

    boundaries = [t_start] + unique_times + [t_end]
    out_chunks = []
    t_mask_used = np.zeros_like(t_eval, dtype=bool)
    for seg_i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[seg_i], boundaries[seg_i + 1]
        in_seg = (~t_mask_used) & (t_eval >= seg_start) & (t_eval <= seg_end)
        t_mask_used |= in_seg
        y_out, y_log_current = _integrate_segment(
            cfg, y_log_current, seg_start, seg_end, t_eval[in_seg], method, rtol, atol,
        )
        out_chunks.append((np.where(in_seg)[0], y_out))
        if seg_i < len(unique_times):  # apply every event at this boundary before the next segment
            y_lin = np.exp(y_log_current)
            for e in events_by_time[unique_times[seg_i]]:
                y_lin = _apply_event(cfg, y_lin, e)
            y_log_current = np.log(y_lin)

    traj_log = np.empty((cfg.n_strains + cfg.n_metabolites + cfg.n_toxins, len(t_eval)))
    for idx, y_out in out_chunks:
        traj_log[:, idx] = y_out
    return np.exp(traj_log.T)
