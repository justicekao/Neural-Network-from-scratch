"""
physics.py

The generalized Monod ODE system. Every process (metabolite growth,
toxin kill, mortality, mutation, translocation) is a SEPARATE additive
term, each driven by its own matrix from config.py — this is what makes
it "generalized": turning a process off just means its matrix is absent
(config.include_mutation=False etc.), not a different set of equations.

State is tracked in log-space (log N, log C, log Tox) purely so that
populations/concentrations can never go negative during numerical
integration, and so scipy's stiff solvers behave well over many orders
of magnitude — this mirrors the approach in the original MATLAB tool,
which was a sound choice; what's different here is the RHS below is
fully vectorized and general (no hardcoded strain indices).

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


def _saturation(conc: np.ndarray, half_sat: np.ndarray) -> np.ndarray:
    """Monod saturation term conc/(half_sat + conc), broadcasting (T,) against (S,T)."""
    return conc[None, :] / (half_sat + conc[None, :] + _EPS)


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

    growth_percapita = np.zeros(S)      # (S,) total per-capita growth rate
    growth_ik = np.zeros((S, M))        # (S, M) per-(strain, metabolite) growth contribution
    if M > 0:
        sat = _saturation(C, cfg.growth_half_sat.values)          # (S, M)
        growth_ik = cfg.growth_rate.values * sat                   # (S, M)
        growth_percapita = growth_ik.sum(axis=1)                   # (S,)

    kill_percapita = np.zeros(S)
    kill_il = np.zeros((S, T))
    if T > 0:
        sat_t = _saturation(Tox, cfg.toxin_half_sat.values)         # (S, T)
        kill_il = cfg.toxin_kill_rate.values * sat_t                 # (S, T)
        kill_percapita = kill_il.sum(axis=1)

    mortality = cfg.mortality.values                                # (S,)

    # --- population dynamics: sum of independent processes ---
    dN = N * (growth_percapita - kill_percapita - mortality)

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

    dydt = np.zeros_like(y)
    dydt[:S] = dN / (N + _EPS)  # d(logN)/dt = dN/dt / N

    # --- metabolite dynamics ---
    if M > 0:
        supply = cfg.metabolite_supply.values
        dilution = cfg.metabolite_dilution.values
        consumed = (cfg.consumption.values * growth_ik).T @ N        # (M,)
        dC = supply - dilution * C - consumed
        dydt[S:S + M] = dC / (C + _EPS)

    # --- toxin dynamics ---
    if T > 0:
        supply_t = cfg.toxin_supply.values
        decay_t = cfg.toxin_decay.values
        secreted = (cfg.secretion.values * kill_il).T @ N            # (T,)
        dTox = supply_t - decay_t * Tox + secreted
        dydt[S + M:S + M + T] = dTox / (Tox + _EPS)

    return dydt


def simulate(
    cfg: SystemConfig, y0: np.ndarray, t_eval: np.ndarray,
    method: str = "RK45", rtol: float = 1e-5, atol: float = 1e-7,
) -> np.ndarray:
    """
    Simulate the system forward from y0 (linear-space initial conditions,
    length S+M+T) at the requested time points.

    Returns:
        traj: (len(t_eval), S+M+T) array in LINEAR space (already
              exponentiated back out of log-space for you).

    NEEDS YOUR INPUT: the defaults (RK45, rtol=1e-5) are chosen for
    SPEED, since fitting calls this hundreds of times per fit — this
    matters in practice, RK45 with these tolerances is roughly 2x faster
    per call than a tighter LSODA setup in testing, and every fit does a
    lot of calls. If your system is very stiff (e.g. a toxin kill rate
    much larger than everything else, or wildly different timescales
    between processes) and you see integration failures, pass
    `method="LSODA"` explicitly — it's slower but handles stiffness
    better. Raises if the solver fails (fitting.py catches this and
    penalizes that parameter set rather than crashing the whole fit).
    """
    y0_log = np.log(np.maximum(y0, _EPS))
    sol = solve_ivp(
        system_rhs, (t_eval[0], t_eval[-1]), y0_log, t_eval=t_eval,
        args=(cfg,), method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return np.exp(sol.y.T)
