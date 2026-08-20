"""
sweep_overlap_example.py

Question: for a system matching "the third dataset" (Master_P14_Passage,
Z1331 x ESBL15, 15-species community -- the same system central to the
earlier "two transconjugation datasets combined" analysis), does
METABOLIC OVERLAP structure (whether two nutrient sources compete for a
strain's uptake capacity, vs. saturate independently) change the
predicted trajectories?

ALL parameter values below are ARBITRARY / illustrative -- loosely in
the range earlier fits found for this kind of system, but NOT fit to
real data here. This is a structural exploration, not a fit.

Since "metabolic overlap" is a discrete structural choice (two
substrates either share a saturation subset or they don't -- see
config.set_subsets), not a continuous number, this sweeps a REAL
continuous parameter (the second nutrient's supply rate -- standing in
for "how much cross-fed nutrient the 15-species community provides")
and compares overlap ON vs OFF at every point along it. That's the
actual way to test "do predictions come out the same regardless of
overlap structure."
"""

import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genmonod.config import default_config, add_conjugative_transfer, set_subsets
from genmonod.physics import simulate

# ---------------------------------------------------------------------------
# 1. Build the base system: 3 strains (matching the real donor/recipient/
#    transconjugant structure), 2 metabolites (primary nutrient + a
#    community cross-fed nutrient), conjugative transfer.
# ---------------------------------------------------------------------------
cfg = default_config(n_strains=3, n_metabolites=2, n_toxins=0)
cfg.strain_names = ["Recipient", "Donor", "Transconjugant"]
cfg.metabolite_names = ["Primary_Nutrient", "Community_Crossfeed"]

# ARBITRARY illustrative values -- loosely informed by earlier fits'
# magnitude (growth rates ~0.7-1.4/hr, mortality ~0.2-0.9/hr for this
# strain set) but chosen by hand here, not fit.
for i in range(3):
    cfg.r.set_fixed(1.0, i, 0)    # growth rate on primary nutrient
    cfg.r.set_fixed(0.6, i, 1)    # growth rate on cross-fed nutrient
    cfg.K.set_fixed(1.5, i, 0)
    cfg.K.set_fixed(1.5, i, 1)
    cfg.c.set_fixed(0.0, i, 0)    # numerical safety at real population scale (see GUIDE.md)
    cfg.c.set_fixed(0.0, i, 1)
cfg.mortality.set_fixed(0.4, 0)   # Recipient
cfg.mortality.set_fixed(0.4, 1)   # Donor
cfg.mortality.set_fixed(0.4, 2)   # Transconjugant
cfg.metabolite_supply.set_fixed(0.3, 0)   # primary nutrient: fixed baseline supply
cfg.metabolite_dilution.set_fixed(0.3, 0)
cfg.metabolite_dilution.set_fixed(0.3, 1)  # cross-feed nutrient's OWN dilution; its supply is swept below

add_conjugative_transfer(cfg, product="Transconjugant", donor="Donor", recipient="Recipient", rate=5e-10)

y0 = np.array([1.6e8, 3.2e3, 20.0, 5.0, 0.5])  # recipient/donor/transconjugant cfu/mL scale + both nutrients
t = np.linspace(0, 120, 60)  # matches the real system's 120hr passage window

# two structural variants, same parameters otherwise
cfg_independent = copy.deepcopy(cfg)  # default: every substrate its own singleton subset (no overlap)

cfg_overlap = copy.deepcopy(cfg)
for strain in cfg_overlap.strain_names:
    set_subsets(cfg_overlap, strain, [["Primary_Nutrient", "Community_Crossfeed"]])  # compete for shared uptake

# ---------------------------------------------------------------------------
# 2. Sweep the cross-feed nutrient's supply rate; compare final
#    transconjugant level under both structures at every point.
# ---------------------------------------------------------------------------
supply_values = np.linspace(0.0, 2.0, 20)
final_transconjugant_independent = []
final_transconjugant_overlap = []

for supply in supply_values:
    for variant, results_list in [(cfg_independent, final_transconjugant_independent),
                                    (cfg_overlap, final_transconjugant_overlap)]:
        trial = copy.deepcopy(variant)
        trial.metabolite_supply.set_fixed(supply, 1)
        traj = simulate(trial, y0, t)
        results_list.append(traj[-1, 2])  # Transconjugant is state index 2

final_transconjugant_independent = np.array(final_transconjugant_independent)
final_transconjugant_overlap = np.array(final_transconjugant_overlap)

max_rel_diff = np.max(np.abs(final_transconjugant_independent - final_transconjugant_overlap) /
                       np.maximum(final_transconjugant_independent, 1e-30))
print(f"Max relative difference in final transconjugant level across the whole sweep: {max_rel_diff:.4f}")
print("NOTE: this diverges hugely by t=120h -- see the analysis below for why that's a")
print("misleading way to compare the two structures (unconstrained exponential growth")
print("over a long window amplifies ANY per-capita rate difference into an astronomical")
print("one, which is a property of long unconstrained exponential growth, not really an")
print("insight about overlap specifically).")

# ---------------------------------------------------------------------------
# 2b. The FAIR, direct comparison: instantaneous per-capita growth rate as a
# function of substrate level, for both structures -- this is what actually
# differs, without the confound of compounding over a long time window.
# For a strain using two substrates with rates r1, r2:
#   independent: r1*x1/(K1+x1) + r2*x2/(K2+x2)   (each saturates separately)
#   overlapping: (r1*x1/K1 + r2*x2/K2) / (1 + x1/K1 + x2/K2)   (ONE shared denominator)
# These are NOT the same function in general -- they only coincide in the
# DILUTE limit (x1/K1, x2/K2 << 1, i.e. well below half-saturation), where
# both reduce to the same linear approximation r1*x1/K1 + r2*x2/K2.
# Independent growth is ALWAYS >= overlapping growth once both substrates
# matter (summing two capped-at-1 saturation curves can reach r1+r2; sharing
# one denominator caps the combined contribution more tightly).
# ---------------------------------------------------------------------------
r1, r2, K1, K2 = 1.0, 0.6, 1.5, 1.5
x2_fixed = 1.0  # hold the cross-feed nutrient at a fixed moderate level
x1_range = np.linspace(0.01, 15, 200)  # from well below K1 to well above it

growth_independent = r1 * x1_range / (K1 + x1_range) + r2 * x2_fixed / (K2 + x2_fixed)
growth_overlap = (r1 * x1_range / K1 + r2 * x2_fixed / K2) / (1 + x1_range / K1 + x2_fixed / K2)

fig0, ax0 = plt.subplots(figsize=(7, 5))
ax0.plot(x1_range, growth_independent, "-", color="#3a6ea5", label="independent (no overlap)")
ax0.plot(x1_range, growth_overlap, "--", color="#c0562f", label="overlapping (shared uptake)")
ax0.axvline(K1, color="gray", linestyle=":", linewidth=1, label=f"K1={K1} (half-saturation)")
ax0.set_xlabel("Primary_Nutrient concentration (x1)")
ax0.set_ylabel("instantaneous per-capita growth rate")
ax0.set_title("The actual comparison: growth rate vs. substrate level\n(both structures, same r/K, cross-feed nutrient held fixed)")
ax0.legend(fontsize=9)
fig0.tight_layout()
fig0.savefig("sweep_overlap_growth_rate_comparison.png", dpi=150)
print("saved sweep_overlap_growth_rate_comparison.png")

rel_diff_dilute = abs(growth_independent[0] - growth_overlap[0]) / growth_independent[0]
rel_diff_saturated = abs(growth_independent[-1] - growth_overlap[-1]) / growth_independent[-1]
print(f"\nAt x1={x1_range[0]:.2f} (dilute, well below K1={K1}): relative difference = {rel_diff_dilute:.4f}")
print(f"At x1={x1_range[-1]:.2f} (well above K1={K1}): relative difference = {rel_diff_saturated:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(supply_values, final_transconjugant_independent, "o-", label="independent (no overlap)", color="#3a6ea5")
axes[0].plot(supply_values, final_transconjugant_overlap, "x--", label="overlapping (shared uptake)", color="#c0562f")
axes[0].set_yscale("log")
axes[0].set_xlabel("cross-feed nutrient supply rate")
axes[0].set_ylabel("final Transconjugant (cfu/mL)")
axes[0].set_title("Overlap ON vs OFF, swept across nutrient supply")
axes[0].legend(fontsize=8)

# ---------------------------------------------------------------------------
# 3. Full trajectory comparison at one representative supply level
# ---------------------------------------------------------------------------
rep_supply = 1.0
names = ["Recipient", "Donor", "Transconjugant"]
colors = ["#3a6ea5", "#c0562f", "#2a7f62"]
for variant, ls, tag in [(cfg_independent, "-", "independent"), (cfg_overlap, "--", "overlap")]:
    trial = copy.deepcopy(variant)
    trial.metabolite_supply.set_fixed(rep_supply, 1)
    traj = simulate(trial, y0, t)
    for j, (name, c) in enumerate(zip(names, colors)):
        axes[1].plot(t, traj[:, j], ls, color=c, linewidth=2 if ls == "-" else 1.5,
                     label=f"{name} ({tag})" if j == 0 or True else None)
axes[1].set_yscale("log")
axes[1].set_xlabel("time (hours)")
axes[1].set_ylabel("cfu/mL")
axes[1].set_title(f"Full trajectories at supply={rep_supply} (solid=independent, dashed=overlap)")
handles = [plt.Line2D([0], [0], color=c, label=n) for n, c in zip(names, colors)]
axes[1].legend(handles=handles, fontsize=8)

fig.suptitle("Arbitrary-parameter system matching the P14 Passage (Z1331 x ESBL15) structure", fontsize=11)
fig.tight_layout()
fig.savefig("sweep_overlap_comparison.png", dpi=150)
print("saved sweep_overlap_comparison.png")
