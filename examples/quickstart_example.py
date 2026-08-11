"""
examples/quickstart_example.py

Shows the core package working end-to-end WITHOUT the visual app —
useful as a sanity check after install, or as a starting point for
scripting fits yourself. Run with:

    python examples/quickstart_example.py

This mirrors realistic use: you usually already know roughly what the
growth/toxin/environment rates are (from prior experiments or
literature), and the thing you're actually trying to determine is one
specific structural question — here, "does strain 1 convert strain 0 on
contact, and at what rate?" So we FIX everything except the
translocation matrix, which is exactly what `set_fixed` is for.

NOTE ON IDENTIFIABILITY (important, read this if translocation/mutation
fits look "wrong" but the curve fit looks good): with only population
counts as data (not strain-tagged measurements), a PAIRWISE contact
process's forward and reverse rates aren't separately identifiable from
population trajectories alone — only their NET difference is, because
that's all that actually shows up in how the populations move. If you
leave both translocation[0,1] and translocation[1,0] free, the fitter
can land on a pair of values whose difference is correct even though
neither individual value matches what generated the data. This isn't a
bug — it's the same reason the original MATLAB tool tracked donor /
recipient / transconjugant as three SEPARATE observable populations
rather than two: that's what makes a transfer direction and rate
uniquely identifiable from data, because you can directly see the third
population appear from (near) zero. The practical fix, used below: if
you know a transfer only happens in one direction, FIX the reverse
entry to 0 rather than leaving both free.

NEEDS YOUR INPUT: nothing, this runs as-is.
"""

import numpy as np

from genmonod.config import default_config
from genmonod.physics import simulate
from genmonod.data_io import Dataset
from genmonod.fitting import fit

KNOWN_VALUE = 0.3  # a stand-in for "parameters you already know from prior work"

# 1. Build a 2-strain, 1-metabolite, 1-toxin system with translocation on,
#    then FIX everything except translocation — those are the unknowns
#    we're actually trying to determine here.
cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=1, include_translocation=True)
for attr in ["growth_rate", "growth_half_sat", "consumption", "toxin_kill_rate",
             "toxin_half_sat", "secretion", "mortality",
             "metabolite_supply", "metabolite_dilution", "toxin_supply", "toxin_decay"]:
    spec = getattr(cfg, attr)
    for idx in np.ndindex(spec.shape):
        spec.set_fixed(KNOWN_VALUE, *idx)
# translocation: we know it's one-directional (strain 1 -> strain 0 only),
# so fix the reverse entry to 0 and leave only the real unknown, [1,0], free.
# See the identifiability note above for why this matters.
cfg.translocation.set_fixed(0.0, 0, 1)

# 2. Make up "true" data by simulating the same known parameters plus a
#    real translocation rate (in real use, this step is replaced by
#    loading your actual measurements instead)
true_cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=1, include_translocation=True)
for attr in ["growth_rate", "growth_half_sat", "consumption", "toxin_kill_rate",
             "toxin_half_sat", "secretion", "mortality",
             "metabolite_supply", "metabolite_dilution", "toxin_supply", "toxin_decay"]:
    spec = getattr(true_cfg, attr)
    spec.values[np.isnan(spec.values)] = KNOWN_VALUE
true_cfg.translocation.values[:, :] = 0.0
true_cfg.translocation.values[1, 0] = 0.2  # strain 1 converts strain 0 on contact

y0 = np.array([1.0, 0.01, 10.0, 0.1])
t = np.linspace(0, 15, 40)
observed = simulate(true_cfg, y0, t)
dataset = Dataset(t=t, Y=observed, y0=y0, name="synthetic_example")

# 3. Fit — only the 2 free translocation entries (off-diagonal) are estimated
result = fit(cfg, dataset)

print("Fit success:", result.success, "cost:", result.cost)
print("Fitted translocation matrix (true value at [1,0] was 0.2, [0,1] was 0.0):")
print(result.config.translocation.values)
