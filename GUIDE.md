# genmonod — complete usage guide

Two ways to use this: the visual app (recommended for most work) or
scripting directly against the Python API (needed for things like
joint fits across dozens of datasets, or warm-started long-running
fits). This guide covers both, start to finish.

---

## Part 1 — The visual app

### 1. Install and launch

```bash
unzip genmonod.zip && cd genmonod
python -m venv .venv && source .venv/bin/activate
pip install -e .
genmonod-app
```

Opens in your browser. Everything below happens on that one page, top to bottom.

### 2. Sidebar — describe your system

- **Number of strains / metabolites / toxins**: counts only. 0
  metabolites means no growth mechanism at all (growth is always
  metabolite-driven) — you almost always want at least 1, even if
  it's unmeasured.
- **Include mutation**: strain-to-strain conversion that does NOT
  require contact (a spontaneous per-capita rate).
- **Include translocation**: strain-to-strain conversion that DOES
  require contact (mass-action, like one strain converting another on
  contact). Different mechanism from mutation, and different from
  conjugative transfer (see below) — translocation only handles ONE
  population converting into ANOTHER (two populations trading mass).
- **Rename strains / metabolites / toxins** (expander): give them real
  names (e.g. "Z1331" not "Strain_1"). This matters — it's what lets
  the strain library and joint fits correctly match the same real
  organism across different experiments.

Changing any count rebuilds the system from scratch and **resets your
matrix edits** — set dimensions first, then edit parameters.

### 3. Parameters — five tabs, one editable grid per matrix

In every grid: **leave a cell blank to fit it, type a number to fix
it at that value.** This is the entire mechanism for encoding what you
already know vs. what you want the fit to determine.

- **Growth / Toxin (r, K, c) tab**: `r` (growth rate constant), `K`
  (Monod constant, shared between growth and consumption), and `c`
  (consumption/uptake rate) — all strains × (metabolites then toxins,
  one combined axis). A toxin is just a substrate with a negative `r`;
  there's no separate toxin-specific matrix anymore.
- **Subsets tab**: which substrates share a saturation denominator
  (compete for a strain's uptake capacity) — this is what "metabolic
  overlap" means precisely in this model. Pick a strain, select 2+
  substrates to group together. Leaving everything ungrouped (the
  default) makes every substrate saturate independently — mathematically
  identical to a plain summed single-substrate Monod model. Group a
  toxin with a metabolite to represent "toxin competes with metabolites
  for uptake" instead of an independent supply.
- **Production tab**: register a strain synthesizing one substrate from
  another (pick product, producing strain, precursor). This is its own
  saturating term with its own Monod constant — different from growth,
  which responds to a substrate but doesn't necessarily make more of
  something else from it.
- **Strain-strain tab**: mortality (per strain), plus mutation and/or
  translocation matrices if you turned them on. Diagonal entries are
  always forced to 0 regardless of what you type (no self-conversion).
- **Environment tab**: metabolite/toxin supply and dilution/decay
  rates (these aren't strain-specific), plus **discrete events**
  (dosing and passage — see below).

### Discrete events — how metabolite (and populations) get introduced

Continuous supply/dilution models a steady, ongoing process (a true
chemostat feed). Real protocols are often discrete instead — a single
addition, or repeated dosing/passage at specific times. Add these from
the Environment tab, or directly:

```python
from genmonod.config import add_dosing_event, add_passage_event

# a one-time addition to a SPECIFIC metabolite/toxin (e.g. sugar added at t=24h)
add_dosing_event(cfg, time=24, target="Glucose", kind="add", amount=10.0)
# kind="set" jumps to exactly `amount` instead of adding to the current value

# a passage/dilution step -- dilutes EVERY state (strains too, not just
# the metabolite) by a factor, e.g. a 10-fold dilution into fresh medium
add_passage_event(cfg, time=24, dilution_factor=0.1)
```

This is what continuous supply/dilution can't do: `metabolite_dilution`
only ever affects the metabolite, never the strain populations. A real
repeated-passage protocol dilutes the whole culture — cells and all —
which is exactly what `add_passage_event` represents. Combine a
`PassageEvent` and a `DosingEvent` at the same timestamp to represent
"diluted, and fresh substrate was added" (dilution is applied first,
then the dose — matching how fresh medium carries substrate into an
already-diluted culture). Register one pair per real dilution/passage
time in your protocol for a repeated (chemostat-style) regime.

**NEEDS YOUR INPUT**: `amount`/`dilution_factor` are fixed at whatever
you specify, not fit as free parameters — the common case is that you
designed the protocol yourself and know these values exactly. If you
genuinely don't know them and want them estimated from data, that's a
natural extension (similar to how `add_conjugative_transfer`'s rate can
be left free) that isn't built yet.

Below the tabs: **"Fill known strains from library"** — pulls in fixed
values for any strain NAME that matches a strain you've fit before
(see Part 3). Only fills entries you left free; never overrides
something you explicitly fixed.

### 4. Data — upload and map columns

Upload one or more CSVs. Multiple files are fit **jointly** — one
shared parameter set explaining all of them at once, which is almost
always what you want (see "why joint fitting matters" below).

You'll be asked which column is time, then for each strain/metabolite/
toxin in your system, which CSV column matches it (or "None" if that
state isn't measured in this data — its initial condition will then
be FIT rather than assumed, so growth on an unmeasured resource still
works). This mapping is set up once from the first file and applied to
every file, so **all uploaded files need the same column names**.

### 5. Structure search (optional, expander under Data)

Not sure whether toxin or a shared-vs-separate metabolite pool
belongs in your model? Check the boxes for what to search, hit **"Run
structure search"**, and it fits every combination and ranks them by
AICc (lower is better). A warning appears if the winning structure
doesn't have enough data to trust the ranking — take that seriously;
it means fit more replicates jointly before drawing conclusions, not
that the search is broken.

### 6. Fit

Three checkboxes above the **Run fit** button:
- **Use past fits to propose a starting guess**: only kicks in once
  you have 8+ prior fits of this exact system shape (same
  strain/metabolite/toxin counts and mutation/translocation flags).
  Before that, fits just start from the middle of each parameter's
  bounds, which is normal.
- **Save this fit for future guessing**: feeds the above.
- **Save this fit's strain values to the library**: feeds "Fill known
  strains from library" in step 3, for future larger systems.

After fitting: a plot per dataset (observed points vs. fitted curve),
and the resolved (no-longer-free) parameter matrices below.

---

## Part 2 — Scripting (for things the app doesn't cover)

The app doesn't currently expose: fitting more than a handful of
datasets jointly (large joint fits can take a while — see the
warm-start pattern below), or registering a conjugative-transfer
process (donor + recipient → a third population, e.g. a
transconjugant — different from translocation, see Part 1 step 2).

### Basic scripted fit

```python
from genmonod.config import default_config
from genmonod.data_io import Dataset
from genmonod.fitting import fit

cfg = default_config(n_strains=2, n_metabolites=1, n_toxins=0)
cfg.strain_names = ["Recipient", "Donor"]
cfg.r.set_fixed(0.5, 0, 0)  # fix what you know; leave the rest free

ds = Dataset.from_csv("data.csv", "Time", {"Recipient": "OD_r", "Donor": "OD_d"}, cfg)
result = fit(cfg, ds)
print(result.config.r.values)  # resolved matrix
```

### Metabolic overlap and toxin competition (subsets)

```python
from genmonod.config import set_subsets

# two metabolites competing for the same strain's uptake capacity
set_subsets(cfg, "StrainA", [["Glucose", "Fructose"]])

# a toxin competing with a metabolite for uptake, instead of an
# independent supply (the default)
set_subsets(cfg, "StrainA", [["Glucose", "Bilirubin"]])
```

Anything NOT mentioned in a group keeps its own independent singleton
subset. This is the precise mechanism behind the "shared vs. separate
metabolite pool" axis `compare_structures` searches automatically (see
below) — you can also build these candidates by hand for more specific
overlap patterns than a full on/off toggle.

### A strain producing one substrate from another

```python
from genmonod.config import add_production

# StrainB synthesizes a toxin from a metabolite precursor
add_production(cfg, product="ToxinX", strain="StrainB", precursor="Glucose")
# rate and half_sat are both free by default; pass rate=... / half_sat=...
# to fix either one if you already know it
```

### Donor + recipient → transconjugant (conjugative transfer)

```python
from genmonod.config import add_conjugative_transfer

add_conjugative_transfer(cfg, product="Transconjugant", donor="Donor",
                          recipient="Recipient", lower=0.0, upper=1e-9)
```

Bounds matter a lot here — this rate multiplies two population sizes
together. At real cfu/mL scale (~1e8–1e9), an unscaled bound (like the
package's other default bounds) makes the ODE catastrophically stiff.
A rough starting bound: `upper ≈ max_observed_product / (median_donor
× median_recipient × experiment_duration) × 100`.

### Joint fitting across many datasets, with warm-starting

For a large joint fit (many datasets, many free parameters), a single
optimization pass can take a long time. `fit()` accepts `init_guess`
so you can run it in short bursts, checkpointing between them:

```python
import json
import numpy as np
from genmonod.fitting import fit

# first pass
result = fit(cfg, datasets, max_nfev=100)
json.dump({"x": result.x.tolist()}, open("checkpoint.json", "w"))

# resume later / in a new process
prev = json.load(open("checkpoint.json"))
result = fit(cfg, datasets, init_guess=np.array(prev["x"]), max_nfev=100)
```

Repeat until the cost stops improving between rounds.

### Structure search from a script

```python
from genmonod.model_selection import compare_structures, summarize

def builder(candidate_cfg):
    # called once per candidate -- "shared" vs "separate" metabolite
    # candidates need different numbers of metabolite columns, so
    # rebuild the Dataset against each candidate's own cfg
    return Dataset.from_csv("data.csv", "Time", column_map, candidate_cfg)

results = compare_structures(builder, n_strains=3, strain_names=["A", "B", "C"])
print(summarize(results))
best_cfg = results[0].result.config
```

### Strain library, directly

```python
from genmonod.strain_library import record_strain_params, apply_library

record_strain_params(result, store_path="my_library.jsonl")  # after a trusted fit

# later, building a bigger system:
big_cfg, filled = apply_library(big_cfg, store_path="my_library.jsonl")
```

---

## Part 3 — Exploring behavior across parameter space (not fitting)

Everything above is about fitting a model TO data. This is the
opposite direction: given parameters, what does the system DO — and
how does that change as you vary two of them? There's no data, no
Dataset, no `fit()` call here.

### One trajectory from specific parameter values

`set_param` sets any single parameter by name — matrix entries
(`r`, `K`, `c`, `mortality`, `mutation`, `translocation`, supply/dilution/decay)
or list entries (`conjugative_transfer`, `production`) — then just call
`simulate` directly:

```python
from genmonod.config import default_config
from genmonod.sweep import set_param
from genmonod.physics import simulate

cfg = default_config(n_strains=1, n_metabolites=1, n_toxins=0)
set_param(cfg, "r", (0, 0), 0.5)          # matrix entry: (row, col)
set_param(cfg, "mortality", 0, 0.05)       # 1D matrix: just the row
# for conjugative_transfer/production (lists), index is (entry_position, field_name):
# set_param(cfg, "conjugative_transfer", (0, "rate"), 1e-9)

traj = simulate(cfg, y0=[1.0, 10.0], t=[0, 1, 2, 3, 4, 5])
```

Every entry in `cfg` needs a real number here (not left free/NaN) —
this is simulation, not fitting, so there's nothing to estimate.

### Sweeping two parameters over a grid

```python
from genmonod.sweep import ParamAxis, run_sweep, heatmap, trajectory_grid

x_axis = ParamAxis(attr="r", index=(0, 0), values=np.linspace(0, 1, 25), label="growth rate")
y_axis = ParamAxis(attr="mortality", index=0, values=np.linspace(0, 1, 25), label="mortality")

result = run_sweep(cfg, y0, t, x_axis, y_axis)

# red/black heatmap: define ANY criteria on the trajectory
def survives(traj, t, state_names):
    return traj[-1, 0] > traj[0, 0]  # ended up bigger than it started

fig = heatmap(result, survives, title="Does the population survive?")

# grid of actual time-series plots, one per (x, y) cell -- use a
# COARSER grid for this than the heatmap (see below)
result_coarse = run_sweep(cfg, y0, t, x_axis_coarse, y_axis_coarse)
fig2 = trajectory_grid(result_coarse)
```

See `examples/sweep_example.py` for a full worked example (transfer
rate vs. recipient mortality, "does the transconjugant establish").

**Two grid sizes for two different outputs.** A heatmap only needs one
number per cell, so it scales fine to a 25x25 or larger grid. A
trajectory grid needs a whole readable subplot per cell, so keep it to
roughly 5x5–6x6 — run the sweep twice, once fine for the heatmap, once
coarse for the trajectory grid (both cheap, since simulation is much
faster than fitting).

**Memory**: `run_sweep(..., store_trajectories=True)` (the default)
keeps every full trajectory in memory — fine for the sizes above, but
if you're running something much larger and only need the heatmap, pass
`store_trajectories=False`; your criteria function then receives just
the final state (`criteria_fn(final_state, state_names)`) instead of
the full trajectory.

A cell where integration fails outright is recorded in `result.failed`
and marked with a white X on the heatmap, rather than crashing the
whole sweep.

## Why joint fitting matters (read this before fitting one file at a time)

A single small experiment (say, 3 strains × 4 timepoints = 12 data
points) very often has FEWER data points than free parameters — the
fit will converge, but the resulting parameter values aren't
trustworthy, even though the curve looks fine. Fitting multiple
datasets that share a real strain identity jointly (same strain name →
same parameters, enforced automatically) multiplies your effective
data without multiplying your free parameters nearly as fast, which is
usually the difference between a fit you can trust and one you can't.
`compare_structures`' `.reliable` flag and the AICc score are there
specifically to catch this — don't ignore a "not reliable" warning
just because the fitted curve looks good.

## Key gotchas

- **Mutation/translocation identifiability**: with only population
  counts (not strain-tagged data), a pairwise strain-strain process is
  only identifiable up to its NET direction. If you leave both
  `mutation[i,j]` and `mutation[j,i]` (or translocation) free, the fit
  can converge perfectly while individual rates are meaningless. Fix
  the direction you know is zero.
- **Consumption/uptake at real population scale**: freely fitting `c`
  at cfu/mL scale (~1e8+) can make the ODE pathologically stiff. The
  structure-search and joint-fit code in this package fixes `c` at 0
  by default for exactly this reason — this now applies to toxin
  uptake too, not just metabolite consumption. Only turn it on with a
  properly scaled, tight bound if resource depletion is the actual
  thing you're studying.
- **Production rate vs. half-saturation tradeoff**: like growth's `r`
  vs. `K`, a production pathway's `rate` and `half_sat` can trade off
  against each other if the precursor's concentration doesn't vary
  much across your data — the fit can match the curve essentially
  exactly while the individual values aren't uniquely pinned down.
  Fix one if you know it.
- **AICc `inf`**: shown when a candidate has too few observations
  relative to its free parameters for the correction term to be
  computable — this is the search telling you the comparison isn't
  meaningful yet, not an error.
- **Continuous vs. discrete environment terms**: `metabolite_supply`/
  `metabolite_dilution` and a `schedule` of dosing/passage events can
  both be active on the same config at once (e.g. a small continuous
  baseline plus periodic top-ups) — they're not mutually exclusive,
  just different mechanisms for different real protocols.
