# genmonod

A generalized, multi-strain Monod kinetics fitter with a visual matrix
editor and a fit-history-based initial-guess model. This is a from-
scratch Python redesign of an earlier MATLAB fitting tool, built to (a)
fix bugs in that tool, and (b) generalize it — arbitrary strain counts,
optional mutation/translocation between any strains, optional toxins,
and any parameter fixed or free, all configured visually rather than by
editing code.

---

## 1. The model

Growth and consumption/uptake follow a SHARED-DENOMINATOR multi-substrate
Monod form derived from a proteome-allocation model of microbial growth
(Piskovsky, Schnepp-Pesch & Foster; extended to multi-strain systems with
toxins in "Modelling Microbe Population Dynamics with Monod Equations
Grounded in Statistical Mechanics") — NOT independently-summed
single-substrate saturation curves. A toxin is mathematically just a
substrate with a negative rate constant: same equations as a metabolite,
whether it competes with metabolites for uptake or has its own
independent supply.

```
dN_i/dt = N_i * [ growth_i(substrates) - mortality_i ]
          + mutation_i(other strains)         [optional]
          + translocation_i(other strains)    [optional]

dC_k/dt = supply_k - dilution_k*C_k + production_k(precursors) - uptake by strains
dTox_l/dt = supply_l - decay_l*Tox_l + production_l(precursors) - uptake by strains
```

**growth**, for strain *k*, is a sum over "subsets" *w* of substrates that
share *k*'s limited uptake capacity:

```
growth_k = Σ_w  [ Σ_(a in w) r[k,a] * x_a/K[k,a] ]  /  [ 1 + Σ_(a in w) x_a/K[k,a] ]
```

Two substrates in the SAME subset compete (one shared denominator — more
of one leaves less "room" for the other, saturating the strain's total
uptake capacity together). Substrates in DIFFERENT subsets are fully
independent. **By default every substrate is its own singleton subset**
for every strain — which makes this exactly equivalent to plain
single-substrate Monod growth summed independently; you only get
competition where you explicitly declare it with `set_subsets`. This is
the precise definition of "metabolic overlap" in this model, and it
subsumes toxins too: put a toxin in the same subset as a metabolite for
"toxin competes with metabolites for uptake"; leave it in its own
subset (the default) for "toxin has an independent supply."

**consumption/uptake** of substrate *a* by strain *k* (in subset *w*)
uses the SAME shared denominator as growth: `c[k,a]*x_a/K[k,a] / (1 + Σ_(a' in w) x_a'/K[k,a'])`.

**production** is a strain SYNTHESIZING one substrate from a precursor
(register with `add_production`) — its own saturating term, using a
production-specific Monod constant (distinct from the precursor's
growth-relevant K), since production and growth needn't saturate at the
same precursor level.

- **mortality**: flat per-strain death rate, independent of substrates.
- **mutation**: a strain spontaneously converts into another at a per-capita rate — doesn't require contact between strains.
- **translocation**: a strain converts another strain on CONTACT (mass-action, like conjugation/plasmid transfer) — does require both strains present.
- **conjugative transfer** (`add_conjugative_transfer`): donor + recipient contact creates a genuinely THIRD population (e.g. a transconjugant) without depleting either parent — different from translocation, which only trades mass between two populations.

Every rate lives in its own matrix (`r`, `K`, `c` are strains × combined-substrate-axis, where the combined axis is metabolites then toxins together), and every entry in every matrix is independently either FIXED (you know its value) or FREE (the fitter estimates it) — set visually in the app.

## 2. Project layout

```
genmonod/
├── src/genmonod/
│   ├── config.py           # SystemConfig + MatrixSpec: the "constraints" object
│   ├── physics.py          # the generalized ODE system
│   ├── data_io.py          # CSV loading with explicit column mapping
│   ├── fitting.py          # pack/unpack free params, scipy least_squares wrapper
│   ├── fit_store.py        # persists past fits to a local file
│   ├── amortized_model.py  # small MLP trained on stored fits -> better initial guesses
│   ├── strain_library.py   # carries fitted values forward by strain name into larger configs
│   ├── model_selection.py  # automated structure search (toxin/overlap/etc.), ranked by AICc
│   ├── sweep.py             # explore behavior across parameter space: single trajectories, 2D sweeps, heatmaps
│   ├── plotting.py         # observed-vs-fit plots
│   ├── app.py              # the visual Streamlit app (the main entry point)
│   └── cli.py               # `genmonod-app` console command
├── examples/
│   ├── quickstart_example.py   # scripted (non-visual) end-to-end fitting example
│   └── sweep_example.py         # 2D parameter sweep: heatmap + trajectory grid
├── tests/
│   ├── test_physics.py
│   ├── test_fitting.py
│   ├── test_model_selection.py
│   └── test_sweep.py
├── data/
│   └── README.md           # CSV format notes
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 3. Installation

```bash
git clone https://github.com/<your-username>/genmonod.git
cd genmonod
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

## 4. Running the visual app

```bash
genmonod-app
```

This opens the tool in your browser. From there:
1. Set strain/metabolite/toxin counts and toggle mutation/translocation in the sidebar.
2. Edit `r`/`K`/`c` — blank = fit it, a number = fix it. Use the **Subsets** tab to declare metabolic overlap (which substrates compete for a strain's uptake) and the **Production** tab for any strain that synthesizes one substrate from another.
3. Upload one or more CSVs (multiple files are fit **jointly** — one shared parameter set explaining all of them, e.g. several replicate tubes) and map their columns to your strains/metabolites/toxins.
4. Click **Run fit**. Results (fitted matrices + plot) appear below.
5. Each fit is optionally saved so future fits of the *same system shape* (same strain/metabolite/toxin counts and mutation/translocation settings) get a better starting guess automatically — the more you use it, the better the initial guesses get.

## 5. Scripting instead of using the app

```bash
python examples/quickstart_example.py
```

See that file for the pattern: `default_config(...)` → optionally `spec.set_fixed(value, i, j)` on anything you know → `Dataset(...)` → `fit(cfg, dataset)`.

## 6. A real modeling gotcha worth knowing up front

If you leave BOTH directions of a mutation or translocation pair free
(e.g. `mutation[0,1]` and `mutation[1,0]` both unfixed), the fit can
converge perfectly on the data while landing on individual rates that
don't match reality — only their *net difference* is actually
identifiable from population-count data. If you know a transfer only
goes one way, fix the reverse entry to `0` (see
`examples/quickstart_example.py`). If you need both directions
separately, you'll need data that distinguishes the strains after
transfer (e.g. a third "transconjugant" population, the way the
original MATLAB tool modeled donor/recipient/transconjugant as three
separate observed states rather than a two-strain conversion).

## 7. Which structure fits best? Let it search

Instead of guessing whether toxin/metabolic-overlap/etc. should be in your
model, `model_selection.compare_structures` fits several structural variants
against your data and ranks them by AICc (small-sample-corrected AIC — plain
AIC gets unreliable once free parameters approach your observation count,
which is common with small biological datasets). It's wired into the app
under "Search structures automatically" once you've uploaded data, or usable
directly:

```python
from genmonod.model_selection import compare_structures, summarize
from genmonod.data_io import Dataset

def builder(cfg):
    # called once per candidate structure -- "shared" vs "separate" metabolite
    # candidates need a different number of metabolite columns, so the
    # dataset has to be rebuilt against each candidate's own cfg
    return Dataset.from_csv("my_data.csv", "Time", column_map, cfg)

results = compare_structures(builder, n_strains=3, strain_names=["A", "B", "C"])
print(summarize(results))  # best (lowest AICc) first
```

Every result carries a `.reliable` flag — when a candidate has too few
observations relative to its free parameters, AICc is flagged as
unreliable rather than silently ranked as if trustworthy. With a single
small dataset this fires often; it's a real signal to fit jointly across
more replicates (see the section above) before trusting a structural
conclusion, not a bug in the search itself.

## 8. What you still need to do

1. If your real system needs a physical process beyond growth/toxin/mortality/mutation/translocation, add it as its own block in `physics.py` (each existing process is clearly separated so you can copy the pattern) and register its matrix in `fitting.py`'s `_MATRIX_ATTRS` list.
2. The amortized guesser (`amortized_model.py`) only kicks in once you have several stored fits of the same system shape — until then, fits just start from the bounds midpoint, which is normal.
3. Bounds for each parameter default to generic ranges in `config.py`'s `default_config` — tighten these once you have a sense of realistic values for your organism.

## 9. License

MIT — see `LICENSE`.
