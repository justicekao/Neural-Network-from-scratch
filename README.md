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

Every process is a SEPARATE additive term in the population equation —
this is what "generalized" means here: switching a process off just
removes its term, it doesn't change the equations for anything else.

```
dN_i/dt = N_i * [ growth_i(metabolites) - toxin_kill_i(toxins) - mortality_i ]
          + mutation_i(other strains)         [optional]
          + translocation_i(other strains)    [optional]

dC_k/dt = supply_k - dilution_k * C_k - consumption by strains
dTox_l/dt = supply_l - decay_l * Tox_l + secretion by strains
```

- **growth**: saturable (Monod) response to each metabolite, own rate + half-saturation per (strain, metabolite) pair.
- **toxin_kill**: saturable response to each toxin, same structure as growth but subtractive.
- **mutation**: a strain spontaneously converts into another at a per-capita rate — doesn't require contact between strains.
- **translocation**: a strain converts another strain on CONTACT (mass-action, like conjugation/plasmid transfer) — does require both strains present.

Every rate above lives in its own matrix (e.g. `growth_rate` is strains × metabolites), and every entry in every matrix is independently either FIXED (you know its value) or FREE (the fitter estimates it) — set visually in the app.

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
│   ├── plotting.py         # observed-vs-fit plots
│   ├── app.py              # the visual Streamlit app (the main entry point)
│   └── cli.py               # `genmonod-app` console command
├── examples/
│   └── quickstart_example.py   # scripted (non-visual) end-to-end example
├── tests/
│   ├── test_physics.py
│   └── test_fitting.py
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
2. Edit each parameter matrix — blank = fit it, a number = fix it.
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

## 7. What you still need to do

1. If your real system needs a physical process beyond growth/toxin/mortality/mutation/translocation, add it as its own block in `physics.py` (each existing process is clearly separated so you can copy the pattern) and register its matrix in `fitting.py`'s `_MATRIX_ATTRS` list.
2. The amortized guesser (`amortized_model.py`) only kicks in once you have several stored fits of the same system shape — until then, fits just start from the bounds midpoint, which is normal.
3. Bounds for each parameter default to generic ranges in `config.py`'s `default_config` — tighten these once you have a sense of realistic values for your organism.

## 8. License

MIT — see `LICENSE`.
