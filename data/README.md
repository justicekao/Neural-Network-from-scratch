# data/

This folder is where you put datasets. Nothing here is committed to git
except this README and this folder itself (see `.gitignore`) — generated
and real data files should stay local / be tracked separately (e.g. with
Git LFS or a data-versioning tool) if they're large.

## Synthetic data

Created by `scripts/generate_synthetic_data.py`, saved as a `.npz` file.
You don't need to hand-edit anything for synthetic data.

## Real data — expected CSV format

`RealCurveDataset` (in `src/monod_pinn/dataset.py`) expects a single CSV
with one row per (curve, timepoint):

| column     | meaning                                                        |
|------------|-----------------------------------------------------------------|
| `curve_id` | identifier grouping rows into one experimental run/strain curve |
| `t`        | time of measurement (must be evenly spaced within a curve_id)   |
| `X`        | measured population/biomass at time t                           |
| `S`        | measured substrate concentration at time t (optional)           |
| `X0`       | initial population for that curve (same value on every row)     |
| `S0`       | initial substrate for that curve (same value on every row)      |

Example:

```csv
curve_id,t,X,S,X0,S0
strainA_rep1,0,0.02,5.0,0.02,5.0
strainA_rep1,1,0.05,4.8,0.02,5.0
strainA_rep1,2,0.11,4.3,0.02,5.0
strainB_rep1,0,0.01,8.0,0.01,8.0
strainB_rep1,1,0.02,7.9,0.01,8.0
```

If you don't measure substrate concentration, you can omit the `S` column
(or leave it as zeros) and pass `has_substrate=False` (the default) when
constructing `RealCurveDataset` — the physics-informed loss only compares
against the population channel in that case.

**NEEDS YOUR INPUT:** if your raw export from the lab/instrument uses
different column names or a wide format (one column per curve instead of
long format), either rename columns before loading, or edit the
`COL_*` constants and the `__init__` method of `RealCurveDataset` in
`src/monod_pinn/dataset.py` to match.
