# data/

Nothing here is committed except this README (see `.gitignore`).

Put your own experiment CSVs here. There's no fixed column-name format
required — unlike a fixed schema, the app asks you to map each CSV
column to a strain/metabolite/toxin interactively when you upload it
(Section "Data" in the app), so any column names work.

A minimal example CSV might look like:

```csv
Time,OD_strainA,OD_strainB,Glucose_mM
0,0.02,0.01,10.0
4,0.08,0.03,7.2
8,0.25,0.11,3.8
12,0.55,0.30,1.1
16,0.70,0.52,0.2
```

When you upload this in the app, you'd map:
- Time column: `Time`
- `Strain_1`: `OD_strainA`
- `Strain_2`: `OD_strainB`
- `Metabolite_1`: `Glucose_mM`

If you have multiple separate experiments/runs, keep them as separate
CSV files and upload/fit them one at a time, or extend `data_io.py` to
load several at once for a joint fit (see `fitting.fit()`, which already
accepts a list of Datasets).
