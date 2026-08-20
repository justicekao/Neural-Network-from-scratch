"""
examples/sweep_example.py

A worked example of the parameter-sweep tool: for a donor/recipient/
transconjugant system, sweep the conjugative transfer rate (x-axis)
against the recipient's mortality (y-axis), and show:
  1. A red/black heatmap: red where the transconjugant population
     establishes (exceeds a threshold by the end), black where it
     doesn't.
  2. A grid of the actual population trajectories for a smaller version
     of the same sweep, so you can see WHY each cell looks the way it does.

Run with:
    python examples/sweep_example.py
Produces sweep_heatmap.png and sweep_trajectories.png in the current directory.

NEEDS YOUR INPUT: nothing to run as-is. Swap in your own two parameters
(any two args accepted by genmonod.sweep.set_param) and your own
criteria function to explore a different question.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from genmonod.config import default_config, add_conjugative_transfer
from genmonod.sweep import ParamAxis, run_sweep, heatmap, trajectory_grid

# --- 1. Build the base system: fix everything except what we're sweeping ---
cfg = default_config(n_strains=3, n_metabolites=1, n_toxins=0)
cfg.strain_names = ["Recipient", "Donor", "Transconjugant"]
for i in range(3):
    cfg.r.set_fixed(0.4, i, 0)
    cfg.K.set_fixed(1.0, i, 0)
    cfg.c.set_fixed(0.0, i, 0)          # numerical safety at any population scale, see GUIDE.md
cfg.metabolite_supply.set_fixed(0.0, 0)
cfg.metabolite_dilution.set_fixed(0.0, 0)
add_conjugative_transfer(cfg, product="Transconjugant", donor="Donor", recipient="Recipient")
# leave conjugative_transfer's rate to be swept (x-axis below), and leave
# Recipient's mortality to be swept too (y-axis below) -- everything else
# above is fixed to a known value since this is simulation, not fitting.
cfg.mortality.set_fixed(0.05, 1)  # Donor
cfg.mortality.set_fixed(0.05, 2)  # Transconjugant

y0 = np.array([1.0, 1.0, 1e-6, 10.0])  # transconjugant starts near zero
t = np.linspace(0, 8, 40)  # shorter window so the sweep shows a real establishment/extinction boundary

# --- 2. Heatmap: fine grid, only need the final state per cell ---
x_axis = ParamAxis(attr="conjugative_transfer", index=(0, "rate"),
                    values=np.linspace(0.0, 0.05, 25), label="conjugative transfer rate")
y_axis = ParamAxis(attr="mortality", index=0,
                    values=np.linspace(0.0, 1.2, 25), label="recipient mortality")

result = run_sweep(cfg, y0, t, x_axis, y_axis)


def transconjugant_establishes(traj, t, names):
    idx = names.index("Transconjugant")
    return traj[-1, idx] > 0.05  # arbitrary establishment threshold for this example


fig1 = heatmap(result, transconjugant_establishes,
               title="Does the transconjugant population establish?")
fig1.savefig("sweep_heatmap.png", dpi=150)
print("saved sweep_heatmap.png")

# --- 3. Trajectory grid: coarser grid (5x5), so each cell's plot stays readable ---
x_axis_coarse = ParamAxis(attr="conjugative_transfer", index=(0, "rate"),
                           values=np.linspace(0.0, 0.05, 5), label="transfer rate")
y_axis_coarse = ParamAxis(attr="mortality", index=0,
                           values=np.linspace(0.0, 1.2, 5), label="recipient mortality")
result_coarse = run_sweep(cfg, y0, t, x_axis_coarse, y_axis_coarse)

fig2 = trajectory_grid(result_coarse)
fig2.savefig("sweep_trajectories.png", dpi=150)
print("saved sweep_trajectories.png")
