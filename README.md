# monod-pinn

A physics-informed convolutional neural network (PI-CNN) that estimates the
parameters of a Monod growth-kinetics model directly from time-series
measurements of microbial strain populations (and, optionally, substrate
concentration).

This repository is a **starting template**, not a finished research product.
It is organized like a real installable Python package so you can push it to
GitHub, `pip install` it (locally or eventually from GitHub), and build on it.
Every file below has a comment block at the top explaining what it does and
whether you need to edit it.

---

## 1. The science (what the model is doing)

The classic Monod model describes microbial growth limited by a single
substrate:

```
dX/dt = mu_max * S/(Ks + S) * X          (biomass / population growth)
dS/dt = -(1/Y) * mu_max * S/(Ks + S) * X  (substrate consumption)
```

Where:
- `X(t)` = population / biomass at time t
- `S(t)` = substrate concentration at time t
- `mu_max` = maximum specific growth rate
- `Ks`   = half-saturation constant (substrate level at which growth rate is half of mu_max)
- `Y`    = yield coefficient (biomass produced per unit substrate consumed)

Normally you'd estimate `(mu_max, Ks, Y)` by nonlinear least-squares curve
fitting against a single dataset. Here, instead, a 1D CNN is trained to look
at a whole time-series curve and directly output the three parameters. It is
made **physics-informed** by adding a loss term that re-simulates the Monod
ODEs using the CNN's predicted parameters and penalizes the network when the
simulated trajectory doesn't match the observed data. This lets you (a) train
on many synthetic curves quickly, and (b) fine-tune / regularize using real
data even when you don't know the "true" parameters for that real curve.

## 2. Project layout

```
monod-pinn/
├── src/monod_pinn/
│   ├── __init__.py       # package version + convenience imports
│   ├── physics.py        # Monod ODE definitions + differentiable RK4 integrator
│   ├── models.py         # the 1D CNN architecture
│   ├── losses.py         # data loss + physics-residual loss
│   ├── dataset.py        # PyTorch Dataset + synthetic data generator
│   ├── train.py          # training loop
│   ├── evaluate.py       # plotting / evaluation helpers
│   └── utils.py          # seeding, normalization, checkpoint helpers
├── scripts/
│   ├── generate_synthetic_data.py   # CLI: make a synthetic training set
│   └── run_training.py              # CLI: train a model end-to-end
├── examples/
│   └── quickstart_example.py        # minimal runnable example
├── tests/
│   ├── test_physics.py
│   └── test_models.py
├── data/
│   └── README.md         # describes the CSV format expected for real data
├── pyproject.toml        # package metadata + dependencies
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 3. Installation

```bash
git clone https://github.com/<your-username>/monod-pinn.git
cd monod-pinn
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

The `-e` (editable) install means changes you make to files under `src/` are
picked up immediately without reinstalling — this is the standard way to
develop a Python package locally.

## 4. Quickstart

Generate a synthetic dataset, then train:

```bash
python scripts/generate_synthetic_data.py --n_curves 2000 --out data/synthetic.npz
python scripts/run_training.py --data data/synthetic.npz --epochs 100 --out checkpoints/
```

Or run the minimal end-to-end example:

```bash
python examples/quickstart_example.py
```

## 5. Using your own real data

Put a CSV per experiment (or one long-format CSV) in `data/` following the
format described in `data/README.md`, then point `dataset.py`'s
`RealCurveDataset` at it. See the comments in `dataset.py` for exactly what
columns are expected.

## 6. What you still need to do

This template runs out of the box on **synthetic** data. To make it useful
for your actual research you will likely need to:

1. Confirm the Monod variant in `physics.py` matches your system (e.g. add
   a death-rate term, multiple substrates, or a lag phase, if relevant).
2. Replace/extend `dataset.py`'s `RealCurveDataset` to parse your actual
   file format.
3. Tune the CNN architecture in `models.py` (kernel sizes, depth) once you
   know the typical length/noise level of your real time series.
4. Adjust the loss weighting in `losses.py` (`physics_weight`) — this is the
   most important hyperparameter in any physics-informed network.

Every file has a `# NEEDS YOUR INPUT` comment marking the specific lines
most likely to require changes for your data.

## 7. License

MIT — see `LICENSE`.
