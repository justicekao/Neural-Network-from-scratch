"""
scripts/generate_synthetic_data.py

Command-line tool to generate a synthetic Monod dataset and save it to a
.npz file for later reuse (so you don't have to regenerate it every time
you train).

Usage:
    python scripts/generate_synthetic_data.py --n_curves 2000 --out data/synthetic.npz

NEEDS YOUR INPUT: nothing to run it, but consider changing --n_curves,
--t_max, --n_timepoints to resemble your real experimental protocol
(e.g. how many hours you sample over, how many time points you collect).
"""

import argparse
import numpy as np

from monod_pinn.dataset import SyntheticMonodDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_curves", type=int, default=2000)
    parser.add_argument("--t_max", type=float, default=24.0)
    parser.add_argument("--n_timepoints", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/synthetic.npz")
    args = parser.parse_args()

    ds = SyntheticMonodDataset(
        n_curves=args.n_curves,
        t_max=args.t_max,
        n_timepoints=args.n_timepoints,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    np.savez(
        args.out,
        t=ds.t.numpy(),
        params=ds.params.numpy(),
        y0=ds.y0.numpy(),
        traj_clean=ds.traj_clean.numpy(),
        traj_noisy=ds.traj_noisy.numpy(),
    )
    print(f"saved {args.n_curves} synthetic curves to {args.out}")


if __name__ == "__main__":
    main()
