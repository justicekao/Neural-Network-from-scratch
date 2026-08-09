"""
scripts/run_training.py

Command-line tool to train a MonodCNN. Trains on synthetic data generated
on the fly by default (no need to run generate_synthetic_data.py first,
though you can point --data at a saved .npz or a real CSV instead).

Usage examples:
    # quick synthetic run
    python scripts/run_training.py --epochs 50

    # train on a real CSV (see data/README.md for the format)
    python scripts/run_training.py --real_csv data/my_experiment.csv --epochs 50 --param_weight 0.0

NEEDS YOUR INPUT: when using --real_csv, param_weight is forced to 0
automatically (no ground-truth parameters exist for real data) — you don't
need to change anything, this is handled for you below.
"""

import argparse
import os

from monod_pinn.dataset import SyntheticMonodDataset, RealCurveDataset
from monod_pinn.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", type=str, default=None,
                         help="path to a real-data CSV; if omitted, trains on synthetic data")
    parser.add_argument("--has_substrate", action="store_true",
                         help="set if your real CSV includes substrate measurements")
    parser.add_argument("--n_curves", type=int, default=2000, help="synthetic dataset size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--physics_weight", type=float, default=1.0)
    parser.add_argument("--param_weight", type=float, default=0.1,
                         help="ignored (forced to 0) when using --real_csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    args = parser.parse_args()

    if args.real_csv:
        dataset = RealCurveDataset(args.real_csv, has_substrate=args.has_substrate)
        param_weight = 0.0  # no ground-truth parameters available for real data
    else:
        dataset = SyntheticMonodDataset(n_curves=args.n_curves)
        param_weight = args.param_weight

    if args.checkpoint:
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    train_model(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        physics_weight=args.physics_weight,
        param_weight=param_weight,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
