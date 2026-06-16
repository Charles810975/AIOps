"""
USAD Pipeline Runner
Usage:
  python usad_run.py                    # Full pipeline
  python usad_run.py --step train       # Training only
  python usad_run.py --step detect     # Detection only
"""

import argparse
import time
from pathlib import Path

from usad_train import train_usad
from usad_detect import run_detection


def main():
    parser = argparse.ArgumentParser(description="USAD Pipeline")
    parser.add_argument("--step", type=str, choices=["train", "detect", "both"], default="both")
    parser.add_argument("--exp_name", type=str, default="experiment_1")
    parser.add_argument("--data", type=str, default="data-collection/cart_cpu_combined.csv")
    parser.add_argument("--pods", type=str, nargs="+",
                        default=["cartservice", "redis-cart"])
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    base_dir = f"reports/usad/{args.exp_name}"
    t0 = time.time()

    if args.step in ("train", "both"):
        print("\n" + "=" * 60)
        print("STEP 1: TRAINING")
        print("=" * 60)
        train_args = argparse.Namespace(
            data=args.data, output_dir=base_dir,
            pods=args.pods, window_size=args.window_size,
            latent_dim=args.latent_dim, downsample=args.downsample,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=1e-3, cpu=args.cpu,
        )
        train_usad(train_args)

    if args.step in ("detect", "both"):
        print("\n" + "=" * 60)
        print("STEP 2: DETECTION & EVALUATION")
        print("=" * 60)
        detect_args = argparse.Namespace(
            output_dir=base_dir, data=args.data,
            pods=args.pods,
            alpha_range=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            quantiles=[0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99],
            batch_size=args.batch_size, cpu=args.cpu,
        )
        run_detection(detect_args)

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {time.time()-t0:.1f}s")
    print(f"Results in: {base_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
