"""
USAD Training Script (KDD 2020)
Two-phase adversarial training on normal data only.
Phase 1: Autoencoder reconstruction
Phase 2: Adversarial - AE1 fools AE2
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from usad_model import USAD
from usad_data import USADDataset


def train_usad(args):
    print("=" * 60)
    print("USAD Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Load normal-only data for training (Phase 1 AE must train on normal data)
    print(f"\nLoading data from: {args.data}")
    raw_df = pd.read_csv(args.data)

    if "label" in raw_df.columns and "normal" in raw_df["label"].values:
        normal_df = raw_df[raw_df["label"] == "normal"]
        tmp_path = Path(tempfile.gettempdir()) / "usad_normal_train.csv"
        normal_df.to_csv(tmp_path, index=False)
        print(f"Training on NORMAL data only: {len(normal_df)} rows -> {tmp_path}")
        train_csv = str(tmp_path)
    else:
        train_csv = args.data
        print("Warning: no label column found, training on full data")

    dataset = USADDataset(
        csv_path=train_csv,
        window_size=args.window_size,
        downsample=args.downsample,
        pods=args.pods,
        normalize=True,
    )
    print(f"Dataset: {dataset.n_windows} windows, {len(dataset.feature_names)} features")
    print(f"Sample features: {dataset.feature_names[:5]}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True
    )

    n_features = len(dataset.feature_names)
    model = USAD(
        window_size=args.window_size,
        n_features=n_features,
        latent_dim=args.latent_dim,
    ).to(device)

    input_dim = args.window_size * n_features
    print(f"\nModel: input={input_dim}, latent={args.latent_dim}, window={args.window_size}, features={n_features}")

    optimizer_E = torch.optim.Adam(model.E.parameters(), lr=args.lr)
    optimizer_D1 = torch.optim.Adam(model.D1.parameters(), lr=args.lr)
    optimizer_D2 = torch.optim.Adam(model.D2.parameters(), lr=args.lr)

    history = {"epoch": [], "loss_ae1": [], "loss_ae2": [], "total_loss": []}

    print(f"\nTraining for {args.epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ae1 = 0.0
        total_ae2 = 0.0
        count = 0

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)

            # Phase 1: reconstruction
            w1 = model.forward_ae1(batch_x)
            w2 = model.forward_ae2(batch_x)
            L_AE1_p1 = torch.mean((batch_x - w1) ** 2)
            L_AE2_p1 = torch.mean((batch_x - w2) ** 2)

            # Phase 2: adversarial
            _ = model.forward_ae1(batch_x)
            w2_recon = model.forward_ae2_of_ae1(batch_x)
            L_AE1_p2 = torch.mean((batch_x - w2_recon) ** 2)
            L_AE2_p2 = -torch.mean((batch_x - w2_recon) ** 2)

            # Evolutionary scheme (n = epoch)
            n = epoch
            L_AE1 = (1.0 / n) * L_AE1_p1 + (1.0 - 1.0 / n) * L_AE1_p2
            L_AE2 = (1.0 / n) * L_AE2_p1 + (1.0 - 1.0 / n) * L_AE2_p2

            optimizer_E.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            L_AE1.backward(retain_graph=True)
            L_AE2.backward(retain_graph=True)
            optimizer_E.step()
            optimizer_D1.step()
            optimizer_D2.step()

            total_ae1 += L_AE1.item()
            total_ae2 += L_AE2.item()
            count += 1

        avg_ae1 = total_ae1 / count
        avg_ae2 = total_ae2 / count

        history["epoch"].append(epoch)
        history["loss_ae1"].append(avg_ae1)
        history["loss_ae2"].append(avg_ae2)
        history["total_loss"].append(avg_ae1 + abs(avg_ae2))

        if epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{args.epochs} | L_AE1={avg_ae1:.6f} | L_AE2={avg_ae2:.6f} | elapsed={elapsed:.1f}s")

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "usad_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Save config
    config = {
        "window_size": args.window_size,
        "n_features": n_features,
        "latent_dim": args.latent_dim,
        "feature_names": dataset.feature_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "downsample": args.downsample,
        "training_time_sec": total_time,
        "train_windows": dataset.n_windows,
        "input_dim": input_dim,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "training_history.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["loss_ae1"], label="L_AE1", color="blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("AE1 Loss (Phase1+Phase2)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["loss_ae2"], label="L_AE2", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("AE2 Loss (Adversarial)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["epoch"], history["total_loss"], label="Total", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Combined Training Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()
    print(f"Loss plot saved: {output_dir / 'training_loss.png'}")

    return model, dataset, history, config


def main():
    parser = argparse.ArgumentParser(description="USAD Training")
    parser.add_argument("--data", type=str, default="data-collection/cart_cpu_combined.csv")
    parser.add_argument("--output_dir", type=str, default="reports/usad/experiment_1")
    parser.add_argument("--pods", type=str, nargs="+",
                        default=["cartservice", "redis-cart"])
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train_usad(args)


if __name__ == "__main__":
    main()
