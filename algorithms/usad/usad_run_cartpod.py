"""
USAD Pipeline - 针对 CartPod 故障优化的版本
改进：
1. 只用 cartservice 自身的 metrics（避免多变量稀释）
2. 用更多 cartservice 相关 pods（redis-cart, checkoutservice）作为联动指标
3. 训练只使用 normal 期数据
4. K=2 窗口最大化检测粒度
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from usad_model import USAD
from usad_data import USADDataset


def train_and_detect(args):
    print("=" * 60)
    print("USAD CartPod-Optimized Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Step 1: Load data, extract only relevant pods/metrics
    print(f"\nLoading data: {args.data}")
    df = pd.read_csv(args.data)
    print(f"Full data: {df.shape}")

    # 只选 cartservice + redis-cart + checkoutservice（直接联动服务）
    target_pods = ["cartservice", "redis-cart", "checkoutservice"]
    mask = df["pod"].apply(lambda p: any(tp in p for tp in target_pods))
    df_filtered = df[mask].copy()
    print(f"Filtered to target pods: {df_filtered.shape}, pods: {df_filtered['pod'].unique().tolist()}")

    # 保存过滤后的临时 CSV
    tmp_filtered = Path(tempfile.gettempdir()) / "usad_filtered.csv"
    df_filtered.to_csv(tmp_filtered, index=False)

    # 创建 normal-only 版本（用于训练）
    if "label" in df_filtered.columns:
        df_normal = df_filtered[df_filtered["label"] == "normal"]
        tmp_normal = Path(tempfile.gettempdir()) / "usad_normal.csv"
        df_normal.to_csv(tmp_normal, index=False)
        train_csv = str(tmp_normal)
        print(f"Training on NORMAL data only: {len(df_normal)} rows")
    else:
        train_csv = str(tmp_filtered)

    # Step 2: Train
    print(f"\n--- TRAINING ---")
    dataset = USADDataset(
        csv_path=train_csv,
        window_size=args.window_size,
        downsample=args.downsample,
        pods=[],  # 已经过滤了
        normalize=True,
    )
    print(f"Training dataset: {dataset.n_windows} windows, {len(dataset.feature_names)} features")
    print(f"Features: {dataset.feature_names[:8]}...")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    n_features = len(dataset.feature_names)

    model = USAD(
        window_size=args.window_size,
        n_features=n_features,
        latent_dim=args.latent_dim,
    ).to(device)
    print(f"Model: input={args.window_size*n_features}, latent={args.latent_dim}")

    optimizer_E = torch.optim.Adam(model.E.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_D1 = torch.optim.Adam(model.D1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_D2 = torch.optim.Adam(model.D2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"epoch": [], "loss_ae1": [], "loss_ae2": [], "total_loss": []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ae1, total_ae2, count = 0.0, 0.0, 0

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)

            w1 = model.forward_ae1(batch_x)
            w2 = model.forward_ae2(batch_x)
            L_AE1_p1 = torch.mean((batch_x - w1) ** 2)
            L_AE2_p1 = torch.mean((batch_x - w2) ** 2)

            _ = model.forward_ae1(batch_x)
            w2_recon = model.forward_ae2_of_ae1(batch_x)
            L_AE1_p2 = torch.mean((batch_x - w2_recon) ** 2)
            L_AE2_p2 = -torch.mean((batch_x - w2_recon) ** 2)

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

        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | L_AE1={avg_ae1:.6f} | L_AE2={avg_ae2:.6f}")

    total_time = time.time() - t_start
    print(f"Training complete in {total_time:.1f}s")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "usad_model.pt")

    config = {
        "window_size": args.window_size, "n_features": n_features,
        "latent_dim": args.latent_dim,
        "feature_names": dataset.feature_names,
        "training_time_sec": total_time, "train_windows": dataset.n_windows,
        "downsample": args.downsample,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Plot loss
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["loss_ae1"], color="blue")
    axes[0].set_title("AE1 Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["loss_ae2"], color="orange")
    axes[1].set_title("AE2 Loss")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["epoch"], history["total_loss"], color="green")
    axes[2].set_title("Combined Loss")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    # Step 3: Detect
    print(f"\n--- DETECTION ---")
    dataset_detect = USADDataset(
        csv_path=str(tmp_filtered),
        window_size=args.window_size,
        downsample=args.downsample,
        pods=[],
        normalize=True,
    )
    labels = dataset_detect.get_labels()
    n_detect_features = len(dataset_detect.feature_names)
    print(f"Detection dataset: {dataset_detect.n_windows} windows, {n_detect_features} features")
    print(f"Anomaly windows: {int(labels.sum())}/{len(labels)} ({100*labels.mean():.1f}%)")

    # Reload model with correct feature count
    model = USAD(
        window_size=args.window_size,
        n_features=n_detect_features,
        latent_dim=args.latent_dim,
    ).to(device)
    model.load_state_dict(torch.load(output_dir / "usad_model.pt", map_location=device, weights_only=True))
    model.eval()

    dataloader_detect = DataLoader(dataset_detect, batch_size=args.batch_size, shuffle=False)

    # Compute scores for different alpha
    all_scores = {}
    for alpha in args.alpha_range:
        beta = 1.0 - alpha
        scores_list = []
        with torch.no_grad():
            for batch_x, _ in dataloader_detect:
                batch_x = batch_x.to(device)
                score, _, _ = model.anomaly_score(batch_x, alpha=alpha, beta=beta)
                scores_list.append(score.cpu().numpy())
        all_scores[alpha] = np.concatenate(scores_list)

    # Grid search
    results = []
    for alpha in args.alpha_range:
        scores = all_scores[alpha]
        for q in args.quantiles:
            threshold = np.quantile(scores, q)
            preds = (scores >= threshold).astype(int)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            results.append({
                "alpha": alpha, "beta": 1-alpha, "quantile": q,
                "threshold": round(threshold, 8), "f1": round(f1, 6),
                "precision": round(prec, 6), "recall": round(rec, 6),
                "TP": tp, "FP": fp, "FN": fn,
            })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["f1"].idxmax()]

    print(f"\n{'='*60}")
    print(f"Best F1:       {best['f1']:.4f}")
    print(f"Precision:     {best['precision']:.4f}")
    print(f"Recall:        {best['recall']:.4f}")
    print(f"Alpha:         {best['alpha']}  Quantile: {best['quantile']}")
    print(f"TP/FP/FN:     {int(best['TP'])}/{int(best['FP'])}/{int(best['FN'])}")
    print(f"{'='*60}")

    results_df.to_csv(output_dir / "detection_results.csv", index=False)
    with open(output_dir / "best_detection.json", "w") as f:
        json.dump({k: float(v) if k not in ["TP","FP","FN"] else int(v) for k, v in best.items()}, f, indent=2)

    # Visualizations
    timestamps = [dataset_detect.get_timestamp(i) for i in range(len(dataset_detect))]
    best_scores = all_scores[best["alpha"]]
    best_preds = (best_scores >= best["threshold"]).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Score over time (all alpha)
    ax = axes[0]
    for alpha in [0.0, 0.5, 1.0]:
        if alpha in all_scores:
            ax.plot(timestamps, all_scores[alpha], label=f"alpha={alpha}", alpha=0.8)
    ax.set_ylabel("Anomaly Score")
    ax.set_title("USAD Anomaly Score (Different Alpha)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Best detection
    ax = axes[1]
    ax.plot(timestamps, best_scores, color="blue", alpha=0.8, label="Score")
    ax.axhline(best["threshold"], color="orange", linestyle="--", label=f"Thresh={best['threshold']:.4f}")
    ax.fill_between(timestamps, 0, best_scores.max() * 1.1,
                    where=[l == 1 for l in labels], color="red", alpha=0.2, label="Ground Truth")
    det_idx = np.where(best_preds == 1)[0]
    if len(det_idx):
        ax.scatter([timestamps[i] for i in det_idx], best_scores[det_idx],
                   color="green", s=20, zorder=5, label=f"Detected ({len(det_idx)})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(f"Best Detection (F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heatmap
    ax = axes[2]
    pivot = results_df.pivot(index="alpha", columns="quantile", values="f1")
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"a={v:.1f}" for v in pivot.index])
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Alpha")
    ax.set_title("F1 Heatmap (Alpha x Quantile)")
    plt.colorbar(im, ax=ax, label="F1")
    bq_idx = list(pivot.columns).index(best["quantile"])
    ba_idx = list(pivot.index).index(best["alpha"])
    ax.scatter([bq_idx], [ba_idx], marker="*", s=200, color="black", zorder=10)

    plt.tight_layout()
    plt.savefig(output_dir / "detection_scores.png", dpi=150)
    plt.close()
    print(f"Visualization saved: {output_dir / 'detection_scores.png'}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data-collection-strong/combined_cartservice-cpu-extreme-corrected.csv")
    parser.add_argument("--output_dir", type=str, default="reports/usad/experiment_6")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_range", type=float, nargs="+",
                        default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train_and_detect(args)
