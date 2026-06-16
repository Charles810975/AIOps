"""
USAD Detection & Evaluation (KDD 2020)
Loads a trained model, computes anomaly scores on the full (normal+anomaly) test dataset,
performs grid search over alpha/beta and score quantiles to find the optimal threshold,
evaluates F1/Precision/Recall, and generates visualizations.
"""

import argparse
import json
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


def run_detection(args):
    print("=" * 60)
    print("USAD Detection & Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = output_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}. Run training first.")
        return
    with open(config_path) as f:
        config = json.load(f)

    # Load full dataset (normal + anomaly) for detection
    print(f"\nLoading data: {args.data}")
    dataset = USADDataset(
        csv_path=args.data,
        window_size=config["window_size"],
        downsample=config["downsample"],
        pods=args.pods,
        normalize=True,
    )
    labels = dataset.get_labels()
    n_features = len(dataset.feature_names)
    print(f"Dataset: {dataset.n_windows} windows, {n_features} features")
    print(f"Anomaly windows: {int(labels.sum())}/{len(labels)} ({100*labels.mean():.1f}%)")

    # Load model
    model = USAD(
        window_size=config["window_size"],
        n_features=n_features,
        latent_dim=config["latent_dim"],
    ).to(device)
    model_path = output_dir / "usad_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {model_path}")

    # Build dataloader (no shuffle for correct time order)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Compute anomaly scores for each (alpha) value
    print("\nComputing anomaly scores...")
    all_scores = {}
    for alpha in args.alpha_range:
        beta = 1.0 - alpha
        scores_list = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                score, _, _ = model.anomaly_score(batch_x, alpha=alpha, beta=beta)
                scores_list.append(score.cpu().numpy())
        all_scores[alpha] = np.concatenate(scores_list)
        print(f"  alpha={alpha:.1f}: score_mean={all_scores[alpha].mean():.6f}, "
              f"score_max={all_scores[alpha].max():.6f}")

    # Grid search: alpha x quantile
    print("\nGrid searching threshold...")
    results = []
    for alpha in args.alpha_range:
        beta = 1.0 - alpha
        scores = all_scores[alpha]
        for q in args.quantiles:
            threshold = np.quantile(scores, q)
            preds = (scores >= threshold).astype(int)

            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "alpha": alpha, "beta": beta,
                "quantile": q, "threshold": round(threshold, 8),
                "f1": round(f1, 6), "precision": round(precision, 6),
                "recall": round(recall, 6),
                "TP": tp, "FP": fp, "FN": fn,
            })

    results_df = pd.DataFrame(results)
    best_idx = results_df["f1"].idxmax()
    best = results_df.loc[best_idx]

    print(f"\n{'='*60}")
    print(f"  Best F1:       {best['f1']:.4f}")
    print(f"  Precision:     {best['precision']:.4f}")
    print(f"  Recall:        {best['recall']:.4f}")
    print(f"  Alpha:         {best['alpha']}")
    print(f"  Beta:          {best['beta']}")
    print(f"  Quantile:      {best['quantile']}")
    print(f"  Threshold:     {best['threshold']:.8f}")
    print(f"  TP/FP/FN:      {int(best['TP'])}/{int(best['FP'])}/{int(best['FN'])}")
    print(f"{'='*60}")

    # Save results
    results_path = output_dir / "detection_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

    # Save best summary
    summary = {
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_alpha": float(best["alpha"]),
        "best_beta": float(best["beta"]),
        "best_quantile": float(best["quantile"]),
        "best_threshold": float(best["threshold"]),
        "TP": int(best["TP"]), "FP": int(best["FP"]), "FN": int(best["FN"]),
        "total_windows": len(labels),
        "anomaly_windows": int(labels.sum()),
    }
    with open(output_dir / "best_detection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Visualizations ----
    timestamps = [dataset.get_timestamp(i) for i in range(len(dataset))]
    best_scores = all_scores[best["alpha"]]
    best_preds = (best_scores >= best["threshold"]).astype(int)

    # 1. Score over time with ground truth
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    for alpha in [0.0, 0.5, 0.9]:
        if alpha in all_scores:
            ax.plot(timestamps, all_scores[alpha], label=f"alpha={alpha}", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("USAD Anomaly Score (Different Sensitivity Settings)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade ground truth anomaly regions
    ax2 = axes[1]
    ax2.plot(timestamps, best_scores, label="Score", color="blue", alpha=0.8)
    ax2.axhline(best["threshold"], color="orange", linestyle="--", linewidth=1.5,
                label=f"Threshold={best['threshold']:.4f}")
    ax2.fill_between(timestamps, 0, best_scores.max() * 1.1,
                     where=[l == 1 for l in labels],
                     color="red", alpha=0.2, label="Ground Truth")
    det_idx = np.where(best_preds == 1)[0]
    if len(det_idx) > 0:
        det_ts = [timestamps[i] for i in det_idx]
        det_vals = best_scores[det_idx]
        ax2.scatter(det_ts, det_vals, color="green", s=15, zorder=5,
                    label=f"Detected ({len(det_idx)})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title(f"Best Detection (alpha={best['alpha']}, F1={best['f1']:.4f}, "
                  f"P={best['precision']:.4f}, R={best['recall']:.4f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Heatmap
    ax3 = axes[2]
    pivot = results_df.pivot(index="alpha", columns="quantile", values="f1")
    im = ax3.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower",
                    vmin=0, vmax=1)
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=8)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels([f"a={v:.1f}" for v in pivot.index])
    ax3.set_xlabel("Quantile (threshold percentile)")
    ax3.set_ylabel("Alpha (sensitivity)")
    ax3.set_title("F1 Heatmap (Alpha x Quantile)")
    plt.colorbar(im, ax=ax3, label="F1")
    # Mark best
    best_q_idx = list(pivot.columns).index(best["quantile"])
    best_a_idx = list(pivot.index).index(best["alpha"])
    ax3.scatter([best_q_idx], [best_a_idx], marker="*", s=200, color="black", zorder=10)

    plt.tight_layout()
    plt.savefig(output_dir / "detection_scores.png", dpi=150)
    plt.close()
    print(f"Visualization saved: {output_dir / 'detection_scores.png'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="USAD Detection")
    parser.add_argument("--output_dir", type=str, default="reports/usad/experiment_1")
    parser.add_argument("--data", type=str, default="data-collection/cart_cpu_combined.csv")
    parser.add_argument("--pods", type=str, nargs="+",
                        default=["cartservice", "redis-cart"])
    parser.add_argument("--alpha_range", type=float, nargs="+",
                        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_detection(args)


if __name__ == "__main__":
    main()
