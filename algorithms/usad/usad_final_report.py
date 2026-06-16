"""
生成 USAD 最终实验报告
"""

import json, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from usad_model import USAD
from usad_data import USADDataset
from torch.utils.data import DataLoader

import torch

def adaptive_threshold(scores, window=20, sigma=2.0):
    scores = np.array(scores)
    n = len(scores)
    adaptive = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        mu = scores[start:i+1].mean()
        std = scores[start:i+1].std()
        adaptive[i] = mu + sigma * max(std, 1e-9)
    return adaptive

def deviation_threshold(scores, lookback=20, mult=1.5):
    scores = np.array(scores)
    local_mean = np.array([scores[max(0,i-lookback):i+1].mean() for i in range(len(scores))])
    return local_mean * mult

def load_and_score(data_path, model_path, config_path, pods_filter=None):
    with open(config_path) as f:
        cfg = json.load(f)
    K = cfg.get("window_size", 2)
    LATENT = cfg.get("latent_dim", 32)
    N_FEAT = cfg.get("n_features", 16)

    model = USAD(window_size=K, n_features=N_FEAT, latent_dim=LATENT)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    df = pd.read_csv(data_path)
    if pods_filter:
        df = df[df['pod'].str.contains(pods_filter)].copy()

    tmp = Path(tempfile.gettempdir()) / "final_cart.csv"
    df.to_csv(tmp, index=False)

    dataset = USADDataset(csv_path=str(tmp), window_size=K, downsample=1, pods=[], normalize=True)
    labels = dataset.get_labels()
    n_feat = len(dataset.feature_names)

    # Reload model with correct feature count
    model2 = USAD(window_size=K, n_features=n_feat, latent_dim=LATENT)
    model2.load_state_dict(torch.load(model_path, weights_only=True))
    model2.eval()

    dl = DataLoader(dataset, batch_size=64, shuffle=False)
    scores = []
    with torch.no_grad():
        for bx, _ in dl:
            s, _, _ = model2.anomaly_score(bx, alpha=0.2, beta=0.8)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores), labels, dataset

def main():
    output_dir = Path("reports/usad/final_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    DATA = "data-collection-strong/combined_cartservice-cpu-extreme-corrected.csv"
    ANOMALY_START_TS = 1781355368

    # Load all experiment scores
    exps = [
        ("Exp1 (weak fault)",  "reports/usad/experiment_1", "data-collection/cart_cpu_combined.csv"),
        ("Exp3 (synthetic)",  "reports/usad/experiment_3", "data-collection-strong/synthetic_cart_cpu.csv"),
        ("Exp5 (clean labels)", "reports/usad/experiment_5", DATA),
        ("Exp7 (cart-only)",   "reports/usad/experiment_7", DATA),
        ("Exp9 (best)",        "reports/usad/experiment_9", DATA),
    ]

    # Collect results from detection_results.csv
    summary = []
    for name, exp_dir, data_path in exps:
        results_csv = Path(exp_dir) / "detection_results.csv"
        if results_csv.exists():
            df_r = pd.read_csv(results_csv)
            best = df_r.loc[df_r["f1"].idxmax()]
            summary.append({
                "Experiment": name,
                "F1": round(best["f1"], 4),
                "Precision": round(best["precision"], 4),
                "Recall": round(best["recall"], 4),
                "TP": int(best["TP"]),
                "FP": int(best["FP"]),
                "FN": int(best["FN"]),
                "Alpha": best["alpha"],
                "Quantile": best["quantile"],
            })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("F1", ascending=False)
    print("=" * 70)
    print("USAD EXPERIMENT SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    # Final best visualization
    print("Generating final visualization...")
    best_exp_dir = Path("reports/usad/experiment_9")
    model_path = best_exp_dir / "usad_model.pt"

    # Re-create data for visualization
    df = pd.read_csv(DATA)
    df_cart = df[df['pod'].str.contains('cartservice')].copy()
    tmp = Path(tempfile.gettempdir()) / "final_viz.csv"
    df_cart.to_csv(tmp, index=False)

    K = 2
    LATENT = 32

    # Train data
    df_normal_tight = df_cart[
        (df_cart['label'] == 'normal') &
        (df_cart['timestamp'].astype(int) < ANOMALY_START_TS)
    ]
    recent_ts = df_normal_tight.drop_duplicates('timestamp').sort_values('timestamp').tail(200)['timestamp'].unique()
    df_norm = df_cart[df_cart['timestamp'].isin(recent_ts)]
    tmp_n = Path(tempfile.gettempdir()) / "final_norm.csv"
    df_norm.to_csv(tmp_n, index=False)

    dataset = USADDataset(csv_path=str(tmp), window_size=K, downsample=1, pods=[], normalize=True)
    labels = dataset.get_labels()
    n_feat = len(dataset.feature_names)

    model = USAD(window_size=K, n_features=n_feat, latent_dim=LATENT)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    dl = DataLoader(dataset, batch_size=64, shuffle=False)
    scores = []
    with torch.no_grad():
        for bx, _ in dl:
            s, _, _ = model.anomaly_score(bx, alpha=0.2, beta=0.8)
            scores.append(s.cpu().numpy())
    scores = np.concatenate(scores)

    # Deviation threshold (best from Exp9)
    dev_thresh = deviation_threshold(scores, lookback=20, mult=1.5)
    preds = (scores > dev_thresh).astype(int)
    tp = int(np.sum((preds==1)&(labels==1)))
    fp = int(np.sum((preds==1)&(labels==0)))
    fn = int(np.sum((preds==0)&(labels==1)))
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f = 2*p*r/(p+r) if p+r else 0

    timestamps = [dataset.get_timestamp(i) for i in range(len(dataset))]

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 14))

    # 1. Score timeline with ground truth
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(timestamps, scores, color="steelblue", alpha=0.8, label="Anomaly Score", linewidth=1.2)
    ax1.plot(timestamps, dev_thresh, color="orange", linestyle="--", alpha=0.8, label="Deviation Threshold", linewidth=1.2)
    ax1.fill_between(timestamps, 0, scores.max()*1.1, where=[l==1 for l in labels],
                     color="red", alpha=0.15, label="Ground Truth")
    det = np.where(preds==1)[0]
    if len(det):
        ax1.scatter([timestamps[i] for i in det], scores[det],
                   color="green", s=30, zorder=5, label=f"Detected (n={len(det)})")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Anomaly Score")
    ax1.set_title(f"USAD Anomaly Detection (F1={f:.4f}, P={p:.4f}, R={r:.4f})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Ground truth vs prediction
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.fill_between(timestamps, 0, 1, where=labels.astype(int), color="red", alpha=0.3, label="Ground Truth")
    ax2.plot(timestamps, preds, color="green", alpha=0.8, linewidth=1.5, label="Prediction")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Anomaly"])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Status")
    ax2.set_title("Ground Truth vs Prediction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Score distribution
    ax3 = fig.add_subplot(3, 2, 3)
    anom_mask = labels == 1
    norm_mask = labels == 0
    ax3.hist(scores[norm_mask], bins=40, alpha=0.6, label=f"Normal (n={norm_mask.sum()})", color="blue")
    ax3.hist(scores[anom_mask], bins=20, alpha=0.6, label=f"Anomaly (n={anom_mask.sum()})", color="red")
    ax3.axvline(dev_thresh.mean(), color="orange", linestyle="--", label=f"Avg Threshold={dev_thresh.mean():.4f}")
    ax3.set_xlabel("Anomaly Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Score Distribution: Normal vs Anomaly")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Experiment comparison bar chart
    ax4 = fig.add_subplot(3, 2, 4)
    exp_names = [r["Experiment"] for _, r in summary_df.iterrows()]
    f1s = [r["F1"] for _, r in summary_df.iterrows()]
    ps = [r["Precision"] for _, r in summary_df.iterrows()]
    rs = [r["Recall"] for _, r in summary_df.iterrows()]
    x = np.arange(len(exp_names))
    width = 0.25
    ax4.bar(x - width, f1s, width, label="F1", color="steelblue")
    ax4.bar(x, ps, width, label="Precision", color="green")
    ax4.bar(x + width, rs, width, label="Recall", color="orange")
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.replace(" (", "\n(") for n in exp_names], fontsize=8)
    ax4.set_ylabel("Score")
    ax4.set_title("Experiment Comparison")
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis="y")
    for i, (fi, pi, ri) in enumerate(zip(f1s, ps, rs)):
        ax4.text(i-width, fi+0.02, f"{fi:.2f}", ha='center', fontsize=7)
        ax4.text(i, pi+0.02, f"{pi:.2f}", ha='center', fontsize=7)
        ax4.text(i+width, ri+0.02, f"{ri:.2f}", ha='center', fontsize=7)

    # 5. Zoom into anomaly window
    ax5 = fig.add_subplot(3, 2, 5)
    # Find anomaly start index
    anomaly_indices = np.where(anom_mask)[0]
    if len(anomaly_indices) > 0:
        zoom_start = max(0, anomaly_indices[0] - 30)
        zoom_end = min(len(timestamps), anomaly_indices[-1] + 20)
    else:
        zoom_start, zoom_end = max(0, len(timestamps)-100), len(timestamps)

    tz = timestamps[zoom_start:zoom_end]
    sz = scores[zoom_start:zoom_end]
    tz_thresh = dev_thresh[zoom_start:zoom_end]
    lz = labels[zoom_start:zoom_end]
    pz = preds[zoom_start:zoom_end]

    ax5.plot(tz, sz, color="steelblue", alpha=0.8, label="Score", linewidth=1.2)
    ax5.plot(tz, tz_thresh, color="orange", linestyle="--", alpha=0.8, label="Threshold")
    ax5.fill_between(tz, 0, sz.max()*1.1, where=lz.astype(int), color="red", alpha=0.2, label="Ground Truth")
    det_z = np.where(pz==1)[0]
    if len(det_z):
        ax5.scatter([tz[i] for i in det_z], sz[det_z], color="green", s=40, zorder=5, label=f"Detected")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Score")
    ax5.set_title("Zoom: Anomaly Window Detection")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Score vs threshold gap
    ax6 = fig.add_subplot(3, 2, 6)
    gap = scores - dev_thresh
    ax6.fill_between(timestamps, 0, gap, where=gap > 0, color="red", alpha=0.5, label="Anomaly Gap")
    ax6.fill_between(timestamps, 0, gap, where=gap <= 0, color="blue", alpha=0.3, label="Normal Gap")
    ax6.axhline(0, color="black", linewidth=0.5)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Score - Threshold")
    ax6.set_title("Score vs Threshold Gap (Positive = Anomaly)")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "final_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Save summary
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)
    print(f"\nFinal report saved to: {output_dir}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
