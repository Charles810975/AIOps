"""
USAD Experiment 7: CartPod-Only Focus
只用 cartservice 自身的指标（8个），排除其他 pod 的干扰
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

def main():
    device = torch.device("cpu")
    output_dir = Path("reports/usad/experiment_7")
    output_dir.mkdir(parents=True, exist_ok=True)

    DATA = "data-collection-strong/combined_cartservice-cpu-extreme-corrected.csv"
    K = 2
    LATENT = 64
    EPOCHS = 100
    BATCH = 64

    print("=" * 60)
    print("USAD Exp 7: CartPod-Only Focus")
    print("=" * 60)

    # Filter to cartservice only
    df = pd.read_csv(DATA)
    df_cart = df[df['pod'].str.contains('cartservice')].copy()
    tmp = Path(tempfile.gettempdir()) / "usad_cart.csv"
    df_cart.to_csv(tmp, index=False)

    # Normal-only for training
    df_normal = df_cart[df_cart['label'] == 'normal']
    tmp_n = Path(tempfile.gettempdir()) / "usad_cart_normal.csv"
    df_normal.to_csv(tmp_n, index=False)

    print(f"Data: {df_cart.shape[0]} rows, {df_cart['metric'].nunique()} metrics")
    print(f"Metrics: {df_cart['metric'].unique().tolist()}")

    # Train
    dataset = USADDataset(
        csv_path=str(tmp_n), window_size=K, downsample=1,
        pods=[], normalize=True
    )
    print(f"\nTraining: {dataset.n_windows} windows, {len(dataset.feature_names)} features")
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    n_feat = len(dataset.feature_names)

    model = USAD(window_size=K, n_features=n_feat, latent_dim=LATENT).to(device)
    print(f"Model: input={K*n_feat}, latent={LATENT}")

    opt_E = torch.optim.Adam(model.E.parameters(), lr=1e-3)
    opt_D1 = torch.optim.Adam(model.D1.parameters(), lr=1e-3)
    opt_D2 = torch.optim.Adam(model.D2.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bx, _ in dataloader:
            bx = bx.to(device)
            w1 = model.forward_ae1(bx); w2 = model.forward_ae2(bx)
            L1_p1 = torch.mean((bx - w1) ** 2); L2_p1 = torch.mean((bx - w2) ** 2)
            _ = model.forward_ae1(bx); w2r = model.forward_ae2_of_ae1(bx)
            L1_p2 = torch.mean((bx - w2r) ** 2); L2_p2 = -torch.mean((bx - w2r) ** 2)
            n = epoch
            L1 = (1/n)*L1_p1 + (1-1/n)*L1_p2
            L2 = (1/n)*L2_p1 + (1-1/n)*L2_p2
            opt_E.zero_grad(); opt_D1.zero_grad(); opt_D2.zero_grad()
            L1.backward(retain_graph=True); L2.backward(retain_graph=True)
            opt_E.step(); opt_D1.step(); opt_D2.step()
        if epoch % 20 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch}/{EPOCHS}")

    torch.save(model.state_dict(), output_dir / "usad_model.pt")

    # Detect
    print(f"\n--- DETECTION ---")
    dataset_d = USADDataset(
        csv_path=str(tmp), window_size=K, downsample=1,
        pods=[], normalize=True
    )
    labels = dataset_d.get_labels()
    n_feat_d = len(dataset_d.feature_names)
    print(f"Detection: {dataset_d.n_windows} windows, {n_feat_d} features")
    print(f"Anomaly: {int(labels.sum())}/{len(labels)} ({100*labels.mean():.1f}%)")

    # Reload model (feature count must match)
    model = USAD(window_size=K, n_features=n_feat_d, latent_dim=LATENT).to(device)
    model.load_state_dict(torch.load(output_dir / "usad_model.pt", weights_only=True))
    model.eval()

    dataloader_d = DataLoader(dataset_d, batch_size=BATCH, shuffle=False)
    alpha_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    quantiles = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]

    all_scores = {}
    for alpha in alpha_range:
        beta = 1.0 - alpha
        sc = []
        with torch.no_grad():
            for bx, _ in dataloader_d:
                bx = bx.to(device)
                s, _, _ = model.anomaly_score(bx, alpha=alpha, beta=beta)
                sc.append(s.cpu().numpy())
        all_scores[alpha] = np.concatenate(sc)

    results = []
    for alpha in alpha_range:
        scores = all_scores[alpha]
        for q in quantiles:
            t = np.quantile(scores, q)
            preds = (scores >= t).astype(int)
            tp = int(np.sum((preds==1)&(labels==1)))
            fp = int(np.sum((preds==1)&(labels==0)))
            fn = int(np.sum((preds==0)&(labels==1)))
            p = tp/(tp+fp) if tp+fp else 0
            r = tp/(tp+fn) if tp+fn else 0
            f = 2*p*r/(p+r) if p+r else 0
            results.append({
                "alpha":alpha,"quantile":q,"threshold":round(t,8),
                "f1":round(f,6),"precision":round(p,6),"recall":round(r,6),
                "TP":tp,"FP":fp,"FN":fn,
            })

    df_r = pd.DataFrame(results)
    best = df_r.loc[df_r["f1"].idxmax()]

    print(f"\n{'='*60}")
    print(f"Best F1:       {best['f1']:.4f}")
    print(f"Precision:     {best['precision']:.4f}")
    print(f"Recall:        {best['recall']:.4f}")
    print(f"Alpha:         {best['alpha']}  Quantile: {best['quantile']}")
    print(f"TP/FP/FN:     {int(best['TP'])}/{int(best['FP'])}/{int(best['FN'])}")
    print(f"{'='*60}")

    df_r.to_csv(output_dir / "detection_results.csv", index=False)
    with open(output_dir / "best_detection.json", "w") as f:
        json.dump({k:float(v) if k not in ["TP","FP","FN"] else int(v) for k,v in best.items()}, f, indent=2)

    # Save config
    cfg = {"window_size":K,"n_features":n_feat_d,"latent_dim":LATENT,"epochs":EPOCHS}
    with open(output_dir / "config.json","w") as f: json.dump(cfg,f,indent=2)

    # Visualize
    ts = [dataset_d.get_timestamp(i) for i in range(len(dataset_d))]
    bs = all_scores[best["alpha"]]
    bp = (bs >= best["threshold"]).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    plot_alphas = [a for a in all_scores.keys()]
    for a in plot_alphas:
        ax.plot(ts, all_scores[a], label=f"alpha={a}", alpha=0.8)
    ax.set_ylabel("Anomaly Score")
    ax.set_title("USAD Score (Different Alpha)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts, bs, color="blue", label="Score", alpha=0.8)
    ax.axhline(best["threshold"], color="orange", linestyle="--", label=f"Thresh={best['threshold']:.4f}")
    ax.fill_between(ts, 0, bs.max()*1.1, where=[l==1 for l in labels], color="red", alpha=0.2, label="Ground Truth")
    di = np.where(bp==1)[0]
    if len(di): ax.scatter([ts[i] for i in di], bs[di], color="green", s=20, zorder=5, label=f"Detected ({len(di)})")
    ax.set_xlabel("Time"); ax.set_ylabel("Score")
    ax.set_title(f"Detection (F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    pivot = df_r.pivot(index="alpha", columns="quantile", values="f1")
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"a={v:.1f}" for v in pivot.index])
    ax.set_title("F1 Heatmap"); plt.colorbar(im, ax=ax, label="F1")
    bqi = list(pivot.columns).index(best["quantile"])
    bai = list(pivot.index).index(best["alpha"])
    ax.scatter([bqi],[bai],marker="*",s=200,color="black",zorder=10)

    plt.tight_layout()
    plt.savefig(output_dir / "detection_scores.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'detection_scores.png'}")


if __name__ == "__main__":
    main()
