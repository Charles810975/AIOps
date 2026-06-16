"""
USAD Experiment 9: Tight Baseline + Adaptive Threshold
策略：
1. 训练只用 anomaly 开始前最后 N 个窗口（最小化时序偏移）
2. 用 rolling mean 作为动态基线（Adaptive Threshold）
3. 聚焦 cartservice 自身指标
"""

import json, tempfile, time
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

def adaptive_threshold(scores, window=20, sigma=2.0):
    """自适应阈值：rolling mean + k*std"""
    scores = np.array(scores)
    n = len(scores)
    adaptive = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        mu = scores[start:i+1].mean()
        std = scores[start:i+1].std()
        adaptive[i] = mu + sigma * max(std, 1e-9)
    return adaptive

def main():
    device = torch.device("cpu")
    output_dir = Path("reports/usad/experiment_9")
    output_dir.mkdir(parents=True, exist_ok=True)

    DATA = "data-collection-strong/combined_cartservice-cpu-extreme-corrected.csv"
    K = 2
    LATENT = 32
    EPOCHS = 150
    BATCH = 64

    print("=" * 60)
    print("USAD Exp 9: Tight Baseline + Adaptive Threshold")
    print("=" * 60)

    # 只选 cartservice
    df = pd.read_csv(DATA)
    df_cart = df[df['pod'].str.contains('cartservice')].copy()
    tmp = Path(tempfile.gettempdir()) / "e9_cart.csv"
    df_cart.to_csv(tmp, index=False)

    # 正常期数据训练：只取 anomaly 开始前的最后 200 个时间点
    ANOMALY_START_TS = 1781355368  # 从之前的分析得到
    df_normal = df_cart[
        (df_cart['label'] == 'normal') &
        (df_cart['timestamp'].astype(int) < ANOMALY_START_TS)
    ]
    # 进一步：只取 anomaly 前最后 200 个时间点
    recent_normal = df_normal.drop_duplicates('timestamp').sort_values('timestamp').tail(200)
    recent_timestamps = set(recent_normal['timestamp'].unique())
    df_normal_tight = df_cart[
        (df_cart['timestamp'].isin(recent_timestamps))
    ]
    tmp_n = Path(tempfile.gettempdir()) / "e9_norm.csv"
    df_normal_tight.to_csv(tmp_n, index=False)

    print(f"Training on TIGHT baseline: {len(df_normal_tight)} rows, {df_normal_tight['timestamp'].nunique()} timestamps")

    # 训练
    dataset = USADDataset(csv_path=str(tmp_n), window_size=K, downsample=1, pods=[], normalize=True)
    print(f"Training: {dataset.n_windows} windows, {len(dataset.feature_names)} features")

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
        if epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch}/{EPOCHS}")

    torch.save(model.state_dict(), output_dir / "usad_model.pt")

    # 检测
    print(f"\n--- DETECTION ---")
    dataset_d = USADDataset(csv_path=str(tmp), window_size=K, downsample=1, pods=[], normalize=True)
    labels = dataset_d.get_labels()
    n_feat_d = len(dataset_d.feature_names)
    print(f"Detection: {dataset_d.n_windows} windows, {n_feat_d} features")
    print(f"Anomaly windows: {int(labels.sum())}/{len(labels)} ({100*labels.mean():.1f}%)")

    model_d = USAD(window_size=K, n_features=n_feat_d, latent_dim=LATENT).to(device)
    model_d.load_state_dict(torch.load(output_dir / "usad_model.pt", weights_only=True))
    model_d.eval()

    dl_d = DataLoader(dataset_d, batch_size=BATCH, shuffle=False)

    # 计算所有 alpha 的分数
    all_scores = {}
    for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
        beta = 1.0 - alpha
        sc = []
        with torch.no_grad():
            for bx, _ in dl_d:
                bx = bx.to(device)
                s, _, _ = model_d.anomaly_score(bx, alpha=alpha, beta=beta)
                sc.append(s.cpu().numpy())
        all_scores[alpha] = np.concatenate(sc)

    # === 方法1: Quantile Threshold ===
    print("\n[Method 1: Quantile Threshold]")
    results_q = []
    for alpha in all_scores:
        for q in [0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]:
            t = np.quantile(all_scores[alpha], q)
            preds = (all_scores[alpha] >= t).astype(int)
            tp = int(np.sum((preds==1)&(labels==1)))
            fp = int(np.sum((preds==1)&(labels==0)))
            fn = int(np.sum((preds==0)&(labels==1)))
            p = tp/(tp+fp) if tp+fp else 0
            r = tp/(tp+fn) if tp+fn else 0
            f = 2*p*r/(p+r) if p+r else 0
            results_q.append({
                "method":"quantile","alpha":alpha,"q":q,"t":round(t,6),
                "f1":round(f,4),"precision":round(p,4),"recall":round(r,4),
                "TP":tp,"FP":fp,"FN":fn,
            })
    df_q = pd.DataFrame(results_q)
    best_q = df_q.loc[df_q["f1"].idxmax()]
    print(f"  Best: F1={best_q['f1']:.4f} P={best_q['precision']:.4f} R={best_q['recall']:.4f} alpha={best_q['alpha']} q={best_q['q']}")

    # === 方法2: Adaptive Threshold ===
    print("\n[Method 2: Adaptive Threshold]")
    results_a = []
    best_alpha = best_q["alpha"]
    base_scores = all_scores[best_alpha]
    for window in [10, 15, 20, 30, 50]:
        for sigma in [1.0, 1.5, 2.0, 2.5, 3.0]:
            adaptive_t = adaptive_threshold(base_scores, window=window, sigma=sigma)
            preds = (base_scores >= adaptive_t).astype(int)
            tp = int(np.sum((preds==1)&(labels==1)))
            fp = int(np.sum((preds==1)&(labels==0)))
            fn = int(np.sum((preds==0)&(labels==1)))
            p = tp/(tp+fp) if tp+fp else 0
            r = tp/(tp+fn) if tp+fn else 0
            f = 2*p*r/(p+r) if p+r else 0
            results_a.append({
                "method":"adaptive","window":window,"sigma":sigma,
                "f1":round(f,4),"precision":round(p,4),"recall":round(r,4),
                "TP":tp,"FP":fp,"FN":fn,
            })
    df_a = pd.DataFrame(results_a)
    best_a = df_a.loc[df_a["f1"].idxmax()]
    print(f"  Best: F1={best_a['f1']:.4f} P={best_a['precision']:.4f} R={best_a['recall']:.4f} window={int(best_a['window'])} sigma={best_a['sigma']}")

    # === 方法3: Deviation from Local Mean ===
    print("\n[Method 3: Deviation from Local Mean]")
    results_d = []
    for lookback in [5, 10, 15, 20]:
        for mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
            local_mean = np.array([base_scores[max(0,i-lookback):i+1].mean() for i in range(len(base_scores))])
            threshold = local_mean * mult
            preds = (base_scores > threshold).astype(int)
            tp = int(np.sum((preds==1)&(labels==1)))
            fp = int(np.sum((preds==1)&(labels==0)))
            fn = int(np.sum((preds==0)&(labels==1)))
            p = tp/(tp+fp) if tp+fp else 0
            r = tp/(tp+fn) if tp+fn else 0
            f = 2*p*r/(p+r) if p+r else 0
            results_d.append({"method":"deviation","lookback":lookback,"mult":mult,
                             "f1":round(f,4),"precision":round(p,4),"recall":round(r,4),
                             "TP":tp,"FP":fp,"FN":fn})
    df_d = pd.DataFrame(results_d)
    best_d = df_d.loc[df_d["f1"].idxmax()]
    print(f"  Best: F1={best_d['f1']:.4f} P={best_d['precision']:.4f} R={best_d['recall']:.4f} lookback={int(best_d['lookback'])} mult={best_d['mult']}")

    # 选择最佳方法
    all_best = [
        (best_q["f1"], "quantile", best_q),
        (best_a["f1"], "adaptive", best_a),
        (best_d["f1"], "deviation", best_d),
    ]
    all_best.sort(reverse=True)
    winner_f1, winner_method, winner = all_best[0]
    winner = winner.to_dict()

    print(f"\n{'='*60}")
    print(f"WINNER: {winner_method}  F1={winner_f1:.4f}")
    print(f"Precision: {winner['precision']:.4f}  Recall: {winner['recall']:.4f}")
    print(f"TP/FP/FN: {int(winner['TP'])}/{int(winner['FP'])}/{int(winner['FN'])}")
    print(f"{'='*60}")

    # 可视化
    ts = [dataset_d.get_timestamp(i) for i in range(len(dataset_d))]
    bs = base_scores

    # 获取最佳预测
    if winner_method == "quantile":
        final_preds = (bs >= best_q["t"]).astype(int)
        final_threshold = best_q["t"]
    elif winner_method == "adaptive":
        adaptive_t = adaptive_threshold(bs, window=int(best_a["window"]), sigma=best_a["sigma"])
        final_preds = (bs >= adaptive_t).astype(int)
        final_threshold = None
    else:
        lookback = int(best_d["lookback"])
        mult = best_d["mult"]
        local_mean = np.array([bs[max(0,i-lookback):i+1].mean() for i in range(len(bs))])
        final_preds = (bs > local_mean * mult).astype(int)
        final_threshold = None

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    ax = axes[0]
    for a in sorted(all_scores.keys()):
        ax.plot(ts, all_scores[a], label=f"alpha={a}", alpha=0.7)
    ax.set_ylabel("Anomaly Score"); ax.set_title("USAD Score (Different Alpha)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts, bs, color="blue", label="Score", alpha=0.8)
    if final_threshold is not None:
        ax.axhline(final_threshold, color="orange", linestyle="--", label=f"Thresh={final_threshold:.4f}")
    elif winner_method == "adaptive":
        ax.plot(ts, adaptive_t, color="orange", linestyle="--", label=f"Adaptive (w={int(best_a['window'])}, s={best_a['sigma']})", alpha=0.8)
    ax.fill_between(ts, 0, bs.max()*1.1, where=[l==1 for l in labels], color="red", alpha=0.2, label="Ground Truth")
    det = np.where(final_preds==1)[0]
    if len(det): ax.scatter([ts[i] for i in det], bs[det], color="green", s=20, zorder=5, label=f"Detected ({len(det)})")
    ax.set_xlabel("Time"); ax.set_ylabel("Score")
    ax.set_title(f"Detection ({winner_method}, F1={winner_f1:.4f}, P={winner['precision']:.4f}, R={winner['recall']:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Ground truth timeline
    ax = axes[2]
    labels_binary = labels.astype(int)
    ax.fill_between(ts, 0, 1.1, where=labels_binary, color="red", alpha=0.3, label="Ground Truth")
    ax.plot(ts, final_preds, color="green", alpha=0.8, label="Prediction")
    ax.set_ylabel("Label / Prediction")
    ax.set_title("Ground Truth vs Prediction")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
    ax.legend(); ax.grid(True, alpha=0.3)

    # Method comparison bar chart
    ax = axes[3]
    methods = ["Quantile", "Adaptive", "Deviation"]
    f1s = [best_q["f1"], best_a["f1"], best_d["f1"]]
    precs = [best_q["precision"], best_a["precision"], best_d["precision"]]
    recalls = [best_q["recall"], best_a["recall"], best_d["recall"]]
    x = np.arange(len(methods))
    width = 0.25
    bars1 = ax.bar(x - width, f1s, width, label="F1", color="steelblue")
    bars2 = ax.bar(x, precs, width, label="Precision", color="green")
    bars3 = ax.bar(x + width, recalls, width, label="Recall", color="orange")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Score"); ax.set_title("Method Comparison")
    ax.legend(); ax.set_ylim(0, 1)
    for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{bar.get_height():.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "detection_scores.png", dpi=150)
    plt.close()

    # 保存结果
    summary = {
        "method": winner_method,
        "best_f1": float(winner_f1),
        "precision": float(winner["precision"]),
        "recall": float(winner["recall"]),
        "TP": int(winner["TP"]), "FP": int(winner["FP"]), "FN": int(winner["FN"]),
        "all_methods": {
            "quantile": {"f1": float(best_q["f1"]), "p": float(best_q["precision"]), "r": float(best_q["recall"])},
            "adaptive": {"f1": float(best_a["f1"]), "p": float(best_a["precision"]), "r": float(best_a["recall"])},
            "deviation": {"f1": float(best_d["f1"]), "p": float(best_d["precision"]), "r": float(best_d["recall"])},
        },
    }
    with open(output_dir / "best_detection.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.concat([df_q, df_a, df_d]).to_csv(output_dir / "all_results.csv", index=False)

    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
