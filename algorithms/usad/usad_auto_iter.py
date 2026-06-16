"""
usad_auto_iter.py  -  USAD 单变量自动迭代器（cartservice cpu_usage 1 维）
================================================================
每次迭代 = 一次完整实验，结果写到 experiments/iter_NN_<tag>/ 下。

流程:
  1) 清理所有 chaos
  2) 采集 normal 数据（5min, 5s 步长）
  3) 应用 1 号 StressChaos（cartservice CPU 4 workers 95% load, 10min）
  4) 等待 60s 让 chaos 生效
  5) 采集 anomaly 数据（10min, 5s 步长）
  6) 清理 chaos
  7) 合并 normal+anomaly → iter/combined.csv
  8) 切出单变量时间序列（cartservice 的 cpu_usage）→ iter/series.npy + series_labels.npy
  9) 训练 USAD（只训练 normal 段）
  10) 在合并序列上算 anomaly score，grid search alpha x quantile → F1
  11) 写 NOTES.md + best.json
  12) 若 F1 < target，修改超参继续下一轮 iter

F1 target: 0.7

用法:
  python algorithms/usad/usad_auto_iter.py --start-iter 1 --max-iters 20
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHAOS_FAULT_YAML = PROJECT_ROOT / "deploy" / "chaos-mesh" / "faults-manual" / "01-stress-cart-cpu.yaml"
PROMETHEUS_URL = "http://localhost:9090"
NAMESPACE = "online-boutique"
FAULT_SERVICE = "cartservice"
FAULT_METRIC = "cpu_usage"
NORMAL_MIN = 5
ANOMALY_MIN = 10
STEP = "5s"
WAIT_AFTER_APPLY = 60   # chaos 生效 + chaos-daemon 调度延迟
CLEANUP_WAIT = 5

# USAD 超参网格（每个 iter 自动选一个）
HYPERPARAMS = {
    "window_size":  [12, 24, 48],
    "latent_dim":   [16, 32, 64],
    "epochs":       [30, 50, 80],
    "batch_size":   [32, 64],
    "downsample":   [1, 2, 3],
}
ALPHA_RANGE = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
QUANTILES    = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
F1_TARGET    = 0.70


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if check and r.returncode != 0:
        log(f"[CMD-FAIL] {cmd}\nSTDOUT: {r.stdout[:300]}\nSTDERR: {r.stderr[:300]}")
    return r


def check_prometheus() -> bool:
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/ready", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def query_range(query: str, start: int, end: int, step: str):
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]["result"]


def cleanup_all_chaos():
    run(f"kubectl delete -f {CHAOS_FAULT_YAML} --ignore-not-found", check=False)
    run(f"kubectl delete podchaos,stresschaos,networkchaos,timechaos -n {NAMESPACE} --all --ignore-not-found", check=False)
    run("kubectl delete podchaos,stresschaos,networkchaos,timechaos -n chaos-mesh --all --ignore-not-found", check=False)
    time.sleep(CLEANUP_WAIT)


def apply_cpu_stress():
    log(f"  Apply chaos: {CHAOS_FAULT_YAML.name}")
    r = run(f"kubectl apply -f {CHAOS_FAULT_YAML}", check=False)
    if r.returncode != 0:
        log(f"  [WARN] apply 失败: {r.stderr.strip()[:200]}")
        return False
    return True


# ---------------------------------------------------------------------------
# 采集（单变量：cartservice cpu_usage）
# ---------------------------------------------------------------------------
def collect_univariate_csv(output_csv: Path, minutes: int, label: str):
    """直接查 Prometheus，只取 cartservice pod 的 cpu_usage，导出 long-format CSV"""
    end = int(time.time())
    start = end - minutes * 60
    query = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"cartservice.*"}}[1m]))'
    log(f"  Query {label} [{minutes}min, step={STEP}]: {query}")
    result = query_range(query, start, end, STEP)
    if not result:
        raise RuntimeError(f"No data for {label}")

    rows = []
    for series in result:
        pod = series["metric"].get("pod", "unknown")
        for ts, value in series.get("values", []):
            rows.append({
                "timestamp": int(float(ts)),
                "pod": pod,
                "metric": FAULT_METRIC,
                "value": float(value),
                "label": label,
                "fault_service": FAULT_SERVICE if label == "anomaly" else "none",
            })
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log(f"  Saved {len(df)} rows → {output_csv.name}")
    return df


# ---------------------------------------------------------------------------
# USAD（手写极简版，单变量一维序列；用 PyTorch）
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Encoder(nn.Module):
    def __init__(self, in_dim, lat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(),
            nn.Linear(in_dim // 2, lat), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, lat, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lat, out_dim // 2), nn.ReLU(),
            nn.Linear(out_dim // 2, out_dim), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


class USAD1D(nn.Module):
    """单变量窗口 USAD: 输入 (B, window), 输出重构 (B, window)"""
    def __init__(self, window_size, latent_dim=32):
        super().__init__()
        self.window_size = window_size
        self.E = _Encoder(window_size, latent_dim)
        self.D1 = _Decoder(latent_dim, window_size)
        self.D2 = _Decoder(latent_dim, window_size)

    def forward_ae1(self, x):
        return self.D1(self.E(x))

    def forward_ae2(self, x):
        return self.D2(self.E(x))

    def forward_ae2_of_ae1(self, x):
        z = self.E(x)
        h1 = self.D1(z)
        z2 = self.E(h1.detach())
        return self.D2(z2)

    def score(self, x, alpha=0.5):
        ae1 = self.forward_ae1(x)
        ae2 = self.forward_ae2_of_ae1(x)
        e1 = ((x - ae1) ** 2).mean(dim=1)
        e2 = ((x - ae2) ** 2).mean(dim=1)
        return alpha * e1 + (1 - alpha) * e2


def make_windows(series: np.ndarray, window: int, downsample: int = 1):
    """滑窗 + 下采样"""
    s = series[::max(1, downsample)]
    n = len(s)
    if n < window:
        return np.zeros((0, window), dtype=np.float32), np.zeros(0, dtype=int)
    X = np.stack([s[i:i+window] for i in range(n - window + 1)], axis=0)
    idx = np.arange(window - 1, n)
    return X.astype(np.float32), idx


def train_usad_1d(X_normal: np.ndarray, latent_dim: int, epochs: int, batch_size: int):
    """只在 normal 数据上训练两阶段对抗"""
    model = USAD1D(window_size=X_normal.shape[1], latent_dim=latent_dim).to(DEVICE)
    opt_E = torch.optim.Adam(model.E.parameters(), lr=1e-3)
    opt_D1 = torch.optim.Adam(model.D1.parameters(), lr=1e-3)
    opt_D2 = torch.optim.Adam(model.D2.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(X_normal), torch.zeros(len(X_normal)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, _ in dl:
            xb = xb.to(DEVICE)
            w1 = model.forward_ae1(xb)
            w2 = model.forward_ae2(xb)
            L_AE1_p1 = ((xb - w1) ** 2).mean()
            L_AE2_p1 = ((xb - w2) ** 2).mean()
            w2_recon = model.forward_ae2_of_ae1(xb)
            L_AE1_p2 = ((xb - w2_recon) ** 2).mean()
            L_AE2_p2 = -((xb - w2_recon) ** 2).mean()
            n_e = epoch
            L_AE1 = (1.0 / n_e) * L_AE1_p1 + (1.0 - 1.0 / n_e) * L_AE1_p2
            L_AE2 = (1.0 / n_e) * L_AE2_p1 + (1.0 - 1.0 / n_e) * L_AE2_p2
            opt_E.zero_grad(); opt_D1.zero_grad(); opt_D2.zero_grad()
            L_AE1.backward(retain_graph=True)
            L_AE2.backward(retain_graph=True)
            opt_E.step(); opt_D1.step(); opt_D2.step()
    return model


def detect_and_score(model, X_all: np.ndarray, labels_all: np.ndarray):
    """grid search alpha x quantile → F1 / P / R"""
    model.eval()
    rows = []
    with torch.no_grad():
        X_t = torch.from_numpy(X_all).to(DEVICE)
        for alpha in ALPHA_RANGE:
            scores = model.score(X_t, alpha=alpha).cpu().numpy()
            for q in QUANTILES:
                thr = np.quantile(scores, q)
                preds = (scores >= thr).astype(int)
                tp = int(((preds == 1) & (labels_all == 1)).sum())
                fp = int(((preds == 1) & (labels_all == 0)).sum())
                fn = int(((preds == 0) & (labels_all == 1)).sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                rows.append({
                    "alpha": alpha, "quantile": q, "threshold": float(thr),
                    "f1": float(f1), "precision": float(p), "recall": float(r),
                    "TP": tp, "FP": fp, "FN": fn,
                })
    df = pd.DataFrame(rows)
    best = df.loc[df["f1"].idxmax()]
    return df, best.to_dict()


# ---------------------------------------------------------------------------
# 单次 iter
# ---------------------------------------------------------------------------
def run_iter(iter_idx: int, hyperparams: dict, notes: list) -> dict:
    tag = f"iter{iter_idx:02d}_w{hyperparams['window_size']}_l{hyperparams['latent_dim']}_d{hyperparams['downsample']}_e{hyperparams['epochs']}"
    iter_dir = EXPERIMENTS_DIR / tag
    iter_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n{'='*70}\n# ITER {iter_idx}: {tag}\n{'='*70}")

    normal_csv = iter_dir / "normal.csv"
    anomaly_csv = iter_dir / "anomaly.csv"
    combined_csv = iter_dir / "combined.csv"

    # 1) 清理 chaos
    log("Step 1/8: 清理 chaos")
    cleanup_all_chaos()

    # 2) 采集 normal
    log(f"Step 2/8: 采集 normal ({NORMAL_MIN}min)")
    df_n = collect_univariate_csv(normal_csv, NORMAL_MIN, "normal")

    # 3) 应用 chaos
    log("Step 3/8: 应用 1 号 CPU chaos")
    if not apply_cpu_stress():
        log("  chaos 应用失败, abort")
        return {"iter": iter_idx, "tag": tag, "f1": 0.0, "abort": True}

    # 4) 等待 chaos 生效
    log(f"Step 4/8: 等待 {WAIT_AFTER_APPLY}s chaos 生效")
    time.sleep(WAIT_AFTER_APPLY)

    # 5) 采集 anomaly
    log(f"Step 5/8: 采集 anomaly ({ANOMALY_MIN}min)")
    df_a = collect_univariate_csv(anomaly_csv, ANOMALY_MIN, "anomaly")

    # 6) 清理 chaos
    log("Step 6/8: 清理 chaos")
    cleanup_all_chaos()

    # 7) 合并
    log("Step 7/8: 合并 csv")
    combined = pd.concat([df_n, df_a], ignore_index=True)
    combined.to_csv(combined_csv, index=False)
    log(f"  combined: {len(df_n)} normal + {len(df_a)} anomaly = {len(combined)} rows")

    # 8) 训练 + 检测
    log("Step 8/8: 训练 USAD + 检测")
    # 取 cartservice pod 的整段 cpu_usage 序列
    cart_df = combined[combined["pod"].str.contains("cartservice", na=False)].copy()
    cart_df = cart_df.sort_values("timestamp")
    series = cart_df["value"].to_numpy(dtype=np.float32)
    labels_raw = (cart_df["label"] == "anomaly").to_numpy(dtype=int)

    # 缺失值填充
    if np.isnan(series).any():
        series = pd.Series(series).ffill().bfill().to_numpy(dtype=np.float32)

    # 归一化到 [0,1]
    s_min, s_max = float(series.min()), float(series.max())
    if s_max > s_min:
        series_norm = (series - s_min) / (s_max - s_min)
    else:
        series_norm = series

    # 滑窗
    X, idx = make_windows(series_norm,
                          window=hyperparams["window_size"],
                          downsample=hyperparams["downsample"])
    y = labels_raw[idx]
    log(f"  windows: {X.shape}, anomaly ratio: {y.mean():.3f}")

    # 切分训练/测试
    # 训练只用 normal 段（前半段就是 normal 段）
    n_norm_windows = int(((~y.astype(bool)).cumsum() == 0).sum())  # 正常段长度
    # 切到第一个 anomaly 出现为止
    anom_idx = np.where(y == 1)[0]
    if len(anom_idx) == 0:
        log("  [WARN] 无 anomaly 窗口，abort")
        return {"iter": iter_idx, "tag": tag, "f1": 0.0, "abort": True}
    split = int(anom_idx[0])
    X_train = X[:split]
    log(f"  train windows: {len(X_train)}, test windows: {len(X)}")

    t0 = time.time()
    model = train_usad_1d(X_train,
                          latent_dim=hyperparams["latent_dim"],
                          epochs=hyperparams["epochs"],
                          batch_size=hyperparams["batch_size"])
    train_time = time.time() - t0

    grid_df, best = detect_and_score(model, X, y)
    f1 = best["f1"]

    # 保存产物
    grid_df.to_csv(iter_dir / "grid.csv", index=False)
    with open(iter_dir / "best.json", "w", encoding="utf-8") as fp:
        json.dump({
            "iter": iter_idx, "tag": tag,
            "hyperparams": hyperparams,
            "f1": float(f1),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "alpha": float(best["alpha"]),
            "quantile": float(best["quantile"]),
            "threshold": float(best["threshold"]),
            "TP": int(best["TP"]), "FP": int(best["FP"]), "FN": int(best["FN"]),
            "train_time_sec": train_time,
            "n_train": int(len(X_train)),
            "n_test": int(len(X)),
            "anomaly_ratio": float(y.mean()),
        }, fp, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), iter_dir / "usad_1d.pt")

    # 画图（分数曲线 + 阈值）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        model.eval()
        with torch.no_grad():
            scores_best = model.score(torch.from_numpy(X).to(DEVICE),
                                      alpha=best["alpha"]).cpu().numpy()
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ax = axes[0]
        ax.plot(scores_best, label="anomaly score", color="blue", alpha=0.7)
        ax.axhline(best["threshold"], color="orange", linestyle="--", label=f"threshold={best['threshold']:.4f}")
        ax.fill_between(range(len(y)), 0, scores_best.max() * 1.1,
                        where=y == 1, color="red", alpha=0.2, label="ground truth anomaly")
        ax.set_title(f"iter {iter_idx} F1={f1:.4f} P={best['precision']:.4f} R={best['recall']:.4f}")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xlabel("window index"); ax.set_ylabel("score")
        # 原始序列
        ax2 = axes[1]
        ax2.plot(series_norm, label="cpu_usage (normalized)", color="green", alpha=0.7)
        ax2.fill_between(range(len(labels_raw)), 0, 1,
                         where=labels_raw == 1, color="red", alpha=0.2, label="anomaly")
        ax2.set_title("cartservice cpu_usage (normalized)")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(iter_dir / "scores.png", dpi=130)
        plt.close()
    except Exception as e:
        log(f"  [WARN] 画图失败: {e}")

    # 追加 NOTES
    note = (f"## iter {iter_idx:02d} - {tag}\n"
            f"- hyperparams: {hyperparams}\n"
            f"- F1={f1:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f}\n"
            f"- alpha={best['alpha']}, quantile={best['quantile']}\n"
            f"- train={len(X_train)} test={len(X)} anomaly_ratio={y.mean():.3f}\n"
            f"- TP/FP/FN={int(best['TP'])}/{int(best['FP'])}/{int(best['FN'])}\n"
            f"- train_time={train_time:.1f}s\n\n")
    notes.append(note)
    (EXPERIMENTS_DIR / "NOTES.md").write_text(
        f"# USAD Single-Variable Auto Iter Log\nTarget F1={F1_TARGET}\n\n" + "".join(notes),
        encoding="utf-8",
    )

    log(f"  >>> ITER {iter_idx} F1={f1:.4f} (P={best['precision']:.4f}, R={best['recall']:.4f})")
    return {
        "iter": iter_idx, "tag": tag, "f1": float(f1),
        "precision": float(best["precision"]), "recall": float(best["recall"]),
        "alpha": float(best["alpha"]), "quantile": float(best["quantile"]),
    }


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-iter", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--f1-target", type=float, default=F1_TARGET)
    ap.add_argument("--no-chaos", action="store_true",
                    help="跳过 chaos 注入（用已有 combined.csv 测试）")
    args = ap.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if not check_prometheus():
        log("ERROR: Prometheus 未就绪，请先跑 kubectl port-forward")
        return

    # 预定义超参序列（按复杂度递增；覆盖小窗→大窗、小latent→大latent）
    hp_queue = [
        {"window_size": 24, "latent_dim": 32, "epochs": 50, "batch_size": 64, "downsample": 1},
        {"window_size": 12, "latent_dim": 16, "epochs": 50, "batch_size": 32, "downsample": 1},
        {"window_size": 48, "latent_dim": 32, "epochs": 80, "batch_size": 64, "downsample": 1},
        {"window_size": 24, "latent_dim": 64, "epochs": 80, "batch_size": 64, "downsample": 1},
        {"window_size": 12, "latent_dim": 32, "epochs": 30, "batch_size": 32, "downsample": 2},
        {"window_size": 24, "latent_dim": 16, "epochs": 50, "batch_size": 32, "downsample": 2},
        {"window_size": 48, "latent_dim": 64, "epochs": 100, "batch_size": 64, "downsample": 1},
        {"window_size": 24, "latent_dim": 32, "epochs": 50, "batch_size": 64, "downsample": 3},
        {"window_size": 12, "latent_dim": 64, "epochs": 50, "batch_size": 32, "downsample": 1},
        {"window_size": 36, "latent_dim": 32, "epochs": 60, "batch_size": 64, "downsample": 1},
        {"window_size": 18, "latent_dim": 24, "epochs": 40, "batch_size": 32, "downsample": 1},
        {"window_size": 60, "latent_dim": 48, "epochs": 100, "batch_size": 64, "downsample": 1},
        {"window_size": 24, "latent_dim": 32, "epochs": 50, "batch_size": 32, "downsample": 1},
        {"window_size": 24, "latent_dim": 16, "epochs": 80, "batch_size": 32, "downsample": 1},
        {"window_size": 36, "latent_dim": 48, "epochs": 80, "batch_size": 64, "downsample": 2},
    ]

    notes = []
    summary = []
    for i in range(args.start_iter, args.start_iter + args.max_iters):
        hp = hp_queue[(i - args.start_iter) % len(hp_queue)]
        if args.no_chaos and i == args.start_iter:
            # 跳过第一次 chaos 注入（让数据复用）
            log(f"  [NO-CHAOS] iter {i} 复用 iter {i-1} 的数据")
            pass
        result = run_iter(i, hp, notes)
        summary.append(result)
        # 写 summary
        pd.DataFrame(summary).to_csv(EXPERIMENTS_DIR / "summary.csv", index=False)
        if result["f1"] >= args.f1_target:
            log(f"\n*** 达成目标 F1={result['f1']:.4f} >= {args.f1_target} @ iter {i} ***")
            log(f"    结果目录: experiments/{result['tag']}")
            return


if __name__ == "__main__":
    main()
