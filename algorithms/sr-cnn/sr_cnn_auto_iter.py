"""
sr_cnn_auto_iter.py  -  SR-CNN 单变量自动迭代器
=========================================================================
按用户最新要求：
  1) 数据源：data-collection-strong/combined_cartservice-cpu-extreme-corrected.csv
  2) 单变量：cartservice cpu_usage（1 维时间序列）
  3) 标签规则：
       - value < 50% * max(value)  -> normal
       - value >= 50% * max(value) -> anomaly
  4) 网格搜索 (amp_window, score_window, diff_order, threshold_quantile)
  5) 每次实验独立文件夹 experiments/sr_iter_NN_<tag>/
  6) F1 >= target 停止
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DEFAULT_CSV = PROJECT_ROOT / "data-collection-strong" / "combined_cartservice-cpu-extreme-corrected.csv"
FAULT_POD = "cartservice"
FAULT_METRIC = "cpu_usage"
REL_THRESHOLD = 0.50   # 相对 max 的 50%
F1_TARGET = 0.60

# 网格 (iter1: 粗)
GRID_AMP = [3, 5, 7, 9, 11]
GRID_SCORE = [7, 11, 15, 21, 27, 31]
GRID_DIFF = [0, 1]
GRID_QUANTILE = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
GRID_SMOOTH = [0]   # iter1 不平滑


# iter2 网格：加入时间窗平滑 + 异常段预加重
GRID_AMP_2 = [5, 7, 9, 11, 13, 17, 21]
GRID_SCORE_2 = [21, 27, 31, 41, 51, 71]
GRID_DIFF_2 = [0, 1, 2]
GRID_QUANTILE_2 = [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
GRID_SMOOTH_2 = [0, 3, 5, 7]   # 滑动窗口均值平滑（去除瞬时尖峰）

# iter3 网格：在差分基础上加入"最大值预加重"和"z-score 归一化"
GRID_AMP_3 = [7, 9, 11, 13]
GRID_SCORE_3 = [21, 27, 31, 41, 51]
GRID_DIFF_3 = [0, 1]                # 移除 diff=2（破坏中间点）
GRID_QUANTILE_3 = [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
GRID_SMOOTH_3 = [0, 3, 5]
GRID_PREPROC_3 = ["none", "zscore", "amplify"]  # 三种预处理


# ---------------------------------------------------------------------------
def spectral_residual(values, amp_window, score_window, eps=1e-8):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        med = np.nanmedian(values[np.isfinite(values)])
        values = np.where(np.isfinite(values), values, med)
    n = len(values)
    if n < 2 * max(amp_window, score_window) + 1:
        return np.zeros(n)
    fft = np.fft.fft(values)
    amplitude = np.abs(fft)
    phase = np.angle(fft)
    log_amp = np.log(amplitude + eps)
    avg_log_amp = np.convolve(log_amp, np.ones(amp_window) / amp_window, mode="same")
    residual = log_amp - avg_log_amp
    saliency = np.abs(np.fft.ifft(np.exp(residual + 1j * phase)))
    avg_sal = np.convolve(saliency, np.ones(score_window) / score_window, mode="same")
    score = (saliency - avg_sal) / (avg_sal + eps)
    return np.maximum(score, 0.0)


def detect(series, amp_window, score_window, threshold_quantile, diff_order=0, smooth=0,
           preproc="none"):
    s = np.asarray(series, dtype=float)
    # 预处理
    if preproc == "zscore":
        mu = np.nanmean(s)
        sigma = np.nanstd(s) + 1e-9
        s = (s - mu) / sigma
    elif preproc == "amplify":
        # 对高于 P50 的点放大倍数（让异常段更突出）
        med = np.nanmedian(s)
        s = np.where(s > med, s * 3.0, s)
    # 平滑
    if smooth > 0:
        kernel = np.ones(smooth) / smooth
        s = np.convolve(s, kernel, mode="same")
    # 差分
    if diff_order > 0:
        s = np.diff(s, n=diff_order)
    scores = spectral_residual(s, amp_window, score_window)
    thr = np.quantile(scores, threshold_quantile)
    pred = (scores >= thr).astype(int)
    return scores, pred, float(thr)


def f1_metrics(pred, target):
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp, fp, fn


# ---------------------------------------------------------------------------
def load_and_label(csv_path: Path, pod: str, metric: str, rel_thr: float):
    """按用户规则打标签：value < 50%*max = normal，否则 anomaly"""
    df = pd.read_csv(csv_path)
    sub = df[(df["pod"].str.contains(pod, na=False)) & (df["metric"] == metric)].copy()
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f"No data for {pod}/{metric}")
    series = sub["value"].to_numpy(dtype=float)
    mx = series.max()
    threshold = mx * rel_thr
    target = (series >= threshold).astype(int)   # 1 = anomaly, 0 = normal
    return series, target, sub, threshold, mx


# ---------------------------------------------------------------------------
def run_iter(iter_idx: int, csv_path: Path, *,
             grid_amp=GRID_AMP, grid_score=GRID_SCORE, grid_diff=GRID_DIFF,
             grid_quantile=GRID_QUANTILE, grid_smooth=GRID_SMOOTH,
             grid_preproc=("none",),
             tag_suffix="") -> dict:
    tag = f"sr_iter{iter_idx:02d}_podcartservice_mcpu{tag_suffix}"
    iter_dir = EXPERIMENTS_DIR / tag
    iter_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*70}\n# {tag}\n{'='*70}")

    # 1) 加载 + 按 50%*max 打标签
    series, target, raw, thr_50, vmax = load_and_label(csv_path, FAULT_POD, FAULT_METRIC, REL_THRESHOLD)
    n_norm = int((target == 0).sum())
    n_anom = int((target == 1).sum())
    print(f"  series len: {len(series)}  vmax={vmax:.6f}  50%thr={thr_50:.6f}")
    print(f"  auto labels: normal={n_norm} ({n_norm/len(series)*100:.1f}%)  "
          f"anomaly={n_anom} ({n_anom/len(series)*100:.1f}%)")

    # 2) 网格
    rows = []
    for amp in grid_amp:
        for sw in grid_score:
            for do in grid_diff:
                for sm in grid_smooth:
                    for pp in grid_preproc:
                        scores, _, _ = detect(series, amp, sw, 0.98, do, sm, pp)
                        for q in grid_quantile:
                            thr_q = np.quantile(scores, q)
                            pred_q = (scores >= thr_q).astype(int)
                            t_use = target[do:] if do > 0 else target
                            p, r, f1, tp, fp, fn = f1_metrics(pred_q, t_use)
                            rows.append({
                                "amp_window": amp, "score_window": sw, "diff_order": do,
                                "smooth": sm, "preproc": pp,
                                "quantile": q, "threshold": float(thr_q),
                                "precision": float(p), "recall": float(r), "f1": float(f1),
                                "TP": tp, "FP": fp, "FN": fn, "pred_count": int(pred_q.sum()),
                            })
    grid_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best = grid_df.iloc[0].to_dict()
    elapsed = time.time() - t0
    print(f"  grid done in {elapsed:.2f}s, {len(grid_df)} combos")
    print(f"  BEST: amp={best['amp_window']} sw={best['score_window']} "
          f"diff={int(best['diff_order'])} q={best['quantile']:.2f}  "
          f"F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}  "
          f"TP/FP/FN={int(best['TP'])}/{int(best['FP'])}/{int(best['FN'])}")

    # 3) 保存
    grid_df.to_csv(iter_dir / "grid.csv", index=False)
    best_scores, best_pred, best_thr = detect(series,
                                               int(best["amp_window"]),
                                               int(best["score_window"]),
                                               float(best["quantile"]),
                                               int(best["diff_order"]),
                                               int(best["smooth"]),
                                               str(best["preproc"]))
    if int(best["diff_order"]) > 0:
        raw_use = raw.iloc[int(best["diff_order"]):].copy().reset_index(drop=True)
    else:
        raw_use = raw.copy()
    raw_use["sr_score"] = best_scores
    raw_use["pred"] = best_pred
    raw_use["label_rule"] = np.where(raw_use["value"] >= thr_50, "anomaly", "normal")
    raw_use.to_csv(iter_dir / "sr_cnn_results.csv", index=False)

    with open(iter_dir / "best.json", "w", encoding="utf-8") as fp:
        json.dump({
            "iter": iter_idx, "tag": tag,
            "data_csv": str(csv_path),
            "n_total": int(len(series)),
            "n_normal_auto": n_norm, "n_anomaly_auto": n_anom,
            "vmax": float(vmax), "threshold_50pct_max": float(thr_50),
            "amp_window": int(best["amp_window"]),
            "score_window": int(best["score_window"]),
            "diff_order": int(best["diff_order"]),
            "smooth": int(best["smooth"]),
            "preproc": str(best["preproc"]),
            "quantile": float(best["quantile"]),
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "TP": int(best["TP"]), "FP": int(best["FP"]), "FN": int(best["FN"]),
            "elapsed_sec": round(elapsed, 2),
        }, fp, indent=2, ensure_ascii=False)

    # 4) 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        ts = pd.to_datetime(raw_use["timestamp"], unit="s")
        axes[0].plot(ts, series[:len(raw_use)] if int(best["diff_order"]) == 0
                     else np.diff(series, n=int(best["diff_order"])),
                     label="cpu_usage", color="green", alpha=0.7)
        t_use = target[int(best["diff_order"]):] if int(best["diff_order"]) > 0 else target
        axes[0].fill_between(ts, 0, series.max() * 1.1,
                              where=t_use == 1, color="red", alpha=0.2,
                              label=f"anomaly (>=50%*max)")
        axes[0].scatter(ts[best_pred == 1], series[:len(raw_use)][best_pred == 1]
                         if int(best["diff_order"]) == 0
                         else np.diff(series, n=int(best["diff_order"]))[best_pred == 1],
                         color="orange", s=18, zorder=5, label="SR-CNN detected")
        axes[0].axhline(thr_50, color="red", linestyle=":", label=f"50%max={thr_50:.6f}")
        axes[0].set_title(f"{tag}  F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, best_scores, label="SR score", color="purple", alpha=0.7)
        axes[1].axhline(best["threshold"], color="orange", linestyle="--",
                         label=f"threshold={best['threshold']:.4f}")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        pivot = grid_df.pivot_table(index="amp_window", columns="score_window",
                                    values="f1", aggfunc="max")
        im = axes[2].imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                            origin="lower", vmin=0, vmax=1)
        axes[2].set_xticks(range(len(pivot.columns)))
        axes[2].set_xticklabels([f"{v}" for v in pivot.columns])
        axes[2].set_yticks(range(len(pivot.index)))
        axes[2].set_yticklabels([f"{v}" for v in pivot.index])
        axes[2].set_xlabel("score_window"); axes[2].set_ylabel("amp_window")
        axes[2].set_title("F1 heatmap (max over quantile, diff_order)")
        plt.colorbar(im, ax=axes[2], label="F1")
        plt.tight_layout()
        plt.savefig(iter_dir / "sr_cnn.png", dpi=130)
        plt.close()
    except Exception as e:
        print(f"  [WARN] plot fail: {e}")

    return {
        "iter": iter_idx, "tag": tag,
        "f1": float(best["f1"]), "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "amp_window": int(best["amp_window"]),
        "score_window": int(best["score_window"]),
        "diff_order": int(best["diff_order"]),
        "smooth": int(best["smooth"]),
        "preproc": str(best["preproc"]),
        "quantile": float(best["quantile"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-iter", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=1)
    ap.add_argument("--f1-target", type=float, default=F1_TARGET)
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    notes = []
    summary = []
    for i in range(args.start_iter, args.start_iter + args.max_iters):
        if i == 1:
            r = run_iter(i, csv_path, tag_suffix="")
        elif i == 2:
            r = run_iter(i, csv_path,
                         grid_amp=GRID_AMP_2, grid_score=GRID_SCORE_2,
                         grid_diff=GRID_DIFF_2, grid_quantile=GRID_QUANTILE_2,
                         grid_smooth=GRID_SMOOTH_2,
                         tag_suffix=f"_v{i}")
        else:
            r = run_iter(i, csv_path,
                         grid_amp=GRID_AMP_3, grid_score=GRID_SCORE_3,
                         grid_diff=GRID_DIFF_3, grid_quantile=GRID_QUANTILE_3,
                         grid_smooth=GRID_SMOOTH_3, grid_preproc=GRID_PREPROC_3,
                         tag_suffix=f"_v{i}")
        summary.append(r)
        note = (f"## iter {i:02d} - {r['tag']}\n"
                f"- F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}\n"
                f"- amp={r['amp_window']}, sw={r['score_window']}, "
                f"diff={r['diff_order']}, smooth={r['smooth']}, "
                f"preproc={r['preproc']}, q={r['quantile']:.2f}\n\n")
        notes.append(note)
        (EXPERIMENTS_DIR / "NOTES.md").write_text(
            f"# SR-CNN Single-Variable Auto Iter (target F1={args.f1_target})\n"
            f"## Label rule\n"
            f"- pod: cartservice\n"
            f"- metric: cpu_usage\n"
            f"- normal: value < 0.5 * max(value)\n"
            f"- anomaly: value >= 0.5 * max(value)\n\n"
            + "".join(notes),
            encoding="utf-8",
        )
        pd.DataFrame(summary).to_csv(EXPERIMENTS_DIR / "summary.csv", index=False)
        if r["f1"] >= args.f1_target:
            print(f"\n*** 达成 F1={r['f1']:.4f} >= {args.f1_target} @ iter {i} ***")
            print(f"    结果: experiments/{r['tag']}")
            return

    print(f"\n完成 {args.max_iters} 轮。最佳 F1 = {max(s['f1'] for s in summary):.4f}")


if __name__ == "__main__":
    main()
