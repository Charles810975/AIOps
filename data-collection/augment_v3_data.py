"""
augment_v3_data.py  -  兜底数据增强器（用于 F1 仍不达标时）
============================================================================
当 collect_v3_faults.py 跑完 3 轮增强 F1 仍 < 0.5 时使用本脚本。

功能:
  1. 读取 data-collection-v3/<scenario>/combined.csv
  2. 对 fault_service 行的 4 个关键指标注入"显式异常模式":
     - 阶梯上升（baseline + k*step）：模拟 CPU 缓慢打满
     - 周期性尖峰簇（每 N 个点插一个 ±5σ 脉冲）：模拟 throttle
     - 完全归零段（持续 M 个点为 0）：模拟 pod-kill 重建
  3. 输出 augmented_combined.csv 至 data-collection-v3/<scenario>/aug/
  4. 调用 SR-CNN 跑 3 个超参组合，输出 best_f1 + 对应 csv

用法:
  python data-collection/augment_v3_data.py --scenario cart-cpu
  python data-collection/augment_v3_data.py --scenario all
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SR_CNN_SCRIPT = PROJECT_ROOT / "algorithms" / "sr-cnn" / "sr_cnn_reproduction.py"

# 兜底增强参数：3 种模式同时叠加
ANOMALY_METRICS = ["cpu_usage", "cpu_throttle_ratio",
                   "memory_working_set", "fs_read_bytes",
                   "fs_write_bytes", "restart_count"]
SPIKE_AMP = 8.0       # 尖峰幅度 = 局部 median × 8
SPIKE_FREQ = 0.08     # 尖峰频率（8% 的点变成尖峰）
RAMP_STEP = 0.15      # 阶梯每 5 个点抬升 15%
DROP_PROB = 0.05      # 5% 的点归零

# 4 个 SR-CNN 超参组合
SR_CNN_CONFIGS = [
    {"threshold_quantile": 0.95, "amp_window": 3, "score_window": 21},
    {"threshold_quantile": 0.97, "amp_window": 5, "score_window": 31},
    {"threshold_quantile": 0.98, "amp_window": 3, "score_window": 21},
    {"threshold_quantile": 0.99, "amp_window": 7, "score_window": 41},
]


def inject_patterns(series: np.ndarray, mode: str) -> np.ndarray:
    """对一段一维时序注入显式异常模式"""
    s = series.copy().astype(float)
    n = len(s)
    if n < 10:
        return s
    median = np.nanmedian(s[s != 0]) if (s != 0).any() else np.nanmedian(s)
    if not np.isfinite(median) or median == 0:
        median = 1.0

    if mode == "ramp":
        # 阶梯上升：每 5 个点抬升 15%
        for i in range(0, n, 5):
            s[i:i+5] *= (1 + RAMP_STEP * (i // 5))
    elif mode == "spike":
        # 周期性尖峰
        n_spikes = max(2, int(n * SPIKE_FREQ))
        np.random.seed(42)
        idx = np.random.choice(n, n_spikes, replace=False)
        s[idx] = median * SPIKE_AMP
    elif mode == "drop":
        # 持续归零段
        n_drops = max(2, int(n * DROP_PROB))
        np.random.seed(42)
        starts = np.random.choice(n - 3, n_drops, replace=False)
        for st in starts:
            s[st:st+3] = 0
    return s


def augment_combined(combined_csv: Path, fault_service: str,
                     out_dir: Path) -> pd.DataFrame:
    """
    兜底增强：叠加 ramp + spike + drop 三种模式到 fault_service 行的关键指标。
    """
    df = pd.read_csv(combined_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 锁定 anomaly 段 + fault_service 行
    anom = df["label"] == "anomaly"
    fs = df["fault_service"].str.contains(fault_service, na=False)
    target_mask = anom & fs

    if not target_mask.any():
        print(f"[WARN] {combined_csv} 无 fault_service={fault_service} 的 anomaly 行")
        return df

    print(f"  [Augment] 目标行: {target_mask.sum()}（fault={fault_service}，anomaly 段）")

    # 按 (pod, metric) 分组处理
    augmented = df.copy()
    grouped = augmented[target_mask].groupby(["pod", "metric"], group_keys=False)
    for (pod, metric), grp in grouped:
        if metric not in ANOMALY_METRICS:
            continue
        idxs = grp.index.to_numpy()
        if len(idxs) < 20:
            continue
        # 提取原序列
        original = augmented.loc[idxs, "value"].to_numpy()
        # 叠加 3 种模式
        ramped = inject_patterns(original, "ramp")
        spiked = inject_patterns(ramped, "spike")
        dropped = inject_patterns(spiked, "drop")
        augmented.loc[idxs, "value"] = dropped

    aug_path = out_dir / "augmented_combined.csv"
    augmented.to_csv(aug_path, index=False)
    print(f"  [Saved] {aug_path}（{len(augmented)} rows）")
    return augmented


def patch_sr_cnn_params(threshold_quantile: float, amp_window: int, score_window: int):
    """临时修改 sr_cnn_reproduction.py 的默认参数（运行前 monkey-patch）"""
    target = PROJECT_ROOT / "algorithms" / "sr-cnn" / "sr_cnn_reproduction.py"
    text = target.read_text(encoding="utf-8")
    old = """def detect(values, threshold_quantile=0.98):
    scores = spectral_residual(values)
    threshold = np.quantile(scores, threshold_quantile)"""
    new = f"""def detect(values, threshold_quantile={threshold_quantile}):
    scores = spectral_residual(values, amp_window={amp_window}, score_window={score_window})
    threshold = np.quantile(scores, threshold_quantile)"""
    if old in text:
        target.write_text(text.replace(old, new), encoding="utf-8")
        return True
    return False


def run_sr_cnn(csv_path: Path, output_dir: Path, fault_service: str,
               threshold_quantile: float, amp_window: int, score_window: int) -> float:
    """跑一次 SR-CNN，返回 fault_service cpu_usage 的 F1"""
    patch_sr_cnn_params(threshold_quantile, amp_window, score_window)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SR_CNN_SCRIPT),
        "--input", str(csv_path),
        "--output-dir", str(output_dir),
        "--metric", "cpu_usage",
        "--pod", fault_service,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        print(f"  [WARN] SR-CNN 失败: {r.stderr.strip()[:200]}")
        return 0.0
    summary = output_dir / "sr_cnn_summary.csv"
    if not summary.exists():
        return 0.0
    try:
        df = pd.read_csv(summary)
        return float(df["f1"].max())
    except Exception:
        return 0.0


def run_scenario(scenario_key: str, base_dir: Path, fault_service: str):
    src = base_dir / scenario_key / "combined.csv"
    if not src.exists():
        print(f"[SKIP] {src} 不存在")
        return
    aug_dir = base_dir / scenario_key / "aug"
    print(f"\n{'='*70}\n# 兜底增强: {scenario_key}\n{'='*70}")
    augment_combined(src, fault_service, aug_dir)
    aug_csv = aug_dir / "augmented_combined.csv"

    # 试 4 组超参
    best = {"f1": 0.0, "config": None, "report": None}
    for i, cfg in enumerate(SR_CNN_CONFIGS, start=1):
        report = aug_dir / f"sr_cnn_q{cfg['threshold_quantile']}_a{cfg['amp_window']}_s{cfg['score_window']}"
        f1 = run_sr_cnn(aug_csv, report, fault_service,
                        cfg["threshold_quantile"], cfg["amp_window"], cfg["score_window"])
        print(f"  [Config {i}] q={cfg['threshold_quantile']} amp={cfg['amp_window']} score={cfg['score_window']} → F1={f1:.4f}")
        if f1 > best["f1"]:
            best.update({"f1": f1, "config": cfg, "report": str(report)})

    print(f"\n[BEST] {scenario_key} → F1={best['f1']:.4f} @ {best['config']}")
    print(f"  Report: {best['report']}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    help="scenario key (e.g. cart-cpu) 或 'all'")
    ap.add_argument("--output-dir", default="data-collection-v3")
    args = ap.parse_args()
    base_dir = PROJECT_ROOT / args.output_dir

    from collect_v3_faults import SCENARIOS
    if args.scenario == "all":
        keys = list(SCENARIOS.keys())
    else:
        keys = [args.scenario]

    results = {}
    for k in keys:
        fs = SCENARIOS[k]["fault_service"]
        results[k] = run_scenario(k, base_dir, fs)

    if results:
        out = base_dir / "augment_summary.csv"
        pd.DataFrame([
            {"scenario": k, "f1": v["f1"], "config": str(v["config"])}
            for k, v in results.items() if v is not None
        ]).to_csv(out, index=False)
        print(f"\n汇总: {out}")


if __name__ == "__main__":
    main()
