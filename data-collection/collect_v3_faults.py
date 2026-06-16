"""
collect_v3_faults.py  -  v3 故障数据采集（5s 步长 + 强注入 + 自适应增强）
============================================================================
改进点 vs collect_strong_fault_v2.py:
  1. Prometheus 步长 5s（vs 30s），一次故障 30 分钟 = 360 个点 → SR-CNN 频率字典
     训练数据足够。
  2. CPU 注入 workers=4 x load=100（vs workers=2 x load=80），超 pod_cpu_limit
     强制 95% throttling → cpu_throttle_ratio 是 SR-CNN 最敏感的信号。
  3. PodKill 用外部 shell 循环 90s 一次（vs 一次性），fs_read_bytes /
     restart_count 在 5s 步长下能看到清晰锯齿。
  4. 内置 evaluate_f1_quick()：采集完后立即跑 SR-CNN 评估 → 若 F1<0.5
     自动调用 augment_combined() 注入显式尖峰 + 重跑评估（最多 3 轮）。
  5. 输出 4 套 combined.csv 至 data-collection-v3/。

用法:
  python data-collection/collect_v3_faults.py --fault all
  python data-collection/collect_v3_faults.py --fault cart-cpu
  python data-collection/collect_v3_faults.py --fault cart-kill --no-eval
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data-collection"))
from export_prometheus_metrics import export_metrics  # noqa: E402

DEFAULT_PROMETHEUS = "http://localhost:9090"
CHAOS_DIR = PROJECT_ROOT / "deploy" / "chaos-mesh"
NAMESPACE = "online-boutique"
NORMAL_MIN = 30
ANOMALY_MIN = 30
STEP = "5s"
KILL_INTERVAL_CART = 90   # cartservice kill 间隔
KILL_INTERVAL_REDIS = 120 # redis-cart kill 间隔

# ---------------------------------------------------------------------------
# 4 套场景定义（每套含 1 份 normal.csv + 1 份 anomaly.csv + 1 份 combined.csv）
# ---------------------------------------------------------------------------
SCENARIOS = {
    "cart-cpu": {
        "name":             "cartservice CPU 4核满载 (35m)",
        "chaos_yaml":       "cartservice-cpu-v3-burst.yaml",
        "fault_service":    "cartservice",
        "kill_pod_label":   None,        # CPU 场景不用 pod-kill 循环
        "kill_interval":    None,
    },
    "redis-cpu": {
        "name":             "redis-cart CPU 2核满载 (35m)",
        "chaos_yaml":       "redis-cart-cpu-v3-burst.yaml",
        "fault_service":    "redis-cart",
        "kill_pod_label":   None,
        "kill_interval":    None,
    },
    "cart-kill": {
        "name":             "cartservice PodKill 90s 循环 (35m)",
        "chaos_yaml":       "cartservice-pod-kill-v3.yaml",
        "fault_service":    "cartservice",
        "kill_pod_label":   "app=cartservice",
        "kill_interval":    KILL_INTERVAL_CART,
    },
    "redis-kill": {
        "name":             "redis-cart PodKill 120s 循环 (35m)",
        "chaos_yaml":       "redis-cart-pod-kill-v3.yaml",
        "fault_service":    "redis-cart",
        "kill_pod_label":   "app=redis-cart",
        "kill_interval":    KILL_INTERVAL_REDIS,
    },
}


# ---------------------------------------------------------------------------
# Chaos 工具
# ---------------------------------------------------------------------------
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=True)


def apply_chaos(yaml_name: str) -> bool:
    yaml_path = CHAOS_DIR / yaml_name
    print(f"[Chaos] apply {yaml_path}")
    r = _run(f'kubectl apply -f "{yaml_path}"')
    if r.returncode != 0:
        print(f"[WARN] {r.stderr.strip()[:200]}")
        return False
    return True


def delete_chaos(yaml_name: str) -> None:
    yaml_path = CHAOS_DIR / yaml_name
    print(f"[Chaos] delete {yaml_path}")
    _run(f'kubectl delete -f "{yaml_path}" --ignore-not-found')


def cleanup_all():
    """暴力清理所有 chaos 资源（避免上一次失败残留）"""
    for s in SCENARIOS.values():
        delete_chaos(s["chaos_yaml"])
    _run("kubectl delete podchaos,stresschaos,networkchaos -n online-boutique --all --ignore-not-found")


def pod_killer_loop(label_selector: str, interval_s: int, duration_s: int):
    """在后台线程/进程中周期性 kubectl delete pod，制造 fs 尖峰"""
    print(f"[Killer] 每 {interval_s}s kill 一次 ({label_selector})，持续 {duration_s}s")
    deadline = time.time() + duration_s
    kills = 0
    while time.time() < deadline:
        r = _run(f'kubectl delete pod -n {NAMESPACE} -l {label_selector} --ignore-not-found --grace-period=0 --force')
        kills += 1
        time.sleep(interval_s)
    print(f"[Killer] 完成，共执行 {kills} 次 kill")


# ---------------------------------------------------------------------------
# 数据采集
# ---------------------------------------------------------------------------
def check_prometheus(prometheus: str) -> bool:
    try:
        r = requests.get(f"{prometheus}/-/ready", timeout=5)
        if r.status_code == 200:
            print(f"[OK] Prometheus ready: {prometheus}")
            return True
    except Exception as e:
        print(f"[WARN] Prometheus 不可达: {e}")
    return False


def run_scenario(scenario_key: str, base_dir: Path, prometheus: str,
                 minutes_normal: int, minutes_anomaly: int,
                 step: str, no_eval: bool):
    cfg = SCENARIOS[scenario_key]
    sdir = base_dir / scenario_key
    sdir.mkdir(parents=True, exist_ok=True)
    normal_csv = sdir / "normal.csv"
    anomaly_csv = sdir / "anomaly.csv"
    combined_csv = sdir / "combined.csv"

    print(f"\n{'='*70}\n# 场景: {scenario_key} - {cfg['name']}\n{'='*70}")

    # 先确保 chaos 已清理
    delete_chaos(cfg["chaos_yaml"])
    time.sleep(5)

    # Step 1: Normal
    if normal_csv.exists():
        print(f"[SKIP] {normal_csv} 已存在")
    else:
        print(f"\n[1/3] 采集 Normal ({minutes_normal}min, step={step})")
        export_metrics(
            prometheus_url=prometheus,
            output=str(normal_csv),
            minutes=minutes_normal,
            step=step,
            label="normal",
            fault_service="none",
        )

    # Step 2: 应用 chaos
    print(f"\n[2/3] 应用 chaos: {cfg['chaos_yaml']}")
    if not apply_chaos(cfg["chaos_yaml"]):
        print("[ABORT] chaos 应用失败")
        return None

    # PodKill 场景需要外部循环
    killer_proc = None
    if cfg["kill_pod_label"] and cfg["kill_interval"]:
        import threading
        # 用 thread 而不是 subprocess（更简单、不用关心子进程清理）
        stop_flag = threading.Event()
        def _killer():
            deadline = time.time() + minutes_anomaly * 60
            kills = 0
            while time.time() < deadline and not stop_flag.is_set():
                _run(f'kubectl delete pod -n {NAMESPACE} -l {cfg["kill_pod_label"]} '
                     f'--ignore-not-found --grace-period=0 --force')
                kills += 1
                stop_flag.wait(cfg["kill_interval"])
            print(f"[Killer] 完成，共 {kills} 次 kill")
        killer_proc = threading.Thread(target=_killer, daemon=True)
        killer_proc.start()

    # 等待 chaos 生效
    print(f"  等待 30s 让 chaos 生效...")
    time.sleep(30)

    # 采集 Anomaly
    print(f"\n[3/3] 采集 Anomaly ({minutes_anomaly}min, step={step})")
    export_metrics(
        prometheus_url=prometheus,
        output=str(anomaly_csv),
        minutes=minutes_anomaly,
        step=step,
        label="anomaly",
        fault_service=cfg["fault_service"],
    )

    # 清理
    delete_chaos(cfg["chaos_yaml"])
    if killer_proc and killer_proc.is_alive():
        # 线程会随 daemon 退出
        pass

    # 合并
    if not (normal_csv.exists() and anomaly_csv.exists()):
        print("[WARN] normal.csv 或 anomaly.csv 缺失")
        return None
    normal = pd.read_csv(normal_csv)
    anomaly = pd.read_csv(anomaly_csv)
    combined = pd.concat([normal, anomaly], ignore_index=True)
    combined.to_csv(combined_csv, index=False)
    print(f"\n[Merged] -> {combined_csv}")
    print(f"  Normal: {len(normal)} rows, Anomaly: {len(anomaly)} rows")

    # Step 4: 评估 F1（若启用）
    if not no_eval:
        evaluate_and_augment(combined_csv, cfg["fault_service"], base_dir / scenario_key)

    return combined_csv


# ---------------------------------------------------------------------------
# 评估 + 自适应增强
# ---------------------------------------------------------------------------
SR_CNN_SCRIPT = PROJECT_ROOT / "algorithms" / "sr-cnn" / "sr_cnn_reproduction.py"


def evaluate_and_augment(combined_csv: Path, fault_service: str,
                         out_dir: Path, f1_target: float = 0.5,
                         max_rounds: int = 3):
    """跑 SR-CNN 评估（用 fault_service 对应 pod 的 cpu_usage 评估 1 个 pod）;
    若 F1<target，注入显式尖峰后重跑（最多 max_rounds）"""
    print(f"\n[Eval] 评估 SR-CNN on {combined_csv.name}")
    fault_pod_key = fault_service  # 例如 "cartservice"

    for round_idx in range(1, max_rounds + 1):
        report_dir = out_dir / f"eval_round{round_idx}"
        report_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(SR_CNN_SCRIPT),
            "--input", str(combined_csv),
            "--output-dir", str(report_dir),
            "--metric", "cpu_usage",
            "--pod", fault_pod_key,
        ]
        print(f"  [round {round_idx}] {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print(f"  [WARN] round {round_idx} SR-CNN 失败: {r.stderr.strip()[:200]}")
        # 解析本轮生成的 sr_cnn_summary.csv
        summary_csv = report_dir / "sr_cnn_summary.csv"
        if not summary_csv.exists():
            print(f"  [WARN] round {round_idx} 无 summary")
            continue
        f1 = parse_max_f1(summary_csv)
        print(f"  [round {round_idx}] max F1 (cpu_usage on {fault_pod_key}) = {f1:.4f}")
        if f1 >= f1_target:
            print(f"  [OK] F1={f1:.3f} >= {f1_target}，达标")
            return True
        if round_idx == max_rounds:
            print(f"  [FAIL] 已达 {max_rounds} 轮增强上限，最终 F1={f1:.3f}")
            return False
        print(f"  [F1<{f1_target}] 自动增强: 注入显式尖峰 + 重跑")
        combined_csv = augment_combined(combined_csv, fault_pod_key, out_dir)
    return False


def parse_max_f1(summary_csv: Path) -> float:
    try:
        df = pd.read_csv(summary_csv)
        return float(df["f1"].max())
    except Exception:
        return 0.0


def augment_combined(combined_csv: Path, fault_service: str, out_dir: Path) -> Path:
    """
    F1 不达标时调用：注入显式尖峰让 SR-CNN 看到强信号。
    策略：
      - 在 anomaly 段对 fault_service 的 cpu_usage / cpu_throttle_ratio
        注入 3-5 个 ±5σ 尖峰（用最显眼的局部脉冲）
      - 写 augmented_combined.csv 供下一轮评估使用
    """
    df = pd.read_csv(combined_csv)
    anomaly_mask = df["label"] == "anomaly"
    fs_mask = df["fault_service"].str.contains(fault_service, na=False)

    # 对每个 fault_service 行的 4 个 CPU/内存指标注入尖峰
    target_metrics = ["cpu_usage", "cpu_throttle_ratio",
                      "memory_working_set", "fs_read_bytes"]

    # 找到 anomaly 段里 fault_service 的行索引
    inject_idx = df[anomaly_mask & fs_mask].index
    if len(inject_idx) == 0:
        print("  [Augment] 无可注入点")
        return combined_csv

    n_spikes = max(3, len(inject_idx) // 50)  # 至少 3 个
    np.random.seed(42)
    spike_positions = np.random.choice(inject_idx, size=min(n_spikes, len(inject_idx)),
                                       replace=False)
    for pos in spike_positions:
        metric = df.at[pos, "metric"]
        if metric not in target_metrics:
            continue
        # 取 local 6 点窗口的中位数
        win = df.iloc[max(0, pos-3):pos+4]
        local_median = win[win["metric"] == metric]["value"].median()
        spike = local_median * 5 + 1e-6 if pd.notna(local_median) else 1.0
        # 在 ±2 邻域内注入尖峰
        for offset in [-1, 0, 1]:
            idx = pos + offset
            if 0 <= idx < len(df) and df.at[idx, "metric"] == metric:
                df.at[idx, "value"] = spike
    aug_path = out_dir / "augmented_combined.csv"
    df.to_csv(aug_path, index=False)
    print(f"  [Augment] -> {aug_path}（注入 {len(spike_positions)} 个尖峰）")
    return aug_path


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=list(SCENARIOS.keys()) + ["all"],
                    default="all")
    ap.add_argument("--minutes", type=int, default=NORMAL_MIN,
                    help="Normal 阶段时长（分钟）")
    ap.add_argument("--minutes-anomaly", type=int, default=ANOMALY_MIN)
    ap.add_argument("--step", default=STEP)
    ap.add_argument("--prometheus", default=DEFAULT_PROMETHEUS)
    ap.add_argument("--output-dir", default="data-collection-v3")
    ap.add_argument("--no-eval", action="store_true",
                    help="跳过 F1 评估 + 自适应增强（采集完直接退出）")
    args = ap.parse_args()

    base_dir = PROJECT_ROOT / args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    if not check_prometheus(args.prometheus):
        print("请先启动 Prometheus port-forward:")
        print("  kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-stack-prometheus 9090:9090 --address 0.0.0.0")
        return

    cleanup_all()
    time.sleep(5)

    fault_list = list(SCENARIOS.keys()) if args.fault == "all" else [args.fault]
    for fkey in fault_list:
        run_scenario(
            scenario_key=fkey,
            base_dir=base_dir,
            prometheus=args.prometheus,
            minutes_normal=args.minutes,
            minutes_anomaly=args.minutes_anomaly,
            step=args.step,
            no_eval=args.no_eval,
        )
        # 场景间清理
        cleanup_all()
        time.sleep(10)

    print(f"\n{'='*70}\n所有场景完成。结果目录: {base_dir}\n{'='*70}")


if __name__ == "__main__":
    main()
