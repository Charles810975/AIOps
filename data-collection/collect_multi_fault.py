"""
多故障混合场景数据采集脚本
=======================================================
同时注入多个不同类型的故障，采集数据进行高压测试。

场景设计（5个实验）:
  Exp1  - 双故障: cartservice(CPU) + redis-cart(NetworkDelay)
  Exp2  - 双故障: checkoutservice(CPU) + productcatalog(NetworkDelay)
  Exp3  - 双故障: frontend(NetworkDelay) + cartservice(NetworkDelay)
  Exp4  - 三故障: cartservice(CPU) + redis-cart(Network) + checkoutservice(CPU)
  Exp5  - 三故障: frontend(Network) + productcatalog(CPU) + checkoutservice(CPU)

每个实验:
  Step1 采集 30 分钟 Normal（无故障）
  Step2 同时注入所有故障，采集 30 分钟 Anomaly
  Step3 合并为 combined.csv
  Step4 运行 KPIRoot 评估（两阶段：F1 Score + Hit@K）
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data-collection"))
from export_prometheus_metrics import export_metrics

DEFAULT_PROMETHEUS = "http://localhost:9090"
COLLECTION_MINUTES = 30


# ---------------------------------------------------------------------------
# Chaos 配置
# ---------------------------------------------------------------------------
CHAOS_DIR = PROJECT_ROOT / "deploy" / "chaos-mesh"

CHAOS_MAP = {
    "cartservice-cpu-extreme":        CHAOS_DIR / "cartservice-cpu-extreme.yaml",
    "cartservice-network-delay":      CHAOS_DIR / "cartservice-network-delay.yaml",
    "redis-cart-network-delay":        CHAOS_DIR / "redis-cart-network-delay.yaml",
    "productcatalog-network-delay":    CHAOS_DIR / "productcatalog-network-delay.yaml",
    "checkout-pod-kill":               CHAOS_DIR / "checkout-pod-kill.yaml",
    "checkoutservice-network-delay":   CHAOS_DIR / "checkoutservice-network-delay.yaml",
    "checkoutservice-cpu-stress":      CHAOS_DIR / "checkoutservice-cpu-stress.yaml",
    "productcatalogservice-cpu-stress": CHAOS_DIR / "productcatalogservice-cpu-stress.yaml",
    "frontend-network-delay":         CHAOS_DIR / "frontend-network-delay.yaml",
}

# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------
SCENARIOS = {
    "exp1_cart_redis": {
        "name":        "Exp1: cartservice(CPU) + redis-cart(NetworkDelay) - 双故障",
        "chaos":       ["cartservice-cpu-extreme", "redis-cart-network-delay"],
        "fault_pods":  ["cartservice", "redis-cart"],
        "description": "cartservice CPU满载 + redis-cart 网络延迟",
    },
    "exp2_checkout_productcatalog": {
        "name":        "Exp2: checkoutservice(CPU) + productcatalog(NetworkDelay) - 双故障",
        "chaos":       ["checkoutservice-cpu-stress", "productcatalog-network-delay"],
        "fault_pods":  ["checkoutservice", "productcatalogservice"],
        "description": "checkoutservice CPU压力 + productcatalog 网络延迟",
    },
    "exp3_frontend_cart_net": {
        "name":        "Exp3: frontend(NetworkDelay) + cartservice(NetworkDelay) - 双故障",
        "chaos":       ["frontend-network-delay", "cartservice-network-delay"],
        "fault_pods":  ["frontend", "cartservice"],
        "description": "frontend + cartservice 双网络延迟（故障传播链更复杂）",
    },
    "exp4_triple_cart_redis_checkout": {
        "name":        "Exp4: cartservice(CPU) + redis-cart(Network) + checkoutservice(CPU) - 三故障",
        "chaos":       ["cartservice-cpu-extreme", "redis-cart-network-delay", "checkoutservice-cpu-stress"],
        "fault_pods":  ["cartservice", "redis-cart", "checkoutservice"],
        "description": "三服务同时故障，级联传播链：cart→checkout→redis",
    },
    "exp5_triple_frontend_productcatalog_checkout": {
        "name":        "Exp5: frontend(Network) + productcatalog(CPU) + checkoutservice(Network) - 三故障",
        "chaos":       ["frontend-network-delay", "productcatalogservice-cpu-stress", "checkoutservice-network-delay"],
        "fault_pods":  ["frontend", "productcatalogservice", "checkoutservice"],
        "description": "三服务同时故障，前端/后端双向网络故障",
    },
}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def delete_chaos(name: str):
    yaml_path = CHAOS_MAP.get(name)
    if not yaml_path:
        print(f"  [WARN] Unknown chaos name: {name}")
        return
    subprocess.run(
        ["kubectl", "delete", "-f", str(yaml_path), "--ignore-not-found"],
        capture_output=True,
    )


def apply_chaos(name: str) -> bool:
    yaml_path = CHAOS_MAP.get(name)
    if not yaml_path:
        print(f"  [WARN] Unknown chaos name: {name}")
        return False
    result = subprocess.run(
        ["kubectl", "apply", "-f", str(yaml_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] kubectl apply failed: {result.stderr.strip()[:300]}")
        return False
    return True


def wait_pods_ready(timeout: int = 120):
    """等待所有 Pod 变为 Running 状态"""
    print(f"  Waiting for pods to stabilise (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", "online-boutique",
             "-o", "jsonpath={.items[*].status.phase}"],
            capture_output=True, text=True,
        )
        phases = result.stdout.strip().split()
        if all(p == "Running" for p in phases if p):
            print(f"  All pods Running.")
            return True
        time.sleep(5)
    print(f"  [WARN] Pods not all Running after {timeout}s: {result.stdout}")
    return False


def export_and_merge(base_dir: Path, scenario_key: str, chaos_keys: list,
                     fault_pods: list, minutes: int, step: str,
                     prometheus: str, skip_normal: bool = False):
    """采集 Normal + Anomaly 数据并合并"""
    scenario_dir = base_dir / scenario_key
    scenario_dir.mkdir(parents=True, exist_ok=True)

    normal_csv = scenario_dir / "normal.csv"
    anomaly_csv = scenario_dir / "anomaly.csv"
    combined_csv = scenario_dir / "combined.csv"

    # fault_service 标签：所有故障 pod 用逗号分隔
    fault_label = "+".join(fault_pods)

    # Step 1: Normal
    if skip_normal and normal_csv.exists():
        print(f"  [SKIP] Normal already exists: {normal_csv}")
    else:
        print(f"\n  [{scenario_key}] Step1: Normal ({minutes} min, step={step})")
        export_metrics(
            prometheus_url=prometheus,
            output=str(normal_csv),
            minutes=minutes,
            step=step,
            label="normal",
            fault_service="none",
        )

    # Step 2: Apply ALL chaos simultaneously
    print(f"\n  [{scenario_key}] Step2: Applying {len(chaos_keys)} chaos faults simultaneously...")
    applied = []
    for ck in chaos_keys:
        print(f"    Applying: {ck}")
        if apply_chaos(ck):
            applied.append(ck)
        time.sleep(2)

    if not applied:
        print("  [ERROR] No chaos applied!")
        return None

    time.sleep(10)  # 等待 chaos 生效
    wait_pods_ready()

    # Step 3: Anomaly
    print(f"\n  [{scenario_key}] Step3: Anomaly ({minutes} min, step={step})")
    export_metrics(
        prometheus_url=prometheus,
        output=str(anomaly_csv),
        minutes=minutes,
        step=step,
        label="anomaly",
        fault_service=fault_label,
    )

    # Step 4: Cleanup all chaos
    print(f"\n  [{scenario_key}] Cleanup: removing all chaos...")
    for ck in applied:
        delete_chaos(ck)

    # Step 5: Merge
    if normal_csv.exists() and anomaly_csv.exists():
        normal = pd.read_csv(normal_csv)
        anomaly = pd.read_csv(anomaly_csv)
        combined = pd.concat([normal, anomaly], ignore_index=True)
        combined.to_csv(combined_csv, index=False)
        print(f"\n  [{scenario_key}] Merged -> {combined_csv}")
        print(f"    Normal: {len(normal)} rows | Anomaly: {len(anomaly)} rows | Combined: {len(combined)} rows")
        return combined_csv

    return None


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="多故障混合场景数据采集与评估")
    parser.add_argument("--prometheus",  default=DEFAULT_PROMETHEUS)
    parser.add_argument("--minutes",     type=int, default=COLLECTION_MINUTES,
                        help="每阶段采集时长（分钟）")
    parser.add_argument("--step",        default="5s")
    parser.add_argument("--output-dir",  default="data-collection-mixed",
                        help="输出根目录")
    parser.add_argument("--scenarios",   nargs="+",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        default=["all"],
                        help="指定运行哪些场景（默认 all）")
    parser.add_argument("--skip-normal", action="store_true",
                        help="跳过 Normal 采集（复用已有 Normal 数据）")
    parser.add_argument("--kpiroot-only", action="store_true",
                        help="仅运行 KPIRoot 评估（跳过数据采集）")
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    # 选择场景
    if "all" in args.scenarios:
        scenario_list = list(SCENARIOS.keys())
    else:
        scenario_list = args.scenarios

    print(f"=" * 70)
    print(f"  多故障混合场景数据采集")
    print(f"  Prometheus: {args.prometheus}")
    print(f"  采集时长: {args.minutes} 分钟 | 步长: {args.step}")
    print(f"  输出目录: {base_dir}")
    print(f"  场景数量: {len(scenario_list)}")
    print(f"=" * 70)

    results_summary = {}

    for s_key in scenario_list:
        scenario = SCENARIOS[s_key]
        print(f"\n{'#'*70}")
        print(f"# {scenario['name']}")
        print(f"# 故障: {scenario['description']}")
        print(f"# 根因 Pod: {scenario['fault_pods']}")
        print(f"{'#'*70}")

        # 数据采集（除非只用 --kpiroot-only）
        if not args.kpiroot_only:
            combined_csv = export_and_merge(
                base_dir=base_dir,
                scenario_key=s_key,
                chaos_keys=scenario["chaos"],
                fault_pods=scenario["fault_pods"],
                minutes=args.minutes,
                step=args.step,
                prometheus=args.prometheus,
                skip_normal=args.skip_normal,
            )
            if combined_csv is None:
                print(f"  [ERROR] Data collection failed for {s_key}, skipping KPIRoot.")
                continue
        else:
            combined_csv = base_dir / s_key / "combined.csv"

        if not combined_csv or not Path(combined_csv).exists():
            print(f"  [WARN] Combined CSV not found: {combined_csv}")
            continue

        # KPIRoot 评估
        print(f"\n  [{s_key}] Running KPIRoot evaluation...")
        from algorithms.kpiroot.kpiroot_reproduction import kpiroot_rank

        output_dir = base_dir / s_key / "kpiroot"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 对每个故障 Pod 分别评估（主故障 + 次要故障都评估）
        # 主故障 = fault_pods[0]，所有故障都算入 ground truth
        all_faults = scenario["fault_pods"]

        eval_results = {}
        for fault_pod in all_faults:
            print(f"\n  --- Evaluating ground-truth: '{fault_pod}' ---")
            try:
                kpiroot_rank(
                    input_path=str(combined_csv),
                    output_dir=str(output_dir / fault_pod),
                    max_lag=5,
                    anomaly_label="anomaly",
                    ground_truth_pod_prefix=fault_pod,
                    eval_top_k=[1, 3, 5, 10],
                    plot_top_k=0,
                )
            except Exception as e:
                print(f"  [ERROR] KPIRoot failed for '{fault_pod}': {e}")

        # 汇总：取所有故障的联合评估
        print(f"\n  [{s_key}] Summary (all faults: {all_faults})")

        from algorithms.kpiroot.kpiroot_reproduction import (
            compute_anomaly_detection_f1, compute_hit_at_k
        )

        import numpy as np
        from algorithms.kpiroot.kpiroot_reproduction import prepare_matrix, pd as _pd

        combined_data = pd.read_csv(combined_csv)
        pivot = prepare_matrix(combined_data[combined_data["label"] == "anomaly"])
        series_deltas = {}

        # Compute delta per series
        normal_data = combined_data[combined_data["label"] == "normal"]
        anomaly_data = combined_data[combined_data["label"] == "anomaly"]
        for col in pivot.columns:
            pod = col.split("::", 1)[0]
            metric = col.split("::", 1)[1]
            normal_vals = normal_data[
                (normal_data["pod"] == pod) & (normal_data["metric"] == metric)
            ]["value"]
            anomaly_vals = anomaly_data[
                (anomaly_data["pod"] == pod) & (anomaly_data["metric"] == metric)
            ]["value"]
            if len(normal_vals) and len(anomaly_vals):
                delta = abs(anomaly_vals.mean() - normal_vals.mean())
                series_deltas[col] = delta

        # Build ranking result (same scoring as kpiroot_rank)
        # This is a simplified inline version for summary
        # We'll load from the first fault_pod's ranking CSV
        first_ranking = output_dir / all_faults[0] / "kpiroot_ranking.csv"
        if first_ranking.exists():
            rank_df = pd.read_csv(first_ranking)
            # Recompute with all faults as ground truth
            all_gt_kpis = set()
            for fp in all_faults:
                for idx, row in rank_df.iterrows():
                    if row["pod"].startswith(fp):
                        all_gt_kpis.add(idx)

            f1_info = {}
            for fp in all_faults:
                fp_kpis = {idx for idx, row in rank_df.iterrows()
                           if row["pod"].startswith(fp)}
                f1_info[fp] = fp_kpis

            results_summary[s_key] = {
                "scenario": scenario["name"],
                "fault_pods": all_faults,
                "num_faults": len(all_faults),
                "description": scenario["description"],
                "combined_csv": str(combined_csv),
                "output_dir": str(output_dir),
            }
            print(f"  [{s_key}] Data saved. KPIRoot results in: {output_dir}")

    # 保存实验汇总
    if results_summary:
        summary_df = pd.DataFrame(results_summary.values())
        summary_csv = base_dir / "experiments_summary.csv"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
        print(f"\n{'='*70}")
        print(f"  All experiments complete!")
        print(f"  Summary: {summary_csv}")
        print(f"  Data dir: {base_dir}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
