"""
强故障数据采集脚本
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data-collection"))

from export_prometheus_metrics import export_metrics

DEFAULT_PROMETHEUS = "http://localhost:9090"
COLLECTION_MINUTES = 30


def check_prometheus(prometheus):
    import requests
    try:
        resp = requests.get(f"{prometheus}/-/ready", timeout=5)
        if resp.status_code == 200:
            print(f"[OK] Prometheus 可访问: {prometheus}")
            return True
    except Exception as e:
        print(f"[WARN] Prometheus 不可访问: {e}")
    return False


def apply_chaos(chaos_yaml):
    print(f"\n[Chaos] 应用故障: {chaos_yaml}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", str(chaos_yaml)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[WARN] kubectl apply 失败: {result.stderr.strip()[:300]}")
        return False
    print(f"[OK] 故障已应用")
    return True


def delete_chaos(chaos_yaml):
    print(f"\n[Chaos] 移除故障: {chaos_yaml}")
    subprocess.run(
        ["kubectl", "delete", "-f", str(chaos_yaml), "--ignore-not-found"],
        capture_output=True, text=True
    )
    print(f"[OK] 故障已移除")


def run_single_fault(fault_name, chaos_yaml, minutes, prometheus, base_dir):
    print("=" * 60)
    print(f"强故障数据采集: {fault_name}")
    print("=" * 60)

    normal_csv = base_dir / f"normal_{fault_name}.csv"
    anomaly_csv = base_dir / f"anomaly_{fault_name}.csv"
    combined_csv = base_dir / f"combined_{fault_name}.csv"

    # 1. 采集 Normal
    if normal_csv.exists():
        print(f"\n[SKIP] Normal 数据已存在: {normal_csv}")
    else:
        print(f"\n[Collection] 采集 Normal 期数据 ({minutes}分钟)")
        export_metrics(
            prometheus_url=prometheus,
            output=str(normal_csv),
            minutes=minutes,
            step="15s",
            label="normal",
            fault_service="none",
        )

    # 2. 应用故障
    print(f"\n[Chaos] 应用故障...")
    if not apply_chaos(chaos_yaml):
        print("[ABORT] 无法应用故障")
        return None

    # 等待30秒让故障生效
    print("  等待30秒让故障生效...")
    time.sleep(30)

    # 3. 采集 Anomaly
    print(f"\n[Collection] 采集 Anomaly 期数据 ({minutes}分钟)")
    export_metrics(
        prometheus_url=prometheus,
        output=str(anomaly_csv),
        minutes=minutes,
        step="15s",
        label="anomaly",
        fault_service=fault_name,
    )

    # 4. 移除故障
    delete_chaos(chaos_yaml)

    # 5. 合并
    if normal_csv.exists() and anomaly_csv.exists():
        import pandas as pd
        normal = pd.read_csv(normal_csv)
        anomaly = pd.read_csv(anomaly_csv)
        combined = pd.concat([normal, anomaly], ignore_index=True)
        combined.to_csv(combined_csv, index=False)
        print(f"\n[Merged] -> {combined_csv}")
        print(f"  Normal: {len(normal)} 行, Anomaly: {len(anomaly)} 行")
        return combined_csv

    return None


def main():
    parser = argparse.ArgumentParser(description="强故障数据采集")
    parser.add_argument("--fault", type=str, required=True,
                        choices=["cartservice-cpu-extreme",
                                  "redis-cart-network-delay",
                                  "cartservice-memory-leak",
                                  "cartservice-pod-kill"],
                        help="故障类型")
    parser.add_argument("--minutes", type=int, default=COLLECTION_MINUTES)
    parser.add_argument("--prometheus", type=str, default=DEFAULT_PROMETHEUS)
    parser.add_argument("--output_dir", type=str, default="data-collection-strong")
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {base_dir}")

    if not check_prometheus(args.prometheus):
        print("请先启动 Prometheus 端口转发：")
        print("  kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-stack-prometheus 9090:9090 --address 0.0.0.0")
        return

    chaos_dir = PROJECT_ROOT / "deploy" / "chaos-mesh"
    chaos_map = {
        "cartservice-cpu-extreme":    chaos_dir / "cartservice-cpu-extreme.yaml",
        "redis-cart-network-delay":   chaos_dir / "redis-cart-network-delay.yaml",
        "cartservice-memory-leak":    chaos_dir / "cartservice-memory-leak.yaml",
        "cartservice-pod-kill":      chaos_dir / "cartservice-pod-kill.yaml",
    }
    chaos_yaml = chaos_map[args.fault]

    result = run_single_fault(
        args.fault, chaos_yaml,
        args.minutes, args.prometheus, base_dir
    )
    if result is not None:
        print(f"\n数据已保存: {result}")
        print(f"\n下一步运行 USAD：")
        print(f"  py algorithms/usad/usad_run.py --data {result} --exp_name experiment_2")


if __name__ == "__main__":
    main()
