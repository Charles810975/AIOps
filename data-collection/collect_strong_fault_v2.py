"""
强故障采集脚本 v2 - 改进版
改进：
1. chaos 故障在 anomaly 采集一开始就注入（不等待30秒）
2. Prometheus 采集步长改为 5s（更多数据点）
3. 正确记录 chaos 开始时间，用于事后标签校正
"""

import argparse, time, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data-collection"))
from export_prometheus_metrics import export_metrics

DEFAULT_PROMETHEUS = "http://localhost:9090"
COLLECTION_MINUTES = 30


def apply_chaos(chaos_yaml):
    import subprocess
    print(f"[Chaos] 应用故障: {chaos_yaml}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", str(chaos_yaml)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[WARN] kubectl apply 失败: {result.stderr.strip()[:300]}")
        return False
    print(f"[OK] 故障已应用 (timestamp={int(time.time())})")
    return True


def delete_chaos(chaos_yaml):
    import subprocess
    print(f"[Chaos] 移除故障: {chaos_yaml}")
    subprocess.run(
        ["kubectl", "delete", "-f", str(chaos_yaml), "--ignore-not-found"],
        capture_output=True, text=True
    )
    print(f"[OK] 故障已移除")


def run_collection(fault_name, chaos_yaml, minutes, prometheus, base_dir, step="5s"):
    print("=" * 60)
    print(f"强故障数据采集 v2: {fault_name}")
    print("=" * 60)

    # 确保 chaos 已清理
    delete_chaos(chaos_yaml)
    time.sleep(5)

    normal_csv = base_dir / f"normal_{fault_name}.csv"
    anomaly_csv = base_dir / f"anomaly_{fault_name}.csv"
    combined_csv = base_dir / f"combined_{fault_name}.csv"

    # Step 1: 采集 Normal（30分钟）
    if normal_csv.exists():
        print(f"\n[SKIP] Normal 数据已存在: {normal_csv}")
    else:
        print(f"\n[Normal] 采集 Normal 期数据 ({minutes}分钟, step={step})")
        export_metrics(
            prometheus_url=prometheus, output=str(normal_csv),
            minutes=minutes, step=step,
            label="normal", fault_service="none",
        )

    # Step 2: 先应用 chaos，再采集 anomaly
    print(f"\n[Chaos] 立即应用故障...")
    if not apply_chaos(chaos_yaml):
        print("[ABORT]")
        return None

    # 采集 anomaly（chaos 已在运行）
    print(f"\n[Anomaly] 采集 Anomaly 期数据 ({minutes}分钟, step={step})")
    export_metrics(
        prometheus_url=prometheus, output=str(anomaly_csv),
        minutes=minutes, step=step,
        label="anomaly", fault_service=fault_name,
    )

    # 清理 chaos
    delete_chaos(chaos_yaml)

    # Step 3: 合并
    if normal_csv.exists() and anomaly_csv.exists():
        import pandas as pd
        normal = pd.read_csv(normal_csv)
        anomaly = pd.read_csv(anomaly_csv)
        combined = pd.concat([normal, anomaly], ignore_index=True)
        combined.to_csv(combined_csv, index=False)
        print(f"\n[Merged] -> {combined_csv}")
        print(f"  Normal: {len(normal)} 行")
        print(f"  Anomaly: {len(anomaly)} 行")
        return combined_csv
    return None


def main():
    parser = argparse.ArgumentParser(description="强故障采集 v2")
    parser.add_argument("--fault", type=str, required=True,
                        choices=["cartservice-cpu-extreme",
                                  "redis-cart-network-delay",
                                  "cartservice-memory-leak",
                                  "cartservice-pod-kill",
                                  "all"])
    parser.add_argument("--minutes", type=int, default=COLLECTION_MINUTES)
    parser.add_argument("--step", type=str, default="5s")
    parser.add_argument("--prometheus", type=str, default=DEFAULT_PROMETHEUS)
    parser.add_argument("--output_dir", type=str, default="data-collection-strong")
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {base_dir}")
    print(f"采集步长: {args.step}（5s = 每5秒一个数据点）")

    chaos_dir = PROJECT_ROOT / "deploy" / "chaos-mesh"
    chaos_map = {
        "cartservice-cpu-extreme":    chaos_dir / "cartservice-cpu-extreme.yaml",
        "redis-cart-network-delay":   chaos_dir / "redis-cart-network-delay.yaml",
        "cartservice-memory-leak":    chaos_dir / "cartservice-memory-leak.yaml",
        "cartservice-pod-kill":      chaos_dir / "cartservice-pod-kill.yaml",
    }

    if args.fault == "all":
        faults = ["cartservice-cpu-extreme", "redis-cart-network-delay", "cartservice-memory-leak"]
    else:
        faults = [args.fault]

    for fault in faults:
        chaos_yaml = chaos_map[fault]
        result = run_collection(fault, chaos_yaml, args.minutes, args.prometheus, base_dir, args.step)
        if result:
            print(f"  -> {result}")


if __name__ == "__main__":
    main()
