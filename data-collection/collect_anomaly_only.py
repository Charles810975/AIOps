"""
采集 anomaly 数据（chaos 已运行，直接采集）
"""
import sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data-collection"))
from export_prometheus_metrics import export_metrics

# 确认 chaos 在运行
import subprocess
result = subprocess.run(
    ["kubectl", "get", "StressChaos", "cartservice-cpu-extreme", "-n", "online-boutique", "-o", "jsonpath={.status.phase}"],
    capture_output=True, text=True
)
print(f"Chaos status: {result.stdout.strip()}")

# 采集 anomaly 数据
output = PROJECT_ROOT / "data-collection-strong" / "anomaly_cartservice-cpu-extreme.csv"
print(f"Collecting anomaly data (30 min, 5s step)...")
export_metrics(
    prometheus_url="http://localhost:9090",
    output=str(output),
    minutes=30,
    step="5s",
    label="anomaly",
    fault_service="cartservice-cpu-extreme",
)
print("Done!")

# 清理 chaos
subprocess.run(
    ["kubectl", "delete", "-f", str(PROJECT_ROOT / "deploy" / "chaos-mesh" / "cartservice-cpu-extreme.yaml"), "--ignore-not-found"],
    capture_output=True
)
print("Chaos cleaned up")
