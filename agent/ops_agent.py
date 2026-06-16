import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAOS_MAP = {
    "cart-cpu": ROOT / "deploy" / "chaos-mesh" / "cartservice-cpu-stress.yaml",
    "product-delay": ROOT / "deploy" / "chaos-mesh" / "productcatalog-network-delay.yaml",
    "checkout-kill": ROOT / "deploy" / "chaos-mesh" / "checkout-pod-kill.yaml",
}
FAULT_SERVICE = {
    "cart-cpu": "cartservice",
    "product-delay": "productcatalogservice",
    "checkout-kill": "checkoutservice",
}


def run_cmd(cmd, cwd=ROOT, check=True):
    print(f"$ {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.returncode


def apply_fault(fault):
    yaml_path = CHAOS_MAP[fault]
    run_cmd(["kubectl", "apply", "-f", str(yaml_path)])


def delete_fault(fault):
    yaml_path = CHAOS_MAP[fault]
    run_cmd(["kubectl", "delete", "-f", str(yaml_path)], check=False)


def collect(prometheus, minutes, output, label, fault_service):
    script = ROOT / "data-collection" / "export_prometheus_metrics.py"
    run_cmd([
        sys.executable,
        str(script),
        "--prometheus", prometheus,
        "--minutes", str(minutes),
        "--output", str(output),
        "--label", label,
        "--fault-service", fault_service,
    ])


def run_algorithms(input_csv, report_dir):
    sr_script = ROOT / "algorithms" / "sr-cnn" / "sr_cnn_reproduction.py"
    kpiroot_script = ROOT / "algorithms" / "kpiroot" / "kpiroot_reproduction.py"
    run_cmd([sys.executable, str(sr_script), "--input", str(input_csv), "--output-dir", str(report_dir / "sr-cnn"), "--metric", "cpu_usage"])
    run_cmd([sys.executable, str(kpiroot_script), "--input", str(input_csv), "--output-dir", str(report_dir / "kpiroot")])


def write_report(report_dir, fault, input_csv):
    report = report_dir / "diagnosis_report.md"
    content = f"""# 智能运维 Agent 诊断报告

## 故障场景

- 故障类型：{fault}
- 根因服务预期：{FAULT_SERVICE.get(fault, 'unknown')}
- 数据文件：{input_csv}

## 自动化流程

1. 调用 ChaosMesh 注入故障。
2. 等待系统产生异常指标。
3. 从 Prometheus 导出 KPI 数据。
4. 运行 KDD19 SR-CNN 异常检测。
5. 运行 ISSRE24 KPIRoot 根因 KPI 排序。
6. 生成检测图和根因排序结果。

## 输出文件

- `sr-cnn/sr_cnn_results.csv`
- `sr-cnn/sr_cnn_summary.csv`
- `kpiroot/kpiroot_ranking.csv`
- `kpiroot/kpiroot_service_ranking.csv`
- `kpiroot/kpiroot_ranking.png`

## 展示建议

答辩时重点展示 Grafana 指标波动、SR-CNN 异常点检测图、KPIRoot 服务排名结果。
"""
    report.write_text(content, encoding="utf-8")
    print(f"Report saved to {report}")


def main():
    parser = argparse.ArgumentParser(description="AIOps Agent for Online Boutique fault injection, metric collection, anomaly detection, and RCA")
    parser.add_argument("--fault", choices=sorted(CHAOS_MAP), required=True)
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--collect-minutes", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/agent-run")
    parser.add_argument("--keep-fault", action="store_true")
    args = parser.parse_args()

    report_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_csv = report_dir / "prometheus_metrics.csv"

    try:
        print("[1] Injecting fault")
        apply_fault(args.fault)
        print(f"[2] Waiting {args.warmup} seconds for metrics")
        time.sleep(args.warmup)
        print("[3] Collecting Prometheus metrics")
        collect(args.prometheus, args.collect_minutes, data_csv, "anomaly", FAULT_SERVICE[args.fault])
        print("[4] Running SR-CNN and KPIRoot")
        run_algorithms(data_csv, report_dir)
        print("[5] Writing report")
        write_report(report_dir, args.fault, data_csv)
    finally:
        if not args.keep_fault:
            print("[6] Cleaning fault")
            delete_fault(args.fault)


if __name__ == "__main__":
    main()
