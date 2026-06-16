"""
离线合成多故障数据生成器
===========================
在真实服务拓扑的基础上，为 Online Boutique 模拟多故障场景，
不依赖真实集群，适合快速验证 KPIRoot 的多故障定位能力。

服务拓扑（Online Boutique）:
  frontend
    └── checkoutservice
          ├── paymentservice
          ├── emailservice
          ├── cursorservice  (依赖 cartservice)
          └── cursorservice  (依赖 inventory)
    └── cartservice  ──依赖──  redis-cart
    └── productcatalogservice
    └── recommendationservice  ──依赖──  productcatalogservice
    └── currencyservice
    └── adservice
    └── shippingservice
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).parent.parent

np.random.seed(42)

# ---------------------------------------------------------------------------
# 服务拓扑定义
# ---------------------------------------------------------------------------
SERVICES = [
    "frontend", "checkoutservice", "cartservice", "redis-cart",
    "productcatalogservice", "recommendationservice", "currencyservice",
    "emailservice", "paymentservice", "adservice", "shippingservice",
]

# 每个服务有哪些指标
METRICS_BY_SERVICE = {
    "default": ["cpu_usage", "memory_usage", "memory_working_set",
                "cpu_throttle_ratio", "network_receive_errors", "network_transmit_errors"],
    "redis-cart": ["cpu_usage", "memory_usage", "memory_rss",
                   "network_receive_errors", "network_transmit_errors"],
}

def get_metrics(service):
    return METRICS_BY_SERVICE.get(service, METRICS_BY_SERVICE["default"])

# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------
SCENARIOS = {
    "exp1_cart_redis": {
        "name": "Exp1: cartservice(CPU) + redis-cart(NetworkDelay) [双故障]",
        "faults": [
            {"service": "cartservice",      "metric": "cpu_usage",         "type": "cpu_higher",       "magnitude": 8.0, "begin_pct": 0.30, "end_pct": 0.80},
            {"service": "redis-cart",        "metric": "network_transmit_errors", "type": "spike",   "magnitude": 50.0, "begin_pct": 0.30, "end_pct": 0.80},
        ],
    },
    "exp2_checkout_productcatalog": {
        "name": "Exp2: checkoutservice(CPU) + productcatalog(NetworkDelay) [双故障]",
        "faults": [
            {"service": "checkoutservice",  "metric": "cpu_usage",         "type": "cpu_higher",       "magnitude": 5.0, "begin_pct": 0.30, "end_pct": 0.80},
            {"service": "productcatalogservice", "metric": "cpu_throttle_ratio", "type": "spike",  "magnitude": 30.0, "begin_pct": 0.30, "end_pct": 0.80},
        ],
    },
    "exp3_frontend_cart_net": {
        "name": "Exp3: frontend(NetworkDelay) + cartservice(NetworkDelay) [双故障-网络]",
        "faults": [
            {"service": "frontend",         "metric": "network_transmit_errors", "type": "spike", "magnitude": 40.0, "begin_pct": 0.25, "end_pct": 0.85},
            {"service": "cartservice",      "metric": "network_receive_errors",  "type": "spike", "magnitude": 30.0, "begin_pct": 0.35, "end_pct": 0.75},
        ],
    },
    "exp4_triple_cart_redis_checkout": {
        "name": "Exp4: cartservice(CPU) + redis-cart(Network) + checkoutservice(CPU) [三故障]",
        "faults": [
            {"service": "cartservice",      "metric": "cpu_usage",         "type": "cpu_higher",  "magnitude": 6.0, "begin_pct": 0.25, "end_pct": 0.85},
            {"service": "redis-cart",        "metric": "network_transmit_errors", "type": "spike", "magnitude": 40.0, "begin_pct": 0.30, "end_pct": 0.80},
            {"service": "checkoutservice",  "metric": "cpu_usage",         "type": "cpu_higher",  "magnitude": 4.0, "begin_pct": 0.35, "end_pct": 0.70},
        ],
    },
    "exp5_triple_frontend_productcatalog_checkout": {
        "name": "Exp5: frontend(Network) + productcatalog(CPU) + checkoutservice(Network) [三故障]",
        "faults": [
            {"service": "frontend",             "metric": "network_transmit_errors", "type": "spike", "magnitude": 50.0, "begin_pct": 0.20, "end_pct": 0.90},
            {"service": "productcatalogservice","metric": "cpu_usage",               "type": "cpu_higher", "magnitude": 5.0, "begin_pct": 0.30, "end_pct": 0.80},
            {"service": "checkoutservice",     "metric": "network_receive_errors",  "type": "spike", "magnitude": 30.0, "begin_pct": 0.40, "end_pct": 0.70},
        ],
    },
}


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------
def generate_normal_series(n_points: int, base_value: float = 1.0,
                          noise_scale: float = 0.1,
                          pod_suffix: str = "abc123") -> np.ndarray:
    """生成带周期性的正常数据序列"""
    t = np.arange(n_points)
    # 日规律 + 噪声
    trend = 0.002 * t / n_points
    daily = 0.1 * np.sin(2 * np.pi * t / n_points * 3)
    noise = np.random.normal(0, noise_scale, n_points)
    return base_value * (1 + trend + daily + noise)


def inject_fault(series: np.ndarray, n_points: int,
                 begin_pct: float, end_pct: float,
                 fault_type: str, magnitude: float) -> np.ndarray:
    """向序列中注入故障"""
    result = series.copy()
    begin = int(n_points * begin_pct)
    end   = int(n_points * end_pct)
    fault_len = end - begin

    if fault_type == "cpu_higher":
        # 逐渐上升再恢复
        ramp = np.linspace(0, magnitude, fault_len)
        result[begin:end] += ramp
    elif fault_type == "spike":
        # 尖峰 + 衰减
        spike = np.zeros(fault_len)
        peak_idx = fault_len // 3
        spike[:peak_idx] = np.linspace(0, magnitude, peak_idx)
        spike[peak_idx:] = np.linspace(magnitude, 0, fault_len - peak_idx)
        result[begin:end] += spike
    elif fault_type == "memory_leak":
        # 线性增长
        result[begin:end] += np.linspace(0, magnitude, fault_len)

    return result


def build_pod_name(service: str) -> str:
    """生成真实的 Pod 名称"""
    suffixes = {
        "cartservice": "77f8cfdff-4gflb",
        "frontend": "fc6b4bb46-wwx7m",
        "checkoutservice": "5fcbc94d48-c8mxz",
        "redis-cart": "6f887989c8-s4r67",
        "productcatalogservice": "59867cb94d-gjfzt",
        "recommendationservice": "9c8dd7cd6-f6x7c",
        "currencyservice": "7c5c4ccd64-8zv72",
        "emailservice": "5c4564f4db-c9pjv",
        "paymentservice": "858d7c66fd-pwd9m",
        "adservice": "7c8b9f944-b7rfw",
        "shippingservice": "3f4d5e6f78-hj2kx",
    }
    return f"{service}-{suffixes.get(service, 'pod-001')}"


def generate_scenario(scenario_key: str, scenario: dict,
                      n_normal: int = 360, n_anomaly: int = 360,
                      timestamps: list = None) -> pd.DataFrame:
    """
    n_normal:   Normal 阶段数据点数
    n_anomaly:  Anomaly 阶段数据点数
    timestamps: 共享时间戳序列（None = 生成新序列）
    """
    if timestamps is None:
        timestamps = list(range(n_normal + n_anomaly))

    rows = []
    faults = scenario["faults"]
    fault_services = {f["service"] for f in faults}

    # 预生成每个服务的基准值（使不同服务数值合理）
    base_values = {svc: np.random.uniform(0.5, 3.0) for svc in SERVICES}

    for service in SERVICES:
        pod = build_pod_name(service)
        metrics = get_metrics(service)

        for metric in metrics:
            # 正常序列
            base_v = base_values[service]
            normal_vals = generate_normal_series(n_normal, base_value=base_v,
                                                  noise_scale=0.08)
            anomaly_vals = generate_normal_series(n_anomaly, base_value=base_v,
                                                  noise_scale=0.08)

            # 应用该服务的故障（如果存在）
            for fault in faults:
                if fault["service"] == service:
                    anomaly_vals = inject_fault(
                        anomaly_vals, n_anomaly,
                        fault["begin_pct"], fault["end_pct"],
                        fault["type"], fault["magnitude"],
                    )

            # 如果是故障服务的下游传播（简化模拟）：
            # downstream services 的某些指标会受到上游故障影响
            if service in fault_services:
                pass  # 已注入
            elif service == "recommendationservice":
                # 受 productcatalogservice 故障传播影响
                for f in faults:
                    if f["service"] == "productcatalogservice":
                        anomaly_vals = inject_fault(
                            anomaly_vals, n_anomaly,
                            f["begin_pct"], f["end_pct"],
                            "spike", f["magnitude"] * 0.3,
                        )
            elif service == "checkoutservice":
                # 受 cartservice 故障传播影响
                for f in faults:
                    if f["service"] == "cartservice":
                        anomaly_vals = inject_fault(
                            anomaly_vals, n_anomaly,
                            f["begin_pct"], f["end_pct"],
                            "spike", f["magnitude"] * 0.4,
                        )
            elif service == "frontend":
                # 受 checkoutservice 故障传播影响
                for f in faults:
                    if f["service"] == "checkoutservice":
                        anomaly_vals = inject_fault(
                            anomaly_vals, n_anomaly,
                            f["begin_pct"], f["end_pct"],
                            "spike", f["magnitude"] * 0.2,
                        )

            # Normal 阶段
            for i, v in enumerate(normal_vals):
                rows.append({
                    "timestamp": timestamps[i],
                    "pod":        pod,
                    "metric":     metric,
                    "value":      float(v),
                    "label":      "normal",
                    "fault_service": "none",
                })

            # Anomaly 阶段
            for i, v in enumerate(anomaly_vals):
                fault_labels = "+".join(f["service"] for f in faults)
                rows.append({
                    "timestamp": timestamps[n_normal + i],
                    "pod":        pod,
                    "metric":     metric,
                    "value":      float(v),
                    "label":      "anomaly",
                    "fault_service": fault_labels,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="离线合成多故障数据生成")
    parser.add_argument("--output-dir", default="data-collection-synthetic",
                        help="输出根目录")
    parser.add_argument("--scenarios", nargs="+",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--n-normal",   type=int, default=360,
                        help="Normal 阶段数据点数（默认 360 = 30min@5s）")
    parser.add_argument("--n-anomaly",  type=int, default=360,
                        help="Anomaly 阶段数据点数")
    args = parser.parse_args()

    if "all" in args.scenarios:
        scenario_list = list(SCENARIOS.keys())
    else:
        scenario_list = args.scenarios

    base_dir = PROJECT_ROOT / args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"  离线合成多故障数据生成")
    print(f"  输出目录: {base_dir}")
    print(f"  Normal: {args.n_normal} pts | Anomaly: {args.n_anomaly} pts")
    print(f"  场景: {scenario_list}")
    print(f"=" * 70)

    for s_key in scenario_list:
        scenario = SCENARIOS[s_key]
        print(f"\n[{s_key}] {scenario['name']}")

        s_dir = base_dir / s_key
        s_dir.mkdir(parents=True, exist_ok=True)

        # 生成共享时间戳
        n_total = args.n_normal + args.n_anomaly
        timestamps = list(range(n_total))

        df = generate_scenario(
            scenario_key=s_key,
            scenario=scenario,
            n_normal=args.n_normal,
            n_anomaly=args.n_anomaly,
            timestamps=timestamps,
        )

        # 拆分 normal / anomaly
        df_normal  = df[df["label"] == "normal"].copy()
        df_anomaly = df[df["label"] == "anomaly"].copy()

        # 保存
        combined_path = s_dir / "combined.csv"
        df.to_csv(combined_path, index=False, encoding="utf-8")

        print(f"  生成 {len(df)} 行 | {df['pod'].nunique()} pods | "
              f"{df['metric'].nunique()} metrics")
        print(f"  故障: {[f['service'] for f in scenario['faults']]}")
        print(f"  -> {combined_path}")

    print(f"\n{'='*70}")
    print(f"  所有合成数据已生成: {base_dir}")
    print(f"  接下来运行: python -m data-collection.run_synthetic_kpiroot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
