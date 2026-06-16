import argparse
import time
from pathlib import Path

import pandas as pd
import requests


DEFAULT_QUERIES = {
    # ---- CPU ----
    "cpu_usage": [
        'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="online-boutique"}[1m]))',
        'sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"cartservice.*|checkoutservice.*|productcatalogservice.*|frontend.*|paymentservice.*|shippingservice.*|recommendationservice.*|currencyservice.*|emailservice.*|adservice.*|redis-cart.*"}[1m]))',
    ],
    # ---- CPU 节流（ChaosMesh CPU 故障的强信号）----
    "cpu_throttle_ratio": [
        'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{namespace="online-boutique"}[1m]) / rate(container_cpu_cfs_periods_total{namespace="online-boutique"}[1m]))',
        'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{pod=~"cartservice.*|checkoutservice.*|productcatalogservice.*|frontend.*|paymentservice.*|shippingservice.*|recommendationservice.*|currencyservice.*|emailservice.*|adservice.*|redis-cart.*"}[1m]) / 1)',
    ],
    # ---- Memory ----
    "memory_working_set": [
        'sum by (pod) (container_memory_working_set_bytes{namespace="online-boutique"})',
        'sum by (pod) (container_memory_working_set_bytes{pod=~"cartservice.*|checkoutservice.*|productcatalogservice.*|frontend.*|paymentservice.*|shippingservice.*|recommendationservice.*|currencyservice.*|emailservice.*|adservice.*|redis-cart.*"})',
    ],
    "memory_usage": [
        'sum by (pod) (container_memory_usage_bytes{namespace="online-boutique"})',
        'sum by (pod) (container_memory_usage_bytes{pod=~"cartservice.*|checkoutservice.*|productcatalogservice.*|frontend.*|paymentservice.*|shippingservice.*|recommendationservice.*|currencyservice.*|emailservice.*|adservice.*|redis-cart.*"})',
    ],
    "memory_rss": [
        'sum by (pod) (container_memory_rss{namespace="online-boutique"})',
        'sum by (pod) (container_memory_rss{pod=~"cartservice.*|checkoutservice.*|productcatalogservice.*|frontend.*|paymentservice.*|shippingservice.*|recommendationservice.*|currencyservice.*|emailservice.*|adservice.*|redis-cart.*"})',
    ],
    "memory_cache": [
        'sum by (pod) (container_memory_cache{namespace="online-boutique"})',
    ],
    # ---- Pod 重启（ChaosMesh 故障间接信号）----
    "restart_count": [
        'sum by (pod) (kube_pod_container_status_restarts_total{namespace="online-boutique"})',
    ],
    # ---- 线程数（Go 服务 Goroutine 泄漏的代理）----
    "thread_count": [
        'sum by (pod) (container_threads{namespace="online-boutique"})',
    ],
    # ---- 进程数 ----
    "process_count": [
        'sum by (pod) (container_processes{namespace="online-boutique"})',
    ],
    # ---- 文件系统 I/O（磁盘故障的信号）----
    "fs_read_bytes": [
        'sum by (pod) (rate(container_fs_reads_bytes_total{namespace="online-boutique"}[1m]))',
    ],
    "fs_write_bytes": [
        'sum by (pod) (rate(container_fs_writes_bytes_total{namespace="online-boutique"}[1m]))',
    ],
    # ---- OOM 事件（内存压力）----
    "oom_events": [
        'sum by (pod) (rate(container_oom_events_total{namespace="online-boutique"}[1m]))',
    ],
    # ---- Pod 资源请求/限制 ----
    "pod_cpu_request": [
        'sum by (pod) (kube_pod_container_resource_requests{namespace="online-boutique",resource="cpu"})',
    ],
    "pod_cpu_limit": [
        'sum by (pod) (kube_pod_container_resource_limits{namespace="online-boutique",resource="cpu"})',
    ],
    "pod_memory_request": [
        'sum by (pod) (kube_pod_container_resource_requests{namespace="online-boutique",resource="memory"})',
    ],
    "pod_memory_limit": [
        'sum by (pod) (kube_pod_container_resource_limits{namespace="online-boutique",resource="memory"})',
    ],
}


def query_range(prometheus_url, query, start, end, step):
    response = requests.get(
        f"{prometheus_url.rstrip('/')}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(payload)
    return payload["data"]["result"]


def export_metrics(prometheus_url, output, minutes, step, label, fault_service, metrics=None):
    end = int(time.time())
    start = end - minutes * 60
    frames = []
    target_metrics = metrics if metrics else DEFAULT_QUERIES.keys()

    for metric_name in target_metrics:
        if metric_name not in DEFAULT_QUERIES:
            print(f"[WARN] Unknown metric '{metric_name}', skipping.")
            continue
        queries = DEFAULT_QUERIES[metric_name]
        result = []
        last_error = None
        for query in queries:
            try:
                result = query_range(prometheus_url, query, start, end, step)
                if result:
                    break
            except Exception as e:
                last_error = e
        if not result:
            print(f"[WARN] No data for metric '{metric_name}' after trying {len(queries)} query variants. "
                  f"Last error: {last_error}")
            continue
        print(f"Metric '{metric_name}': {len(result)} series ({len(queries)} queries tried)")
        for series in result:
            pod = series.get("metric", {}).get("pod", "unknown")
            rows = []
            for ts, value in series.get("values", []):
                rows.append({
                    "timestamp": int(float(ts)),
                    "pod": pod,
                    "metric": metric_name,
                    "value": float(value),
                    "label": label,
                    "fault_service": fault_service,
                })
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        raise RuntimeError("No data returned from Prometheus")

    data = pd.concat(frames, ignore_index=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8")
    print(f"Exported {len(data)} rows to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--output", default="data-collection/prometheus_metrics.csv")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--step", default="15s")
    parser.add_argument("--label", choices=["normal", "anomaly"], default="normal")
    parser.add_argument("--fault-service", default="none")
    parser.add_argument("--metrics", nargs="+",
                        choices=list(DEFAULT_QUERIES.keys()),
                        help="Which metrics to collect (default: all). "
                             "Examples: --metrics cpu_usage memory_usage")
    args = parser.parse_args()
    export_metrics(args.prometheus, args.output, args.minutes, args.step,
                   args.label, args.fault_service, args.metrics)


if __name__ == "__main__":
    main()
