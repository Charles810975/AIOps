#!/usr/bin/env python3
"""Quick Prometheus connectivity and metric availability check."""

import requests

PROMETHEUS = "http://localhost:9090"


def check_api(path, params=None):
    try:
        r = requests.get(PROMETHEUS + path, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


print("=== 1. Prometheus port-forward ===")
r = check_api("/-/healthy")
if "error" in r:
    print(f"[FAIL] Cannot connect to {PROMETHEUS}")
    print("       Run: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090")
else:
    print(f"[OK] Prometheus healthy")

print("\n=== 2. Check available container metrics ===")
container_metrics = [
    "container_cpu_usage_seconds_total",
    "container_memory_working_set_bytes",
    "container_memory_usage_bytes",
    "container_network_receive_bytes_total",
    "container_network_transmit_bytes_total",
    "kube_pod_container_status_restarts_total",
]

for m in container_metrics:
    result = check_api("/api/v1/query", {"query": f"count({m})"})
    if "error" in result:
        print(f"[ERR] {m}  ->  {result['error']}")
    elif result.get("status") == "success":
        count = result["data"]["result"][0]["value"][1]
        print(f"[OK]  {m}  ->  {count} series")
    else:
        print(f"[---] {m}  ->  no data")

print("\n=== 3. Check namespace on network metrics ===")
for m in ["container_network_receive_bytes_total", "container_network_transmit_bytes_total"]:
    result = check_api("/api/v1/query", {"query": m})
    if result.get("status") == "success" and result["data"]["result"]:
        sample_labels = result["data"]["result"][0]["metric"]
        print(f"  Sample labels: {sample_labels}")
    else:
        print(f"  [---] {m}: 0 series")

print("\n=== 4. Alternative: node-level network metrics ===")
for m in ["node_network_receive_bytes_total", "node_network_transmit_bytes_total"]:
    result = check_api("/api/v1/query", {"query": f"count({m})"})
    if result.get("status") == "success":
        count = result["data"]["result"][0]["value"][1]
        print(f"[OK]  {m}  ->  {count} series")
    else:
        print(f"[---] {m}  ->  no data")

print("\n=== 5. Check minikube namespace context ===")
result = check_api("/api/v1/query", {"query": 'count({__name__=~"container_.*", namespace="online-boutique"})'})
if result.get("status") == "success":
    count = result["data"]["result"][0]["value"][1]
    print(f"  container_* metrics in 'online-boutique' namespace: {count} series")
else:
    print(f"  [---] No container_* metrics in 'online-boutique' namespace")

result2 = check_api("/api/v1/query", {"query": 'count({__name__=~"container_.*"})'})
if result2.get("status") == "success":
    count2 = result2["data"]["result"][0]["value"][1]
    print(f"  All container_* metrics (any namespace): {count2} series")

print("\n=== 6. What's actually in Prometheus? (top 20 metric names) ===")
result = check_api("/api/v1/label/__name__/values")
if result.get("status") == "success":
    names = result["data"]
    network_names = [n for n in names if "network" in n.lower()]
    container_names = [n for n in names if "container" in n.lower()]
    print(f"  Total metric names: {len(names)}")
    if network_names:
        print(f"  Network-related: {network_names}")
    else:
        print(f"  Network-related: NONE")
    if container_names:
        print(f"  Container-related (first 10): {container_names[:10]}")
else:
    print(f"  [---] Cannot get metric names: {result}")
