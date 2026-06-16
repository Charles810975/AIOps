#!/usr/bin/env python3
"""Check what metrics Prometheus actually has available."""

import requests
import json

PROMETHEUS = "http://localhost:9090"

def query(q):
    try:
        r = requests.get(f"{PROMETHEUS}/api/v1/query", params={"query": q}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def query_range(q, minutes=60):
    end = int(requests.get(f"{PROMETHEUS}/-/healthy", timeout=3).elapsed.total_seconds())  # just check alive
    now = 1893456000  # placeholder
    try:
        r = requests.get(f"{PROMETHEUS}/api/v1/query", params={"query": q}, timeout=10)
        return r.json()
    except:
        pass
    # Just list metric names
    return None

print("=== Checking all available metric names ===")
r = requests.get(f"{PROMETHEUS}/api/v1/label/__name__/values", timeout=10)
if r.status_code != 200:
    print(f"[FAIL] {r.status_code}: {r.text}")
else:
    names = r.json().get("data", [])
    print(f"Total metric names in Prometheus: {len(names)}\n")

    # Categorize
    network = [n for n in names if "network" in n.lower()]
    container = [n for n in names if "container" in n.lower()]
    pod = [n for n in names if "pod" in n.lower()]
    node = [n for n in names if "node_" in n.lower()]
    http = [n for n in names if "http" in n.lower() or "request" in n.lower() or "latency" in n.lower()]
    kube_state = [n for n in names if n.startswith("kube_")]
    prometheus = [n for n in names if "prometheus" in n.lower()]

    print(f"=== Network metrics ({len(network)}) ===")
    for n in network: print(f"  {n}")

    print(f"\n=== Container metrics ({len(container)}) ===")
    for n in container: print(f"  {n}")

    print(f"\n=== Pod-level metrics ({len(pod)}) ===")
    for n in pod: print(f"  {n}")

    print(f"\n=== Node metrics ({len(node)}) ===")
    for n in node: print(f"  {n}")

    print(f"\n=== HTTP/Request/Latency metrics ({len(http)}) ===")
    for n in http: print(f"  {n}")

    print(f"\n=== kube_* state metrics ({len(kube_state)}) ===")
    for n in kube_state[:20]: print(f"  {n}")
    if len(kube_state) > 20: print(f"  ... and {len(kube_state)-20} more")

    print(f"\n=== Sample per-category series counts ===")
    for cat_name, metric_list in [
        ("container_cpu_usage_seconds_total", ["container_cpu_usage_seconds_total"]),
        ("container_memory_working_set_bytes", ["container_memory_working_set_bytes"]),
        ("container_memory_usage_bytes", ["container_memory_usage_bytes"]),
        ("kube_pod_container_status_restarts_total", ["kube_pod_container_status_restarts_total"]),
        ("container_network_receive_bytes_total", ["container_network_receive_bytes_total"]),
        ("container_network_transmit_bytes_total", ["container_network_transmit_bytes_total"]),
        ("node_network_receive_bytes_total", ["node_network_receive_bytes_total"]),
    ]:
        for m in metric_list:
            r = query(f"count({m})")
            if r.get("status") == "success":
                val = r["data"]["result"][0]["value"][1] if r["data"]["result"] else "0"
                print(f"  {m}: {val} series")

    print(f"\n=== Check online-boutique namespace-specific metrics ===")
    # Check if namespace filter makes a difference
    r = query('count({__name__=~"container_.*", namespace="online-boutique"})')
    if r.get("status") == "success" and r["data"]["result"]:
        print(f"  container_* in 'online-boutique': {r['data']['result'][0]['value'][1]} series")
    else:
        print(f"  container_* in 'online-boutique': 0 series")

    r = query('count({__name__=~"container_.*"})')
    if r.get("status") == "success" and r["data"]["result"]:
        print(f"  container_* in ALL namespaces: {r['data']['result'][0]['value'][1]} series")

    print(f"\n=== List pods in online-boutique ===")
    r = query('count by (pod) (container_cpu_usage_seconds_total{namespace="online-boutique"})')
    if r.get("status") == "success":
        for item in r["data"]["result"]:
            print(f"  {item['metric'].get('pod', '?')}: {item['value'][1]} series")
