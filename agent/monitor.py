# -*- coding: utf-8 -*-
"""
Live monitor: every N seconds, pull real CPU time-series from Prometheus
(via query_range, last ~40 min @ 5s step → 482 points), run v3 SR-CNN.
If anomaly detected, invoke the AIOps agent to diagnose + self-heal.
"""
import os
import sys
import time
import json
import logging
import subprocess
import requests
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import detect
from aiops_agent import AIOpsAgent

NAMESPACE = "online-boutique"
TARGET_LB = "app=cartservice"
INTERVAL  = 5  # seconds
WINDOW    = 482  # points (≈40 min @ 5s step, matches v3 training window)
STEP      = "5s"

# Prometheus endpoints: try in-cluster first, fall back to localhost
PROM_URL      = "http://10.244.0.1:9090"
PROM_RANGE    = f"{PROM_URL}/api/v1/query_range"
PROM_FALLBACK = "http://localhost:9090"
PROM_RANGE_FB = f"{PROM_FALLBACK}/api/v1/query_range"

# quiet down openai / httpx noise
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def banner(title, char="#"):
    line = char * 70
    print(f"\n{line}\n  [{now_str()}] {title}\n{line}")


def get_live_pod() -> str:
    try:
        return subprocess.check_output(
            ["kubectl", "get", "pod", "-n", NAMESPACE, "-l", TARGET_LB,
             "-o", "jsonpath={.items[0].metadata.name}"],
            timeout=10
        ).decode("utf-8").strip()
    except Exception:
        return None


def _get_node_cpu_capacity(node: str) -> float:
    """Get node CPU capacity in cores; fall back to 4 cores."""
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "node", node, "-o",
             "jsonpath={.status.capacity.cpu}"],
            timeout=5
        ).decode().strip()
        return float(out)
    except Exception:
        return 4.0


def _get_cpu_limit(pod: str, namespace: str = NAMESPACE) -> float:
    """Get pod CPU limit in cores from the first container; fall back to 0.3."""
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "pod", pod, "-n", namespace, "-o",
             "jsonpath={.spec.containers[0].resources.limits.cpu}"],
            timeout=5
        ).decode().strip()
        if out.endswith("m"):
            return float(out[:-1]) / 1000.0
        return float(out)
    except Exception:
        return 0.3


def fetch_cpu_series(pod: str, window: int = WINDOW, step: str = STEP) -> np.ndarray:
    """
    Pull real CPU usage time-series from Prometheus via query_range.
    Returns an array of CPU % values (length=window).
    Falls back to zeros if Prometheus is unreachable.
    """
    # Determine node + CPU capacity to convert cores → %
    node = None
    node_cores = 4.0
    try:
        node = subprocess.check_output(
            ["kubectl", "get", "pod", pod, "-n", NAMESPACE,
             "-o", "jsonpath={.spec.nodeName}"],
            timeout=5
        ).decode().strip()
        if node:
            node_cores = _get_node_cpu_capacity(node)
    except Exception:
        pass

    cpu_limit = _get_cpu_limit(pod)

    # We want container CPU seconds / second  →  cores.
    # Divide by cpu_limit to get utilisation %.
    query = (
        f'sum(rate(container_cpu_usage_seconds_total{{'
        f'namespace="{NAMESPACE}",'
        f'pod=~"{pod}.*",'
        f'cpu="total"'
        f'}}[2m])) / {cpu_limit} * 100'
    )

    end_time   = datetime.now().timestamp()
    start_time = end_time - window * 5  # window * step (5s)

    # Try in-cluster range first, then localhost range.
    # Short timeout (3s) so we don't block the whole cycle on a dead host.
    for label, range_url in [("in-cluster", PROM_RANGE),
                              ("localhost",  PROM_RANGE_FB)]:
        try:
            r = requests.get(
                range_url,
                params={
                    "query": query,
                    "start": start_time,
                    "end":   end_time,
                    "step":  step,
                },
                timeout=3,
            )
            if r.status_code != 200:
                print(f"[{now_str()}] [warn] Prometheus {label} HTTP {r.status_code}")
                continue
            j = r.json()
            results = j.get("data", {}).get("result", [])
            if not results:
                print(f"[{now_str()}] [warn] Prometheus {label}: empty result")
                continue
            values = results[0].get("values", [])
            if not values:
                print(f"[{now_str()}] [warn] Prometheus {label}: no values in result")
                continue

            raw = [float(v[1]) for v in values]
            series = np.clip(np.array(raw, dtype=float), 0.0, 100.0)

            if len(series) < window:
                pad = np.full(window - len(series), series[0] if len(series) else 0.0)
                series = np.concatenate([pad, series])
            elif len(series) > window:
                series = series[-window:]

            print(f"[{now_str()}] [data] Prometheus ({label}): got {len(values)} real points, "
                  f"node={node or '?'} cores={node_cores:.1f} limit={cpu_limit:.3f}c, "
                  f"cpu_now={series[-1]:.1f}%")
            return series

        except Exception as e:
            print(f"[{now_str()}] [warn] Prometheus {label} unreachable: {e}")
            continue

    print(f"[{now_str()}] [warn] all Prometheus endpoints failed, returning zeros")
    return np.zeros(window)


def main():
    banner(f"Live monitor started (interval={INTERVAL}s)  target=cartservice (real Prometheus data)")
    cycle = 0
    incident = False

    while True:
        cycle += 1
        pod = get_live_pod()
        if not pod:
            print(f"[{now_str()}] [error] cannot resolve cartservice pod, retrying next cycle")
            time.sleep(INTERVAL)
            continue

        # 1) Collect REAL CPU time-series from Prometheus (no synthetic noise)
        series = fetch_cpu_series(pod)
        cpu_now = float(series[-1])

        # 2) Run v3 SR-CNN detector
        res = detect(series)
        is_anom = res["score_peak"] > res["threshold"]

        print(f"[{now_str()}] cycle {cycle:03d} | pod={pod[-12:]} | "
              f"cpu_now={cpu_now:.2f}% | "
              f"score={res['score']:.3f} thr={res['threshold']:.3f} | "
              f"{'[ANOMALY]' if is_anom else '[ok]'}")

        if is_anom and not incident:
            incident = True
            banner(f"[ANOMALY DETECTED] score={res['score']:.3f} > thr={res['threshold']:.3f}  "
                   f"cpu_now={cpu_now:.1f}%  -> engaging AIOps agent")

            agent = AIOpsAgent()
            user_input = (
                f"检测到 {pod} CPU 异常：score={res['score']:.3f}, 阈值={res['threshold']:.3f}, "
                f"当前 CPU={cpu_now:.1f}%。"
                f"请按 ReAct 工作流诊断并自愈。namespace={NAMESPACE}。"
            )
            try:
                final = agent.think(user_input)
                print("\n" + "=" * 70)
                print("AGENT FINAL OUTPUT")
                print("=" * 70)
                print(final)
            except Exception as e:
                print(f"[agent error] {e}")

            # Suppress re-trigger for 3 cycles while agent finishes
            for _ in range(3):
                time.sleep(INTERVAL)
            incident = False

        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[monitor] stopped by user")
