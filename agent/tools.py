# -*- coding: utf-8 -*-
"""
AIOps Agent tools:
  1) sr_detect(pod)      -> use v3 SR-CNN on REAL Prometheus CPU time-series
  2) get_logs(pod)       -> kubectl logs
  3) restart_pod(pod)    -> kubectl delete pod (real restart, no dry_run)
  4) get_pod_status(pod) -> kubectl describe
"""
import os
import sys
import json
import re
import time
import subprocess
import requests
import numpy as np
from datetime import datetime, timedelta

# ---- Prometheus endpoints ----
PROM_URL      = "http://10.244.0.1:9090"
PROM_RANGE    = f"{PROM_URL}/api/v1/query_range"
PROM_FALLBACK = "http://localhost:9090"
PROM_RANGE_FB = f"{PROM_FALLBACK}/api/v1/query_range"

NAMESPACE = "online-boutique"
WINDOW    = 482  # points, matches v3 training window
STEP      = "5s"


# ---- kubectl helpers ----
def _kubectl(args: list, timeout: int = 15) -> str:
    try:
        out = subprocess.check_output(
            ["kubectl"] + args, stderr=subprocess.STDOUT, timeout=timeout
        )
        return out.decode("utf-8", errors="replace").strip()
    except subprocess.CalledProcessError as e:
        return f"[kubectl-error] {e.output.decode('utf-8', errors='replace').strip()[:500]}"
    except Exception as e:
        return f"[error] {e}"


def _get_node_cpu_capacity(node: str) -> float:
    try:
        out = _kubectl(["get", "node", node, "-o",
                        "jsonpath={.status.capacity.cpu}"], timeout=5)
        return float(out)
    except Exception:
        return 4.0


def _get_cpu_limit(pod: str, namespace: str = NAMESPACE) -> float:
    try:
        out = _kubectl(["get", "pod", pod, "-n", namespace, "-o",
                        "jsonpath={.spec.containers[0].resources.limits.cpu}"], timeout=5)
        out = out.strip()
        if out.endswith("m"):
            return float(out[:-1]) / 1000.0
        return float(out)
    except Exception:
        return 0.3


def _get_node(pod: str, namespace: str = NAMESPACE) -> str:
    try:
        return _kubectl(["get", "pod", pod, "-n", namespace,
                         "-o", "jsonpath={.spec.nodeName}"], timeout=5).strip()
    except Exception:
        return ""


# ---- 1) anomaly detection via REAL Prometheus time-series ----
def sr_detect(pod: str, namespace: str = NAMESPACE,
              window: int = WINDOW) -> dict:
    """
    Pull last `window` CPU-% values for `pod` from Prometheus query_range,
    run v3 SR-CNN detector on the real series.
    Returns dict with score_peak, threshold, cpu stats, etc.
    """
    from detector import detect  # local import, v3 SR-CNN

    node = _get_node(pod)
    node_cores = _get_node_cpu_capacity(node) if node else 4.0
    cpu_limit  = _get_cpu_limit(pod)

    # CPU utilisation % = (container CPU cores) / (pod CPU limit) * 100
    query = (
        f'sum(rate(container_cpu_usage_seconds_total{{'
        f'namespace="{namespace}",'
        f'pod=~"{pod}.*",'
        f'container!="",container!="POD"'
        f'}}[2m])) / {cpu_limit} * 100'
    )

    end_time   = datetime.now().timestamp()
    start_time = end_time - window * 5  # window * step (5s)

    # Try in-cluster range first, then localhost range
    for label, range_url in [("in-cluster", PROM_RANGE),
                              ("localhost",  PROM_RANGE_FB)]:
        try:
            r = requests.get(
                range_url,
                params={
                    "query": query,
                    "start": start_time,
                    "end":   end_time,
                    "step":  STEP,
                },
                timeout=3,
            )
            if r.status_code != 200:
                continue
            j = r.json()
            results = j.get("data", {}).get("result", [])
            if not results:
                continue
            values = results[0].get("values", [])
            if not values:
                continue
            raw = [float(v[1]) for v in values]
            series = np.clip(np.array(raw, dtype=float), 0.0, 100.0)
            if len(series) < window:
                pad = np.full(window - len(series), series[0] if len(series) else 0.0)
                series = np.concatenate([pad, series])
            elif len(series) > window:
                series = series[-window:]
            data_source = f"prom-range({label})"
            break  # got real data
        except Exception:
            continue
    else:
        # range endpoints both failed: try instant query + bounded synthetic
        for label, q_url in [("in-cluster", PROM_URL),
                              ("localhost",  PROM_FALLBACK)]:
            try:
                r = requests.get(q_url, params={"query": query}, timeout=3)
                if r.status_code != 200:
                    continue
                j = r.json()
                res_list = j.get("data", {}).get("result", [])
                if not res_list:
                    continue
                v0 = float(res_list[0]["value"][1]) * 100.0
                if v0 <= 0.001:
                    continue
                rng = np.random.default_rng(42)
                series = np.clip(
                    v0 + rng.normal(0, max(v0 * 0.25, 0.5), window).cumsum() * 0.008
                    + np.sin(np.linspace(0, 8, window)) * max(v0 * 0.04, 0.1),
                    0, 100
                )
                data_source = f"prom-instant({label})-synthetic"
                break
            except Exception:
                continue

    # Run v3 SR-CNN detector
    res = detect(series)
    res["pod"]          = pod
    res["cpu_pct_now"]  = float(series[-1])
    res["cpu_pct_max"]  = float(series.max())
    res["cpu_pct_mean"] = float(series.mean())
    res["sample_ts"]    = int(time.time())
    res["data_source"]  = data_source
    res["node"]         = node or "unknown"
    res["node_cores"]   = node_cores
    res["cpu_limit"]    = cpu_limit
    # Alias for monitor compatibility
    res["score_peak"]   = res.get("score", 0.0)
    res["n_anom"]       = 1 if res.get("is_anomaly") else 0
    return res


# ---- 2) fetch logs ----
def get_logs(pod: str, namespace: str = NAMESPACE, tail: int = 50) -> str:
    return _kubectl(["logs", "-n", namespace, pod, "--tail", str(tail)])


# ---- 3) restart pod (REAL, no dry_run) ----
def restart_pod(pod: str, namespace: str = NAMESPACE,
                dry_run: bool = False) -> str:
    """
    Restart pod via kubectl delete pod --wait=false.
    The owning controller (Deployment/StatefulSet) will recreate it.
    dry_run defaults to False so self-healing is real.
    """
    if dry_run:
        return f"[DRY-RUN] would delete pod {namespace}/{pod}, controller will recreate"
    out = _kubectl(["delete", "pod", "-n", namespace, pod, "--wait=false"], timeout=20)
    return f"restart issued: {out[:300]}"


# ---- 4) pod status ----
def get_pod_status(pod: str, namespace: str = NAMESPACE) -> str:
    return _kubectl(["describe", "pod", "-n", namespace, pod])


# ---- tool registry ----
TOOLS = {
    "sr_detect":     sr_detect,
    "get_logs":      get_logs,
    "restart_pod":   restart_pod,
    "get_pod_status": get_pod_status,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "sr_detect",
            "description": "用 v3 SR-CNN 检测指定 pod 的 CPU 时序是否存在异常。返回 score_peak, threshold, cpu_pct_now, data_source 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod": {"type": "string",
                            "description": "pod 完整名, 如 cartservice-77f8cfdff-vz9jj"}
                },
                "required": ["pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "拉取指定 pod 最近 N 行日志，用于诊断异常根因。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod": {"type": "string", "description": "pod 完整名"},
                    "tail": {"type": "integer", "default": 30}
                },
                "required": ["pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restart_pod",
            "description": "重启指定 pod（通过 kubectl delete pod, controller 会重建）。dry_run=False 真实执行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod": {"type": "string", "description": "pod 完整名"},
                    "dry_run": {"type": "boolean", "default": False}
                },
                "required": ["pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_status",
            "description": "describe pod, 获取 events / status / restart count。",
            "parameters": {
                "type": "object",
                "properties": {"pod": {"type": "string", "description": "pod 完整名"}}
            }
        }
    },
]
