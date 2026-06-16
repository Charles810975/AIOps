# -*- coding: utf-8 -*-
"""
AIOps Agent — minimal closed-loop demo with screenshots.

3 scenarios in one run:
  S1 baseline  - normal CPU, agent says "all good"
  S2 anomaly   - inject high CPU; agent detects via SR-CNN, pulls logs, diagnoses
  S3 self-heal - agent calls restart_pod (real kubectl delete), K8s recreates

For each scenario we print a clean ReAct trace, then save:
  docs/screenshots/<scenario>.md  (markdown report)
  docs/screenshots/<scenario>.txt  (clean text for direct paste into slide)

The detector is the *real* v3 SR-CNN loaded from
experiments/sr_iter03_podcartservice_mcpu_v3/best.json — only the input
series is synthetic + controllable so the demo is deterministic.
"""
import io, os, sys, json, time, subprocess

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from openai import OpenAI

from detector import detect, V3_BEST
from tools import get_logs, get_pod_status

# ------------- Agnes -------------
AGNES_API_KEY  = "sk-cuFAlEhZnHX3nlrAkPo9HcSqpx4siryXEKMAH2mwwTXGhV9T"
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL    = "agnes-2.0-flash"
client = OpenAI(api_key=AGNES_API_KEY, base_url=AGNES_BASE_URL)

# ------------- tool schemas -------------
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "sr_detect",
        "description": "用 v3 SR-CNN 检测指定 pod 的 CPU 时序是否存在异常。返回 score 峰值、阈值、异常点数、当前 CPU% 等。",
        "parameters": {"type": "object", "properties": {
            "pod": {"type": "string", "description": "pod 完整名"}
        }, "required": ["pod"]}
    }},
    {"type": "function", "function": {
        "name": "get_logs",
        "description": "拉取指定 pod 最近 N 行日志，用于诊断异常根因。",
        "parameters": {"type": "object", "properties": {
            "pod": {"type": "string", "description": "pod 完整名"},
            "tail": {"type": "integer", "default": 30}
        }, "required": ["pod"]}
    }},
    {"type": "function", "function": {
        "name": "restart_pod",
        "description": "重启指定 pod。dry_run=True 时只模拟不执行。",
        "parameters": {"type": "object", "properties": {
            "pod": {"type": "string", "description": "pod 完整名"},
            "dry_run": {"type": "boolean", "default": True}
        }, "required": ["pod"]}
    }},
    {"type": "function", "function": {
        "name": "get_pod_status",
        "description": "describe pod, 获取 events / status / restart count。",
        "parameters": {"type": "object", "properties": {
            "pod": {"type": "string", "description": "pod 完整名"}
        }, "required": ["pod"]}
    }},
]

SYSTEM = """你是 AIOps 自愈 Agent，运行在 K8s online-boutique 集群上。
你可以调用 4 个工具：sr_detect（用 v3 SR-CNN 检测 CPU 异常）、get_logs（拉日志诊断）、
restart_pod（重启 pod）、get_pod_status（看 pod 状态）。

工作流（ReAct）：
1. 先 sr_detect 评估异常严重程度
2. 若 score_peak > threshold，调 get_logs 找根因
3. 根据日志判断是否需要 get_pod_status 进一步确认
4. 若确认是 OOM/死锁/异常重启 等可自愈故障，调用 restart_pod(dry_run=False)
5. 给出最终诊断结论和处置建议

输出要求：中文，简洁，工程化。每次工具调用前先说明意图。
最终结论包含：根因 / 已执行动作 / 建议后续。
"""

# ------------- helpers -------------

def make_series(mode: str, n: int = 482) -> np.ndarray:
    """Make a deterministic CPU series for the demo.

    Calibrated so SR-CNN score_peak (v3 best) lands in:
      baseline: < 0.20 (no anomaly)
      anomaly : 0.45-0.70 (above 0.39 threshold)
      critical: > 0.80 (strongly above threshold)
    """
    rng = np.random.default_rng({"baseline": 1, "anomaly": 2, "critical": 3}[mode])
    if mode == "baseline":
        # very low frequency, no spikes -> low SR saliency
        s = (2.0
             + np.sin(np.linspace(0, 0.5, n)) * 0.3
             + rng.normal(0, 0.05, n))
        return np.clip(s, 0, 100)
    if mode == "anomaly":
        # gentle ramp + repeated moderate spikes
        s = np.linspace(20, 75, n) + rng.normal(0, 1.5, n)
        s[150:165] += 40
        s[280:295] += 50
        s[400:415] += 45
        return np.clip(s, 0, 100)
    if mode == "critical":
        # saturated + multiple spikes -> high SR score
        s = np.full(n, 90.0) + rng.normal(0, 1.5, n)
        s[100:115] += 80
        s[240:255] += 100
        s[380:395] += 60
        return np.clip(s, 0, 200)
    raise ValueError(mode)


def tool_sr_detect(pod: str) -> str:
    """Use a mode flag we stash in the agent state, not the pod name."""
    mode = _current_mode
    series = make_series(mode)
    r = detect(series)
    r["pod"] = pod
    r["cpu_pct_now"] = float(series[-1])
    r["cpu_pct_max"] = float(series.max())
    r["cpu_pct_mean"] = float(series.mean())
    r["mode"] = mode
    return json.dumps(r, ensure_ascii=False, indent=2)


def tool_get_logs(pod: str, tail: int = 30) -> str:
    mode = _current_mode
    if mode == "baseline":
        return "\n".join(["INFO  cartservice: CheckoutService] PlaceOrder called (cart_id=7c2a)"] * 8
                       + ["INFO  redis: GET cart=7c2a -> 3 items"] * 5
                       + ["INFO  cartservice: empty cart, status=200"] * 5
                       + [f"INFO  cartservice: heartbeat #{i}" for i in range(tail - 18)])
    if mode == "anomaly":
        return "\n".join([f"WARN  cartservice: CPU saturated {x:.1f}%"
                          for x in np.linspace(40, 95, 12)]
                       + ["ERROR cartservice: request timeout (3000ms exceeded)"] * 5
                       + ["WARN  checkout: downstream=cartservice latency 2800ms"] * 6
                       + ["ERROR cartservice: goroutine leak detected, count=2417"] * 4
                       + [f"INFO  cartservice: request_id=req-{i:04d} status=500" for i in range(3)])
    if mode == "critical":
        return "\n".join(["FATAL cartservice: Out of memory (heap=128Mi limit)"] * 6
                       + ["ERROR cartservice: OOMKilled by kubelet"] * 4
                       + ["WARN  cartservice: GC pause 4.2s (long, hurts p99)"] * 3
                       + ["FATAL cartservice: panic: runtime error: invalid memory address"] * 3
                       + ["INFO  kubelet: Killing container with PID 1 due to OOM"] * 3
                       + ["ERROR checkout: 5xx rate=72% (upstream cartservice)"] * 3
                       + [f"INFO  cartservice: retry_id={i} status=500" for i in range(3)])
    return ""


def tool_restart_pod(pod: str, dry_run: bool = True) -> str:
    if dry_run:
        return f"[DRY-RUN] would delete pod {pod}; K8s controller will recreate"
    out = subprocess.run(
        ["kubectl", "delete", "pod", "-n", "online-boutique", pod, "--wait=false"],
        capture_output=True, timeout=20, text=True
    )
    return f"restart issued: {out.stdout.strip()[:200] or out.stderr.strip()[:200]}"


def tool_get_pod_status(pod: str) -> str:
    return subprocess.run(
        ["kubectl", "describe", "pod", "-n", "online-boutique", pod],
        capture_output=True, timeout=15, text=True
    ).stdout[:1500] or "(no output)"


TOOLS = {
    "sr_detect": tool_sr_detect,
    "get_logs": tool_get_logs,
    "restart_pod": tool_restart_pod,
    "get_pod_status": tool_get_pod_status,
}

_current_mode = "baseline"


# ------------- agent -------------
def run_agent(query: str, mode: str, max_iter: int = 6) -> dict:
    """Run one ReAct session and return the trace + final text."""
    global _current_mode
    _current_mode = mode

    history = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]
    trace = [("USER", query)]
    final = ""
    t0 = time.time()
    for i in range(max_iter):
        r = client.chat.completions.create(
            model=AGNES_MODEL, messages=history,
            tools=TOOL_SCHEMAS, tool_choice="auto",
            temperature=0.2, max_tokens=600, timeout=45
        )
        msg = r.choices[0].message
        history.append(msg)
        trace.append(("ASSISTANT", msg.content or ""))
        if not msg.tool_calls:
            final = msg.content or ""
            break
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            trace.append(("TOOL_CALL", f"{tc.function.name}({args})"))
            try:
                out = TOOLS[tc.function.name](**args)
            except Exception as e:
                out = f"tool error: {e}"
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False, indent=2)
            if len(out) > 1500:
                out = out[:1500] + "\n... [truncated]"
            trace.append(("TOOL_RESULT", out))
            history.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    else:
        final = history[-1].content or "(max iter reached)"
    return {"trace": trace, "final": final, "latency": time.time() - t0}


# ------------- screenshot output -------------
def write_screenshot(tag: str, query: str, mode: str, result: dict):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "docs", "screenshots")
    os.makedirs(out_dir, exist_ok=True)

    md = [f"# AIOps Agent — {tag}（{mode}）\n",
          f"**query**: {query}\n",
          f"**model**: agnes-2.0-flash  ",
          f"**latency**: {result['latency']:.1f}s  ",
          f"**trace steps**: {len(result['trace'])}\n",
          "\n---\n\n## ReAct Trace\n"]
    txt = [f"AIOps Agent — {tag}（{mode}）",
           f"query: {query}",
           f"model: agnes-2.0-flash | latency: {result['latency']:.1f}s",
           "=" * 60, ""]
    for kind, content in result["trace"]:
        if kind == "USER":
            md.append(f"\n**[USER]**\n> {content}\n")
            txt.append(f"[USER]\n{content}\n")
        elif kind == "ASSISTANT":
            md.append(f"\n**[AGENT]**\n{content or '(thinking...)'}\n")
            txt.append(f"\n[AGENT]\n{content or '(thinking...)'}\n")
        elif kind == "TOOL_CALL":
            md.append(f"\n**>>> TOOL_CALL** `{content}`\n")
            txt.append(f"\n>>> TOOL_CALL  {content}\n")
        elif kind == "TOOL_RESULT":
            short = content[:800] + ("\n... [truncated]" if len(content) > 800 else "")
            md.append(f"\n**<<< RESULT**\n```\n{short}\n```\n")
            txt.append(f"\n<<< RESULT\n{short}\n")
    md.append("\n---\n\n## Final Diagnosis\n"); md.append(result["final"])
    txt.append("\n" + "=" * 60); txt.append("FINAL DIAGNOSIS")
    txt.append(result["final"])

    base = os.path.join(out_dir, tag)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("".join(md))
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print(f"saved: {base}.md  +  {base}.txt")


# ------------- main -------------
if __name__ == "__main__":
    # Get current pod
    try:
        pod = subprocess.check_output(
            ["kubectl", "get", "pod", "-n", "online-boutique", "-l", "app=cartservice",
             "-o", "jsonpath={.items[0].metadata.name}"],
            timeout=10
        ).decode("utf-8").strip() or "cartservice-77f8cfdff-b94mn"
    except Exception:
        pod = "cartservice-77f8cfdff-b94mn"

    scenarios = [
        ("S1-baseline",  "baseline",
         f"对 cartservice pod（{pod}）做一次例行健康巡检，评估 CPU 时序是否异常。"),
        ("S2-anomaly",   "anomaly",
         f"SR-CNN 检测到 cartservice（{pod}）score_peak 超过阈值，请按 ReAct 工作流诊断根因。"),
        ("S3-selfheal",  "critical",
         f"cartservice（{pod}）持续 5 分钟 CPU > 95%，日志大量 OOM Killed，已影响 checkout 链路订单转化率。"
         f"请诊断并自愈。"),
    ]
    for tag, mode, q in scenarios:
        print(f"\n{'#' * 70}\n# {tag}  ({mode})\n{'#' * 70}")
        r = run_agent(q, mode, max_iter=6)
        print(f"latency: {r['latency']:.1f}s  | steps: {len(r['trace'])}")
        write_screenshot(tag, q, mode, r)

    print("\nAll scenarios captured. Files in docs/screenshots/")
