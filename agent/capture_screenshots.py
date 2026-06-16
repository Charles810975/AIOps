# -*- coding: utf-8 -*-
"""
Capture 3 representative screenshots of the AIOps agent's ReAct trace,
saved as a single report file showing the agent's diagnostic capability.
"""
import os
import sys
import io
import time
import json
import subprocess

# utf-8 everywhere
for s in (sys.stdout, sys.stderr):
    if hasattr(s, "buffer"):
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiops_agent import AIOpsAgent

SCENARIOS = [
    ("1-routine",    "巡检 cartservice pod 状态，做一次基线评估"),
    ("2-anomaly",    "检测到 cartservice CPU 异常飙升（score_peak > threshold），请诊断根因"),
    ("3-selfheal",   "cartservice 持续 5 分钟 CPU > 95%，日志大量 OOM Killed，已影响订单，请自愈"),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

for tag, query in SCENARIOS:
    print(f"\n{'#' * 72}\n# SCENARIO: {tag}\n# query: {query}\n{'#' * 72}\n")
    agent = AIOpsAgent()
    t0 = time.time()
    final = agent.think(
        f"{query}。pod=cartservice-77f8cfdff-*, namespace=online-boutique。"
        f"请按 ReAct 工作流分析。"
    )
    dt = time.time() - t0

    # write report
    report = []
    report.append(f"# AIOps Agent Scenario: {tag}\n")
    report.append(f"**query**: {query}\n")
    report.append(f"**model**: agnes-2.0-flash  ")
    report.append(f"**latency**: {dt:.1f}s  ")
    report.append(f"**trace steps**: {len(agent.steps)}\n")
    report.append("\n---\n\n## ReAct Trace\n")
    for kind, content in agent.steps:
        if kind == "USER":
            report.append(f"\n**USER**\n> {content}\n")
        elif kind == "ASSISTANT":
            report.append(f"\n**AGENT**\n{content or '(thinking...)'}\n")
        elif kind == "TOOL_CALL":
            report.append(f"\n**>>> TOOL_CALL** `{content}`\n")
        elif kind == "TOOL_RESULT":
            short = content[:800] + ("\n... [truncated]" if len(content) > 800 else "")
            report.append(f"\n**<<< RESULT**\n```\n{short}\n```\n")
    report.append("\n---\n\n## Final Diagnosis\n")
    report.append(final)
    report.append("\n")

    out_path = os.path.join(OUT_DIR, f"{tag}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    print(f"saved: {out_path}")
