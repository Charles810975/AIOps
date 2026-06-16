# -*- coding: utf-8 -*-
"""
AIOps Agent (ReAct loop) with Agnes-2.0-flash.
"""
import json
import os
import sys
import time
import io
from openai import OpenAI

# force utf-8 stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

# local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import TOOLS, TOOL_SCHEMAS, sr_detect, get_logs, restart_pod, get_pod_status

AGNES_API_KEY  = "sk-cuFAlEhZnHX3nlrAkPo9HcSqpx4siryXEKMAH2mwwTXGhV9T"
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL    = "agnes-2.0-flash"

SYSTEM_PROMPT = """你是 AIOps 自愈 Agent，运行在 K8s online-boutique 集群上。
你可以调用 4 个工具：sr_detect（用 v3 SR-CNN 检测 CPU 异常）、get_logs（拉日志诊断）、
restart_pod（重启 pod）、get_pod_status（看 pod 状态）。

工作流（ReAct）：
1. 先 sr_detect 评估异常严重程度
2. 若 score_peak > threshold，调 get_logs 找根因
3. 根据日志判断是否需要 get_pod_status 进一步确认
4. 若确认是 OOM/死锁/异常重启 等可自愈故障，调用 restart_pod(dry_run=False)
5. 给出最终诊断结论和处置建议

输出要求：
- 中文，简洁，工程化
- 每次工具调用前先说明意图
- 最终结论包含：根因 / 已执行动作 / 建议后续
"""


class AIOpsAgent:
    def __init__(self, model: str = AGNES_MODEL):
        self.client = OpenAI(api_key=AGNES_API_KEY, base_url=AGNES_BASE_URL)
        self.model = model
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.steps = []  # (role, content) for display

    def _chat(self, messages, tools=None, tool_choice="auto"):
        kwargs = dict(model=self.model, messages=messages,
                      temperature=0.2, max_tokens=600, timeout=30)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def think(self, user_input: str, max_iter: int = 6) -> str:
        """Run ReAct loop, return final assistant text."""
        self.history.append({"role": "user", "content": user_input})
        self.steps.append(("USER", user_input))

        for i in range(max_iter):
            r = self._chat(self.history, tools=TOOL_SCHEMAS)
            msg = r.choices[0].message
            self.history.append(msg)
            self.steps.append(("ASSISTANT", msg.content or ""))

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                fn = TOOLS.get(tc.function.name)
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                self.steps.append(("TOOL_CALL", f"{tc.function.name}({args})"))
                if fn is None:
                    out = f"unknown tool: {tc.function.name}"
                else:
                    try:
                        out = fn(**args)
                        if not isinstance(out, str):
                            out = json.dumps(out, ensure_ascii=False, indent=2)
                    except Exception as e:
                        out = f"tool error: {e}"
                # truncate huge outputs
                if len(out) > 1500:
                    out = out[:1500] + "\n... [truncated]"
                self.steps.append(("TOOL_RESULT", out))
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out
                })

        return self.history[-1].content or "(max iter reached)"


def pretty_print(steps):
    print("\n" + "=" * 72)
    print("AIOps AGENT  ReAct trace")
    print("=" * 72)
    for kind, content in steps:
        bar = {"USER": "="*10 + " USER", "ASSISTANT": "-"*10 + " AGENT",
               "TOOL_CALL": ">>> TOOL", "TOOL_RESULT": "<<< RESULT"}[kind]
        print(f"\n{bar}")
        print(content)


if __name__ == "__main__":
    # get current pod name dynamically
    try:
        out = subprocess.check_output(
            ["kubectl", "get", "pod", "-n", "online-boutique",
             "-l", "app=cartservice",
             "-o", "jsonpath={.items[0].metadata.name}"],
            timeout=10
        ).decode("utf-8").strip()
    except Exception:
        out = "cartservice-77f8cfdff-b94mn"
    pod = out or "cartservice-77f8cfdff-b94mn"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "巡检"
    user_input = (f"{cmd}：pod={pod}, namespace=online-boutique。"
                  f"先 sr_detect 评估，再决定是否拉日志/重启。")
    agent = AIOpsAgent()
    t0 = time.time()
    final = agent.think(user_input)
    dt = time.time() - t0
    print(f"\n[agent latency] {dt:.1f}s, {len(agent.steps)} trace steps\n")
    pretty_print(agent.steps)
    print("\n" + "=" * 72)
    print("FINAL DIAGNOSIS")
    print("=" * 72)
    print(final)
