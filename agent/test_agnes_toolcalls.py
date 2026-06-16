"""测试 Agnes 是否支持 function calling (tool_calls)"""
import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-cuFAlEhZnHX3nlrAkPo9HcSqpx4siryXEKMAH2mwwTXGhV9T",
    base_url="https://apihub.agnes-ai.com/v1"
)

# 1.5-flash 和 2.0-flash 都试一下
for model in ["agnes-2.0-flash", "agnes-1.5-flash"]:
    print("=" * 60)
    print(f"model: {model}")
    print("=" * 60)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "查询 cartservice 服务当前 CPU 使用率"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_cpu",
                    "description": "查询某个 K8s pod 的 CPU 使用率",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pod_name": {"type": "string", "description": "pod 名字"},
                        },
                        "required": ["pod_name"]
                    }
                }
            }],
            tool_choice="auto",
            max_tokens=200,
            timeout=30
        )
        msg = r.choices[0].message
        print(f"content:    {msg.content}")
        print(f"tool_calls: {msg.tool_calls}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  -> {tc.function.name}({tc.function.arguments})")
            print("[SUPPORTS] function calling")
        else:
            print("[NO TOOL CALL] 纯对话, 无法用 tool_calls")
    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode('ascii')[:300]
        print(f"[FAIL] {err}")
    print()
