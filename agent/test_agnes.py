import os
import requests
from openai import OpenAI

api_key  = "sk-cuFAlEhZnHX3nlrAkPo9HcSqpx4siryXEKMAH2mwwTXGhV9T"
base_url = "https://apihub.agnes-ai.com/v1"

print("=" * 60)
print("Step 1: 验证 base_url 通了 (列出可用 model)")
print("=" * 60)
try:
    r = requests.get(f"{base_url}/models",
                     headers={"Authorization": f"Bearer {api_key}"},
                     timeout=15)
    print(f"HTTP {r.status_code}")
    print(r.text[:2000])
except Exception as e:
    print(f"REQ FAIL: {e}")

print()
print("=" * 60)
print("Step 2: 用通用名试一下 chat")
print("=" * 60)
client = OpenAI(api_key=api_key, base_url=base_url)

# 直接试真名
candidates = ["agnes-1.5-flash", "agnes-2.0-flash"]
for m in candidates:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "ping, 只回 pong"}],
            max_tokens=10,
            timeout=15
        )
        content = r.choices[0].message.content
        print(f"[OK] {m:25s} | {content}")
    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode('ascii')[:160]
        print(f"[FAIL] {m:25s} | {err}")
