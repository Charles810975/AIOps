# AIOps Agent（ Agnes 全模态 LLM 驱动）

最小闭环 AIOps 自愈 Agent。**目标：用 v3 训练好的 SR-CNN 检测器 + Agnes 全模态 LLM 做 ReAct 推理，对 K8s pod 异常进行检测 → 诊断 → 自愈。**

---

## 🎬 一键部署

```bash
py monitor.py
```
每隔五秒轮询

---

## 🏗️ 架构

```
┌─────────────┐  ReAct  ┌──────────┐  tool_call  ┌──────────────┐
│  USER query │ ──────▶ │  Agnes   │ ──────────▶ │ 4 工具       │
└─────────────┘         │  LLM     │ ◀────────── │ sr_detect    │
                        │  (云端)  │ tool_result │ get_logs     │
                        └──────────┘             │ get_pod_status│
                                                │ restart_pod  │
                                                └──────────────┘
```

- **LLM**：`https://apihub.agnes-ai.com/v1` 上 `agnes-2.0-flash`（OpenAI 兼容协议）
- **Detector**：`detector.detect()`，加载 `experiments/sr_iter03_podcartservice_mcpu_v3/best.json` 的 v3 超参
- **Tools**：4 个函数，真实对接 K8s（`kubectl get/logs/describe/delete`）

---

## 📁 文件结构

```
agent/
├── detector.py        # v3 SR-CNN detector（severity 评分）
├── tools.py           # 4 工具：sr_detect / get_logs / restart_pod / get_pod_status
├── aiops_agent.py     # ReAct 循环（独立命令行版）
├── monitor.py         # 5s 轮询 monitor
├── demo.py            # ⭐ 一键跑 3 场景 + 写截图
├── fault_inject.py    # 故障注入（CPU 压测）
├── capture_screenshots.py  # 截图采集（旧版）
├── test_agnes.py      # 验证 Agnes 联通 + model 列表
└── test_agnes_toolcalls.py # 验证 function calling 支持

```

---

## 🧪 独立验证 Agnes

```bash
py test_agnes.py
# 列出 5 个可用 model，ping 验证 key 有效

py test_agnes_toolcalls.py
# 验证 agnes-1.5-flash / agnes-2.0-flash 都支持 function calling
```

**model 列表**（来自 `apihub.agnes-ai.com/v1/models`）：

| model | 用途 |
|---|---|
| `agnes-2.0-flash` | 文本对话（**默认**） |
| `agnes-1.5-flash` | 文本对话（备选） |
| `agnes-image-2.0/2.1-flash` | 图片理解 |
| `agnes-video-v2.0` | 视频理解 |

---

## 🧠 Agent 推理工作流（ReAct）

```
1. sr_detect   → 拿到 score + severity + is_anomaly
   ↓
2. get_logs    → 拉 30 行日志找根因（goroutine leak / OOM / panic）
   ↓
3. get_pod_status → 确认 Pod 当前状态（Running / RestartCount）
   ↓
4. restart_pod(dry_run=False) → 真实 kubectl delete pod，K8s 重建
   ↓
5. 输出最终诊断：根因 / 已执行动作 / 建议后续
```

**真实动作**：restart_pod 在 S2/S3 里**真的删过 cartservice pod**，Deployment 自动重建。
可以用 `kubectl get pods -n online-boutique -l app=cartservice` 看到 pod 名变化。


## 🔑 Agnes API 接入要点

| 项 | 值 |
|---|---|
| base_url | `https://apihub.agnes-ai.com/v1` |
| auth | `Authorization: Bearer sk-cuFAlEhZnHX3nlrAkPo9HcSqpx4siryXEKMAH2mwwTXGhV9T` |
| 协议 | OpenAI 兼容（直接用 `openai` SDK） |
| function calling | ✅ 支持（已验证） |

---

## ⚠️ 已知限制

1. **detector 边缘效应**：workspace 里的 `spectral_residual` 实现对短序列边缘点有偏差，demo 用 `_severity()` 做了校准，但 SR-CNN 的"原始"score_max 仍可读出来作 v3 best.json 的对照
2. **get_pod_status 偶尔返回空**：当 describe 跨多页时，subprocess 输出可能被截断
3. **fault_inject 镜像拉不到**：集群不通外网（Docker Hub DNS 超时），所以没真注入 CPU 压测，demo 走"合成时序"路径
