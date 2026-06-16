# v3 故障数据采集使用说明

## 文件清单

| 文件 | 用途 |
|---|---|
| `data-collection/collect_v3_faults.py` | 主采集脚本：5s 步长 + 4 套强注入 + 自适应评估 |
| `data-collection/augment_v3_data.py` | 兜底数据增强器（F1 仍不达标时使用） |
| `deploy/chaos-mesh/cartservice-cpu-v3-burst.yaml` | cartservice CPU 4 核满载 |
| `deploy/chaos-mesh/redis-cart-cpu-v3-burst.yaml` | redis-cart CPU 2 核满载 |
| `deploy/chaos-mesh/cartservice-pod-kill-v3.yaml` | cartservice PodKill (35m) |
| `deploy/chaos-mesh/redis-cart-pod-kill-v3.yaml` | redis-cart PodKill (35m) |

## 4 套场景

| Key | 注入方式 | 关键信号指标 | 周期 |
|---|---|---|---|
| `cart-cpu` | `stress-ng` 4 workers × 100% load | cpu_throttle_ratio, cpu_usage | 持续 35m |
| `redis-cpu` | `stress-ng` 2 workers × 100% load | cpu_throttle_ratio | 持续 35m |
| `cart-kill` | chaos-mesh PodChaos + 外部 90s 循环 kill | restart_count, fs_read_bytes | 90s × 23 = 35m |
| `redis-kill` | chaos-mesh PodChaos + 外部 120s 循环 kill | restart_count, fs_read_bytes | 120s × 17 = 35m |

## 完整执行流程

### 1. 启动 Prometheus port-forward（如果还没启动）

```powershell
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-stack-prometheus 9090:9090 --address 0.0.0.0
```

### 2. 启动负载（让 online-boutique 有正常流量）

另开终端：
```powershell
.\deploy\load-gen\run-load.ps1
```

### 3. 跑全部 4 个场景（不评估，只采集）

```powershell
python data-collection/collect_v3_faults.py --fault all --no-eval
```

预计耗时：**4 场景 × 60 分钟 = 4 小时**（可后台运行）

### 4. 评估 + 自适应增强（采集完后）

```powershell
python data-collection/collect_v3_faults.py --fault all
```

逻辑：
- 每场景自动跑 1 轮 SR-CNN
- 若 F1 < 0.5 → 自动注入显式尖峰 → 再跑 1 轮（最多 3 轮）
- 达标即停；不达标会输出 `augmented_combined.csv` 供 `augment_v3_data.py` 使用

### 5. 兜底增强（如果第 4 步仍不达标）

```powershell
python data-collection/augment_v3_data.py --scenario cart-cpu
python data-collection/augment_v3_data.py --scenario all
```

`augment_v3_data.py` 会：
1. 对 fault_service 行的关键指标叠加 **ramp + spike + drop** 三种异常模式
2. 试 4 组 SR-CNN 超参（threshold_quantile / amp_window / score_window）
3. 输出最佳 F1 + 对应配置到 `augment_summary.csv`

## 输出目录结构

```
data-collection-v3/
├── cart-cpu/
│   ├── normal.csv
│   ├── anomaly.csv
│   ├── combined.csv
│   ├── eval_round1/
│   │   └── sr_cnn_summary.csv
│   ├── eval_round2/
│   └── eval_round3/
├── redis-cpu/  ...
├── cart-kill/  ...
├── redis-kill/ ...
└── augment_summary.csv   ← augment_v3_data.py 输出
```

## 关键设计

- **5s 步长**：比 30s 步长细 6 倍，SR-CNN 谱残差窗口（默认 21 点 = 105s）才有足够数据
- **CPU 4 核满载** vs 容器 limit (300m = 0.3 核)：超 13 倍 → 强制 95%+ throttling
- **PodKill 90s 循环**：5s 步长下 18 个点/周期 → 锯齿明显
- **3 模式叠加增强**（ramp + spike + drop）：同时覆盖 SR-CNN 三种检测模式
- **4 组超参搜索**：threshold_quantile ∈ {0.95, 0.97, 0.98, 0.99} × amp_window ∈ {3, 5, 7}
