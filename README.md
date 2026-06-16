# Online Boutique 微服务部署、监控、测试与智能运维大作业
本项目围绕 Google Online Boutique 微服务系统完成部署、监控、测试、故障注入、异常检测、根因分析和智能运维 Agent 实现。
## 目标档位
第二档：自行选择更复杂的开源微服务系统进行部署、监控和维护。
本项目选择 Online Boutique，包含 frontend、cartservice、checkoutservice、productcatalogservice、paymentservice、shippingservice、recommendationservice、adservice、currencyservice、emailservice、redis-cart、loadgenerator 等服务。
## 论文复现
1. KDD 2019 SR-CNN / Time-Series Anomaly Detection Service at Microsoft
   - 复现 Spectral Residual 时间序列异常检测核心思想。
   - 使用 Prometheus 导出的 KPI 数据进行异常检测。
2. ISSRE 2024 KPIRoot
   - 复现其"相似性分析 + 因果分析 + 根因 KPI 排序"的核心思想。
   - 使用 Prometheus 多指标数据定位根因服务和根因指标。
   - 服务级 F1=1.0（@K=1），KPI 级 F1=1.0（@K=1）。
3. **KDD 2020 USAD / UnSupervised Anomaly Detection**
   - 复现 Adversarial Autoencoder + 2-player 训练策略的核心思想。
   - 在 cartservice 注入故障数据集上做 9 组实验，最佳结果：F1=0.89, Precision=1.0, Recall=0.81。
   - 三种阈值方法对比：deviation / adaptive / quantile，deviation 最佳。
## 项目结构
```text
deploy/
  online-boutique/     Online Boutique 部署脚本
  prometheus/          Prometheus 监控配置
  grafana/             Grafana 配置与 dashboard
  chaos-mesh/          ChaosMesh 故障注入配置
tests/
  selenium/            Selenium 功能测试
  jmeter/              JMeter 性能测试计划
data-collection/       Prometheus 数据导出脚本
algorithms/
  sr-cnn/              SR-CNN / Spectral Residual 复现
  kpiroot/             KPIRoot 根因定位复现
  usad/                USAD 对抗自编码器异常检测复现（KDD 2020）
agent/                 智能运维 Agent（Agnes 全模态 LLM + ReAct）
microservices/
  api-gateway/          API Gateway 微服务（路由 + 限流 + 指标导出 + Dashboard）
  notification-service/ Notification 微服务（告警接收、聚合、Web Dashboard）
reports/               实验报告和结果图
```
## 自研微服务

在 Online Boutique 原生 12 个服务之外，本项目新增两个微服务，承担"统一入口"和"智能告警展示"职责。

### API Gateway（`microservices/api-gateway/`）

基于 Flask + gunicorn 的统一入口网关，部署在 `default` namespace，NodePort `30080`（通过 `kubectl port-forward` 暴露为 `127.0.0.1:18080`）。

功能：
- 路径代理：`/route/<service>/<path>` 转发到 `frontend`、`productcatalog`、`cartservice`、`recommendationservice`
- 滑动窗口限流：默认 100 req / 60s，按客户端 IP 计数（`X-Forwarded-For` 优先，否则 `request.remote_addr`）
- Prometheus 指标：`api_gateway_requests_total`、`api_gateway_request_latency_seconds`、`api_gateway_active_connections`、`api_gateway_rate_limit_hits_total`、`api_gateway_backend_health`
- 内嵌 Web Dashboard：`/dashboard` 展示限流配置、活跃 IP 数、当前窗口请求数、上游服务健康状态（5 秒自动刷新）

配置项（`api-gateway-config` ConfigMap）：
| 变量 | 默认 | 说明 |
|---|---|---|
| `BACKEND_HOST` | `frontend` | 上游 `frontend` 主机名（默认 namespace 内可解析） |
| `BACKEND_PORT` | `8080` | 上游端口 |
| `RATE_LIMIT_REQUESTS` | `100` | 窗口内允许的最大请求数 |
| `RATE_LIMIT_WINDOW` | `60` | 窗口秒数 |

关键接口：
- `GET /health`：健康检查 + 上游服务清单
- `GET /status`：当前 IP 的请求计数与活跃 IP 总数
- `GET /rate-limit-info`：限流策略
- `GET /metrics`：Prometheus 格式
- `GET /dashboard`：Web UI

启动与访问：
```bash
# 在 default namespace 部署
kubectl apply -f microservices/api-gateway/manifests/

# 本地访问
kubectl port-forward -n default svc/api-gateway 18080:8080 --address 0.0.0.0
# 浏览器打开 http://127.0.0.1:18080/dashboard
```

### Notification Service（`microservices/notification-service/`）

基于 Flask + WebSocket 的告警聚合与展示服务，部署在 `default` namespace，NodePort `30081`（通过 `kubectl port-forward` 暴露为 `127.0.0.1:18081`）。

功能：
- 告警接收：HTTP `POST /api/alerts` 接收 JSON 告警（SR-CNN / KPIRoot / 外部系统推送）
- 告警聚合：按 `service` + `severity` 维度去重计数
- 实时推送：通过 WebSocket `/ws/alerts` 主动推送给浏览器
- Web Dashboard：`/dashboard` 展示告警时间线、严重等级分布、按服务聚合的告警次数

数据结构（`POST /api/alerts` body）：
```json
{
  "id": "alert-001",
  "timestamp": "2026-06-13T18:30:00Z",
  "service": "productcatalogservice",
  "metric": "request_latency_p99",
  "severity": "critical",
  "score": 0.92,
  "message": "Anomaly detected by SR-CNN",
  "root_cause": "productcatalogservice",
  "labels": {"namespace": "online-boutique"}
}
```

关键接口：
- `GET  /api/alerts`：列出所有告警（分页 `?limit=50&offset=0`）
- `GET  /api/alerts/stats`：按 service / severity 聚合统计
- `POST /api/alerts`：注入新告警（`Content-Type: application/json`）
- `WS   /ws/alerts`：WebSocket 实时推送
- `GET  /dashboard`：Web UI

启动与访问：
```bash
kubectl apply -f microservices/notification-service/manifests/

kubectl port-forward -n default svc/notification-service 18081:8080 --address 0.0.0.0
# 浏览器打开 http://127.0.0.1:18081/dashboard
```

### 与原系统的集成

- API Gateway 作为统一入口承接 JMeter / Selenium 流量
- Notification Service 作为 SR-CNN 和 KPIRoot 输出的下游消费者
- Prometheus 同时抓取两个新服务的 `/metrics`，与 Online Boutique 原生服务的指标统一呈现

## 快速流程
1. 启动 Minikube。
2. 部署 Online Boutique。
3. 部署 Prometheus 和 Grafana。
4. 部署 ChaosMesh。
5. 部署 API Gateway 和 Notification Service，并使用 Selenium/JMeter 产生流量。
6. 使用 ChaosMesh 注入故障。
7. 使用 Prometheus 导出指标。
8. 运行 SR-CNN 进行异常检测。
9. 运行 KPIRoot 进行根因定位。
10. 检测结果通过 Notification Service Dashboard 实时呈现，Agent 自动完成注入、采集、检测和诊断。
详细命令见各目录下脚本。