# 大作业实验报告框架

## 1. 项目背景

本实验围绕 Online Boutique 微服务系统，完成微服务部署、监控、测试、故障注入、异常检测与根因分析。项目目标是验证微服务系统在正常和故障场景下的可观测性与可维护性，并复现 KDD19 SR-CNN 与 ISSRE24 KPIRoot 的核心思想。

## 2. 微服务系统介绍

Online Boutique 是 Google 开源的云原生微服务示例系统，模拟电商购物场景，包含前端、商品目录、购物车、结算、支付、物流、推荐、广告、邮件和 Redis 等服务。

可展示内容：

- Kubernetes Pod 列表；
- Service 列表；
- 前端页面；
- 服务调用关系图。

## 3. 部署方案

### 3.1 Docker 与 Minikube 环境

说明 Docker、Minikube、kubectl、Helm 的版本。

### 3.2 Online Boutique 部署

展示部署命令和运行状态截图。

### 3.3 Prometheus 与 Grafana 部署

说明采集指标类型：CPU、内存、网络、重启次数、服务状态等。

## 4. 测试方案

### 4.1 Selenium 功能测试

测试步骤：

1. 打开首页；
2. 浏览商品详情；
3. 加入购物车；
4. 查看购物车。

记录页面加载时间和交互响应时间。

### 4.2 JMeter 性能测试

测试不同并发用户数下的平均响应时间、P95 延迟、吞吐量和错误率。

## 5. 故障注入与监控

使用 ChaosMesh 注入以下故障：

| 故障编号 | 服务 | 故障类型 | 预期影响 |
|---|---|---|---|
| F1 | cartservice | CPU 压力 | 购物车操作变慢 |
| F2 | productcatalogservice | 网络延迟 | 商品列表加载变慢 |
| F3 | checkoutservice | Pod Kill | 下单失败或重试 |

展示 Grafana 中故障前后的指标变化。

## 6. KDD19 SR-CNN 复现

### 6.1 论文核心思想

SR-CNN 将时间序列转换到频域，通过 Spectral Residual 突出异常点，再结合 CNN 学习异常模式。本项目复现其核心 Spectral Residual 异常检测方法，用于 Prometheus KPI 指标异常检测。

### 6.2 数据来源

使用 Prometheus 导出的 Online Boutique KPI 数据，包括 CPU、内存、网络等指标。

### 6.3 实验结果

展示：

- 原始指标曲线；
- SR 异常分数；
- 检测到的异常点；
- Precision、Recall、F1。

## 7. ISSRE24 KPIRoot 复现

### 7.1 论文核心思想

KPIRoot 结合相似性分析和因果分析定位根因 KPI。论文使用 SAX 表示提升效率，并对 KPI 与整体异常表现进行综合相关性评估。

### 7.2 本项目实现

本项目实现简化版 KPIRoot：

1. 构造全局异常目标信号；
2. 对每个 KPI 计算 Pearson、Spearman 与趋势相似性；
3. 计算滞后相关因果分数；
4. 加入异常波动强度；
5. 综合排序根因 KPI 和根因服务。

### 7.3 实验结果

展示：

- 根因 KPI Top 10；
- 根因服务 Top 10；
- Hit@K 结果；
- 与注入故障服务的一致性。

## 8. 智能运维 Agent 加分项

Agent 自动完成：

1. 注入故障；
2. 等待指标变化；
3. 导出 Prometheus 数据；
4. 运行 SR-CNN 异常检测；
5. 运行 KPIRoot 根因分析；
6. 输出诊断报告。

## 9. 总结与优化建议

总结 Online Boutique 的部署、监控和维护经验，分析 SR-CNN 与 KPIRoot 在实际微服务系统中的效果，并提出优化方向：

- 引入 OpenTelemetry Trace；
- 增加 HTTP 请求延迟指标；
- 使用更多故障类型；
- 使用多变量深度学习模型对比；
- 将 Agent 与自动恢复策略结合。
