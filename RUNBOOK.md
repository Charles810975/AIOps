# 运行指南

## 1. 环境要求

- Docker Desktop
- Minikube
- kubectl
- Helm
- Python 3.10+
- Chrome + ChromeDriver 或 Selenium Manager
- JMeter 5.6+

安装 Python 依赖：

```powershell
py -m pip install -r requirements.txt
```

## 2. 部署 Online Boutique

```powershell
.\deploy\online-boutique\deploy-online-boutique.ps1
```

查看访问地址：

```powershell
minikube -p online-boutique service frontend-external -n online-boutique --url
```

## 3. 部署 Prometheus 和 Grafana

```powershell
.\deploy\prometheus\install-monitoring.ps1
kubectl apply -f .\deploy\grafana\online-boutique-dashboard-configmap.yaml
```

启动端口转发：

```powershell
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
```

Grafana 登录：`admin / admin`。

## 4. 部署 ChaosMesh

```powershell
.\deploy\chaos-mesh\install-chaosmesh.ps1
```

注入故障示例：

```powershell
kubectl apply -f .\deploy\chaos-mesh\cartservice-cpu-stress.yaml
kubectl apply -f .\deploy\chaos-mesh\productcatalog-network-delay.yaml
kubectl apply -f .\deploy\chaos-mesh\checkout-pod-kill.yaml
```

删除故障：

```powershell
kubectl delete -f .\deploy\chaos-mesh\cartservice-cpu-stress.yaml
```

## 5. Selenium 功能测试

```powershell
py .\tests\selenium\test_online_boutique.py --url http://127.0.0.1:30080 --headless --output reports\selenium_results.csv
```

如果使用 Minikube service URL，将 `--url` 替换成实际 URL。

## 6. JMeter 性能测试

```powershell
jmeter -n -t .\tests\jmeter\online-boutique-load-test.jmx -l reports\online-boutique-results.jtl -e -o reports\jmeter-html
```

如端口不同，可在 JMeter 中修改 `HOST` 和 `PORT`。

## 7. 导出 Prometheus 指标

> **重要**：每次导出前，请确保 Prometheus 端口转发已就绪：
> ```powershell
> kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
> ```

### 正常数据采集（建议 30 分钟以上）

```powershell
py .\data-collection\export_prometheus_metrics.py --prometheus http://localhost:9090 --minutes 30 --label normal --output data-collection\normal_metrics.csv
```

> **注意**：如果某些指标采不到数据（如 `network_receive` / `network_transmit`），脚本会打印 `[WARN]` 并跳过。
> 请检查 Prometheus 中是否存在对应的指标（可用 Grafana Explore 查询）。采集完成后检查输出文件行数，正常应有 5 种指标（`cpu_usage`, `memory_usage`, `restart_count`, `network_receive`, `network_transmit`）。

### 故障数据采集（建议与正常阶段时长一致）

```powershell
py .\data-collection\export_prometheus_metrics.py --prometheus http://localhost:9090 --minutes 30 --label anomaly --fault-service cartservice --output data-collection\cart_cpu_anomaly.csv
```

### 合并数据（KPIRoot 使用 combined 数据效果最佳）

```powershell
py .\data-collection\merge_metrics.py --normal data-collection\normal_metrics.csv --anomaly data-collection\cart_cpu_anomaly.csv --output data-collection\cart_cpu_combined.csv --normalize-time
```

> `--normalize-time` 将 normal 和 anomaly 的时间戳对齐拼接，使时间序列连续。
> 如不使用该参数，则保留原始时间戳。

### 调试：仅采集指定指标

```powershell
py .\data-collection\export_prometheus_metrics.py --prometheus http://localhost:9090 --minutes 10 --label normal --metrics cpu_usage memory_usage --output data-collection\debug_metrics.csv
```

### 运行 KPIRoot（推荐使用 combined 数据）

```powershell
py .\algorithms\kpiroot\kpiroot_reproduction.py --input data-collection\cart_cpu_combined.csv --output-dir reports\kpiroot
```

> **为什么用 combined 而非 anomaly 数据？** KPIRoot 的 `build_target_signal()` 用全局 Z-score 均值构建异常信号。
> combined 数据同时包含正常基线期和异常期，Z-score 在异常期会显著升高，从而更好地识别根因。
> 仅有 anomaly 数据时，Z-score 归一化后各服务差异不够大，导致排名不够准确。

## 8. 运行 KDD19 SR-CNN 复现

```powershell
py .\algorithms\sr-cnn\sr_cnn_reproduction.py --input data-collection\cart_cpu_anomaly.csv --metric cpu_usage --output-dir reports\sr-cnn
```

## 9. 运行 ISSRE24 KPIRoot 复现

```powershell
py .\algorithms\kpiroot\kpiroot_reproduction.py --input data-collection\cart_cpu_anomaly.csv --output-dir reports\kpiroot
```

## 10. 一键智能运维 Agent

```powershell
py .\agent\ops_agent.py --fault cart-cpu --prometheus http://localhost:9090 --warmup 60 --collect-minutes 10 --output-dir reports\agent-cart-cpu
```

可选故障：

- `cart-cpu`
- `product-delay`
- `checkout-kill`
