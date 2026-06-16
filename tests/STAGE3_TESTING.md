# 阶段三：Selenium 与 JMeter 测试运行说明

## 1. 前置条件

确保 Online Boutique 前端可访问。推荐使用端口转发：

```powershell
kubectl port-forward -n online-boutique svc/frontend-external 8080:80
```

浏览器打开：

```text
http://localhost:8080
```

## 2. Selenium 功能测试

### Chrome 无头模式

```powershell
py .\tests\selenium\test_online_boutique.py --url http://localhost:8080 --browser chrome --headless --output reports\selenium\chrome_results.csv
```

### Edge 无头模式

```powershell
py .\tests\selenium\test_online_boutique.py --url http://localhost:8080 --browser edge --headless --output reports\selenium\edge_results.csv
```

### 输出结果

```text
reports/selenium/chrome_results.csv
reports/selenium/edge_results.csv
reports/selenium/screenshots/*.png
```

Selenium 脚本覆盖：

1. 打开首页；
2. 浏览商品详情；
3. 加入购物车；
4. 打开购物车；
5. 尝试填写并提交结算流程。

CSV 中记录每一步是否成功、耗时和错误信息，截图用于报告展示。

## 3. JMeter 性能测试

确认本机已安装 JMeter，并且 `jmeter` 命令可用：

```powershell
jmeter -v
```

运行多负载压测：

```powershell
.\tests\jmeter\run_jmeter_loads.ps1 -HostName 127.0.0.1 -Port 8080 -Threads 10,30,50 -Ramp 20 -Loops 5
```

如 JMeter 没有加入 PATH，可以指定完整路径：

```powershell
.\tests\jmeter\run_jmeter_loads.ps1 -JMeter "D:\apache-jmeter-5.6.3\bin\jmeter.bat" -HostName 127.0.0.1 -Port 8080
```

### 输出结果

```text
reports/jmeter/users_10/results.jtl
reports/jmeter/users_10/html/index.html
reports/jmeter/users_30/results.jtl
reports/jmeter/users_30/html/index.html
reports/jmeter/users_50/results.jtl
reports/jmeter/users_50/html/index.html
reports/jmeter/summary.csv
```

`summary.csv` 包含：

- 样本数；
- 成功数；
- 失败数；
- 错误率；
- 平均响应时间；
- P90/P95/P99 响应时间；
- 吞吐量。

## 4. 报告展示建议

报告中建议展示：

1. Selenium 每一步测试结果表；
2. Selenium 页面截图；
3. JMeter 不同并发用户数下的响应时间对比；
4. JMeter HTML 报告截图；
5. 正常场景与故障场景下的性能对比。
