# Prometheus connectivity check
$ErrorActionPreference = "Continue"

Write-Host "=== 1. Port forwarding check ==="
$response = try { Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -TimeoutSec 5 -UseBasicParsing } catch { $null }
if ($response) {
    Write-Host "[OK] Prometheus port-forward is active (status: $($response.StatusCode))"
} else {
    Write-Host "[FAIL] Cannot connect to http://localhost:9090"
    Write-Host "       Run: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
}

Write-Host ""
Write-Host "=== 2. Check Prometheus API ==="
$api = try { Invoke-RestMethod -Uri "http://localhost:9090/api/v1/query?query=up" -TimeoutSec 10 } catch { $null }
if ($api) {
    if ($api.status -eq "success") {
        Write-Host "[OK] Prometheus API responding"
        Write-Host "    Targets up: $($api.data.result.Count)"
    } else {
        Write-Host "[FAIL] Prometheus API error: $($api | ConvertTo-Json -Compress)"
    }
} else {
    Write-Host "[FAIL] Cannot reach Prometheus API"
}

Write-Host ""
Write-Host "=== 3. Check which container metrics exist ==="
$metrics = @(
    "container_cpu_usage_seconds_total",
    "container_memory_working_set_bytes",
    "container_memory_usage_bytes",
    "container_network_receive_bytes_total",
    "container_network_transmit_bytes_total",
    "kube_pod_container_status_restarts_total"
)

foreach ($m in $metrics) {
    $result = try {
        $r = Invoke-RestMethod -Uri "http://localhost:9090/api/v1/query?query=count($m)" -TimeoutSec 10
        if ($r.status -eq "success") { $r.data.result[0].value[1] } else { "ERROR" }
    } catch { "ERROR" }
    if ($result -ne "ERROR") {
        Write-Host "[OK]  $m  ->  series count: $result"
    } else {
        Write-Host "[---] $m  ->  no data"
    }
}

Write-Host ""
Write-Host "=== 4. Check namespace filter on network metrics ==="
$net_test = try {
    $r = Invoke-RestMethod -Uri 'http://localhost:9090/api/v1/query?query=container_network_receive_bytes_total' -TimeoutSec 10
    if ($r.status -eq "success") { $r.data.result.Count } else { 0 }
} catch { -1 }

if ($net_test -gt 0) {
    Write-Host "[OK] container_network_receive_bytes_total exists ($net_test series)"
    Write-Host "     Sample labels:"
    $sample = try {
        Invoke-RestMethod -Uri 'http://localhost:9090/api/v1/query?query=container_network_receive_bytes_total&limit=3' -TimeoutSec 10
    } catch { $null }
    if ($sample -and $sample.status -eq "success") {
        $sample.data.result | ForEach-Object {
            Write-Host "       $($_.metric | ConvertTo-Json -Compress)"
        }
    }
} elseif ($net_test -eq 0) {
    Write-Host "[---] container_network_receive_bytes_total has 0 series"
} else {
    Write-Host "[FAIL] Cannot query container_network_receive_bytes_total"
}

Write-Host ""
Write-Host "=== 5. Check minikube / node-level network metrics ==="
$node_net = try {
    $r = Invoke-RestMethod -Uri 'http://localhost:9090/api/v1/query?query=node_network_receive_bytes_total' -TimeoutSec 10
    if ($r.status -eq "success") { $r.data.result.Count } else { 0 }
} catch { -1 }
if ($node_net -gt 0) {
    Write-Host "[OK] node_network_receive_bytes_total exists ($node_net series)"
} elseif ($node_net -eq 0) {
    Write-Host "[---] node_network_receive_bytes_total has 0 series"
} else {
    Write-Host "[FAIL] Cannot query node_network_receive_bytes_total"
}
