@echo off
echo === Prometheus Connectivity Check ===
echo.

echo [1] Prometheus port-forward alive?
curl -s "http://localhost:9090/-/healthy" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Cannot connect to http://localhost:9090
    echo       Run: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
    goto :end
) else (
    echo [OK] Prometheus responding
)

echo.
echo [2] API working?
curl -s "http://localhost:9090/api/v1/query?query=up" | findstr /C:"success"
if errorlevel 1 (
    echo [FAIL] API not returning success
) else (
    echo [OK] API returning success
)

echo.
echo [3] Check container metrics existence (series count):
for %%m in (
    "container_cpu_usage_seconds_total"
    "container_memory_working_set_bytes"
    "container_memory_usage_bytes"
    "container_network_receive_bytes_total"
    "container_network_transmit_bytes_total"
    "kube_pod_container_status_restarts_total"
) do (
    for %%i in (%%m) do (
        curl -s "http://localhost:9090/api/v1/query?query=count(%%~m)" | findstr /C:"\"value\"" >nul 2>&1
        if errorlevel 1 (
            echo [---] %%~m ^<- NO DATA
        ) else (
            for /f "delims=" %%r in ('curl -s "http://localhost:9090/api/v1/query?query=count(%%~m)"') do (
                echo [??] %%~m: %%r
            )
        )
    )
)

echo.
echo [4] What metric names exist? (top-level label __name__)
curl -s "http://localhost:9090/api/v1/label/__name__/values" > "%TEMP%\prom_names.json" 2>&1
findstr /C:"error" "%TEMP%\prom_names.json" >nul 2>&1
if errorlevel 1 (
    echo [OK] Got metric names list
    echo Filtering network-related names...
    findstr /C:"network" "%TEMP%\prom_names.json"
) else (
    echo [FAIL] Could not get metric names
    type "%TEMP%\prom_names.json"
)

:end
echo.
echo Done.
