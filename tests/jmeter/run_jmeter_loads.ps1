param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8080,
    [string]$JMeter = "jmeter",
    [string]$Template = "tests\jmeter\online-boutique-load-test-template.jmx",
    [string]$OutputDir = "reports\jmeter",
    [int[]]$Threads = @(10, 30, 50),
    [int]$Ramp = 20,
    [int]$Loops = 5
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($Thread in $Threads) {
    $RunName = "users_$Thread"
    $RunDir = Join-Path $OutputDir $RunName
    $Jtl = Join-Path $RunDir "results.jtl"
    $Html = Join-Path $RunDir "html"
    $Plan = Join-Path $RunDir "generated-plan.jmx"

    if (Test-Path $RunDir) {
        Remove-Item -Recurse -Force $RunDir
    }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $PlanContent = Get-Content $Template -Raw
    $PlanContent = $PlanContent.Replace("__HOST__", $HostName)
    $PlanContent = $PlanContent.Replace("__PORT__", [string]$Port)
    $PlanContent = $PlanContent.Replace("__THREADS__", [string]$Thread)
    $PlanContent = $PlanContent.Replace("__RAMP__", [string]$Ramp)
    $PlanContent = $PlanContent.Replace("__LOOPS__", [string]$Loops)
    Set-Content -Path $Plan -Value $PlanContent -Encoding UTF8

    Write-Host "`n==> Running JMeter load test: $Thread users"
    & $JMeter -n -t $Plan -l $Jtl
    if ($LASTEXITCODE -ne 0) {
        throw "JMeter failed for $Thread users"
    }

    if ((Test-Path $Jtl) -and ((Get-Item $Jtl).Length -gt 0)) {
        Write-Host "==> Generating JMeter HTML report for $Thread users"
        & $JMeter -g $Jtl -o $Html
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "JMeter HTML report generation failed for $Thread users, but JTL was generated."
        }
    } else {
        Write-Warning "No JTL samples generated for $Thread users. Check URL and generated plan: $Plan"
    }
}

Write-Host "`nAll JMeter runs completed. Summarizing..."
py .\tests\jmeter\summarize_jmeter.py --input-dir $OutputDir --output (Join-Path $OutputDir "summary.csv")
