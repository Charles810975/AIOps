# Build & Load Custom Microservices Images for Online Boutique

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Building Custom Microservices Docker Images" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$Images = @(
    @{ Name = "api-gateway"; Tag = "v1.0.0" },
    @{ Name = "notification-service"; Tag = "v1.0.0" }
)

foreach ($img in $Images) {
    $FullName = "$($img.Name):$($img.Tag)"
    $Dir = Join-Path $ProjectRoot "microservices\$($img.Name)"

    Write-Host "`n==> Building $FullName" -ForegroundColor Yellow
    docker build -t $FullName -f "$Dir\Dockerfile" $Dir
    if ($LASTEXITCODE -ne 0) { throw "docker build failed: $FullName" }

    docker tag $FullName "$($img.Name):latest"

    Write-Host "==> Loading $FullName into Minikube" -ForegroundColor Yellow
    minikube -p online-boutique image load $FullName
    if ($LASTEXITCODE -ne 0) { throw "minikube image load failed: $FullName" }

    minikube -p online-boutique image load "$($img.Name):latest"
    if ($LASTEXITCODE -ne 0) { throw "minikube image load failed: $($img.Name):latest" }
}

Write-Host "`n==> Images loaded in Minikube:" -ForegroundColor Cyan
minikube -p online-boutique image list | Select-String -Pattern "api-gateway|notification-service"

Write-Host "`nBuild and load complete!" -ForegroundColor Green
