param(
    [string]$Profile = "online-boutique"
)

$ErrorActionPreference = "Stop"

$Images = @(
    "ghcr.io/chaos-mesh/chaos-mesh:v2.8.3",
    "ghcr.io/chaos-mesh/chaos-daemon:v2.8.3",
    "ghcr.io/chaos-mesh/chaos-dashboard:v2.8.3",
    "ghcr.io/chaos-mesh/chaos-coredns:v0.2.8",
    "gcr.io/google-containers/pause:latest"
)

foreach ($Image in $Images) {
    Write-Host "`n==> Pulling $Image"
    docker pull $Image
    if ($LASTEXITCODE -ne 0) { throw "docker pull failed: $Image" }

    Write-Host "==> Loading $Image into minikube profile $Profile"
    minikube -p $Profile image load $Image
    if ($LASTEXITCODE -ne 0) { throw "minikube image load failed: $Image" }
}

Write-Host "`nRestarting ChaosMesh workloads..."
kubectl rollout restart deployment/chaos-controller-manager -n chaos-mesh
kubectl rollout restart deployment/chaos-dashboard -n chaos-mesh
kubectl rollout restart deployment/chaos-dns-server -n chaos-mesh
kubectl rollout restart daemonset/chaos-daemon -n chaos-mesh
kubectl get pods -n chaos-mesh
