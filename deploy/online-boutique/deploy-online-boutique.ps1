param(
    [string]$Profile = "online-boutique",
    [int]$Cpus = 4,
    [string]$Memory = "6144mb",
    [string]$KubernetesVersion = "v1.28.3",
    [switch]$UseChinaMirror,
    [switch]$EnableIngress
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

Write-Host "Starting Minikube profile: $Profile with CPU=$Cpus Memory=$Memory Kubernetes=$KubernetesVersion"
$StartArgs = @("start", "-p", $Profile, "--cpus=$Cpus", "--memory=$Memory", "--driver=docker", "--kubernetes-version=$KubernetesVersion")
if ($UseChinaMirror) {
    $StartArgs += "--image-mirror-country=cn"
}
Invoke-Step "Start Minikube" { minikube @StartArgs }

Write-Host "Enabling useful addons"
Invoke-Step "Enable metrics-server" { minikube -p $Profile addons enable metrics-server }
if ($EnableIngress) {
    Invoke-Step "Enable ingress" { minikube -p $Profile addons enable ingress }
} else {
    Write-Host "Skipping ingress addon. Online Boutique can be accessed by minikube service."
}

Write-Host "Creating namespace"
kubectl create namespace online-boutique --dry-run=client -o yaml | kubectl apply -f -
if ($LASTEXITCODE -ne 0) { throw "Step failed: Create namespace" }

Write-Host "Deploying Online Boutique"
Invoke-Step "Apply Online Boutique manifests" { kubectl apply -n online-boutique -f https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml }

Write-Host "Waiting for pods"
Invoke-Step "Wait deployments" { kubectl wait --for=condition=available --timeout=600s deployment --all -n online-boutique }

Write-Host "Deployment status"
kubectl get pods -n online-boutique
kubectl get svc -n online-boutique

Write-Host "Expose frontend with:"
Write-Host "minikube -p $Profile service frontend-external -n online-boutique"
