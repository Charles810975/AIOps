param(
    [string]$Profile = "online-boutique"
)

$ErrorActionPreference = "Stop"

$Images = @(
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/adservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/cartservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/checkoutservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/currencyservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/emailservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/frontend:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/loadgenerator:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/paymentservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/productcatalogservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/recommendationservice:v0.10.5",
    "us-central1-docker.pkg.dev/google-samples/microservices-demo/shippingservice:v0.10.5",
    "redis:alpine",
    "busybox:latest"
)

foreach ($Image in $Images) {
    Write-Host "`n==> Pulling $Image"
    docker pull $Image
    if ($LASTEXITCODE -ne 0) { throw "docker pull failed: $Image" }

    Write-Host "==> Loading $Image into minikube profile $Profile"
    minikube -p $Profile image load $Image
    if ($LASTEXITCODE -ne 0) { throw "minikube image load failed: $Image" }
}

Write-Host "`nAll images have been loaded into Minikube. Restarting deployments..."
kubectl rollout restart deployment -n online-boutique
kubectl get pods -n online-boutique
