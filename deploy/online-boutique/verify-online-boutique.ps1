param(
    [string]$Profile = "online-boutique"
)

kubectl get pods -n online-boutique
kubectl get svc -n online-boutique
kubectl top pods -n online-boutique

Write-Host "Frontend URL:"
minikube -p $Profile service frontend-external -n online-boutique --url
