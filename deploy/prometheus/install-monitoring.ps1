param(
    [string]$Profile = "online-boutique"
)

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack `
  --namespace monitoring `
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false `
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false `
  --set grafana.adminPassword=admin `
  --set grafana.service.type=NodePort

kubectl wait --for=condition=available --timeout=600s deployment/kube-prometheus-stack-grafana -n monitoring
kubectl wait --for=condition=available --timeout=600s deployment/kube-prometheus-stack-operator -n monitoring

Write-Host "Prometheus port-forward: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
Write-Host "Grafana port-forward: kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80"
Write-Host "Grafana login: admin / admin"
