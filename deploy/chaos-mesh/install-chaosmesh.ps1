helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update

kubectl create namespace chaos-mesh --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh `
  --namespace chaos-mesh `
  --set chaosDaemon.runtime=containerd `
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock `
  --set dashboard.create=true

kubectl wait --for=condition=available --timeout=600s deployment/chaos-dashboard -n chaos-mesh

Write-Host "Chaos Dashboard: kubectl port-forward -n chaos-mesh svc/chaos-dashboard 2333:2333"
