# Get the current daemonset as JSON
$ds = kubectl get ds -n chaos-mesh chaos-daemon -o json | ConvertFrom-Json

# Fix the command: change runtime to docker
$ds.spec.template.spec.containers[0].command = @(
    "/usr/local/bin/chaos-daemon",
    "--runtime", "docker",
    "--http-port", "31766",
    "--grpc-port", "31767",
    "--pprof",
    "--ca", "/etc/chaos-daemon/cert/ca.crt",
    "--cert", "/etc/chaos-daemon/cert/tls.crt",
    "--key", "/etc/chaos-daemon/cert/tls.key",
    "--runtime-socket-path", "/var/run/docker.sock"
)

# Fix the hostPath volume: /run/containerd -> /var/run
$ds.spec.template.spec.volumes[0].hostPath.path = "/var/run"

# Remove conflicting fields
$ds.PSObject.Properties.Remove('status')
$ds.metadata.PSObject.Properties.Remove('resourceVersion')
$ds.metadata.PSObject.Properties.Remove('uid')
$ds.metadata.PSObject.Properties.Remove('generation')
if ($ds.metadata.annotations.'kubectl.kubernetes.io/restartedAt') {
    $ds.metadata.annotations.PSObject.Properties.Remove('kubectl.kubernetes.io/restartedAt')
}
if ($ds.metadata.annotations.'deprecated.daemonset.template.generation') {
    $ds.metadata.annotations.PSObject.Properties.Remove('deprecated.daemonset.template.generation')
}

# Write to file
$ds | ConvertTo-Json -Depth 20 | Out-File -FilePath "d:\???\???????\Final\chaos-daemon-ds-fixed.json" -Encoding utf8

Write-Host "Fixed JSON written"

# Apply it
kubectl apply -f "d:\???\???????\Final\chaos-daemon-ds-fixed.json"
