import json
import subprocess

# Get the current daemonset
result = subprocess.run(
    ["kubectl", "get", "ds", "-n", "chaos-mesh", "chaos-daemon", "-o", "json"],
    capture_output=True, text=True, encoding="utf-8"
)
ds = json.loads(result.stdout)

# Fix the command: change runtime to docker
ds["spec"]["template"]["spec"]["containers"][0]["command"] = [
    "/usr/local/bin/chaos-daemon",
    "--runtime", "docker",
    "--http-port", "31766",
    "--grpc-port", "31767",
    "--pprof",
    "--ca", "/etc/chaos-daemon/cert/ca.crt",
    "--cert", "/etc/chaos-daemon/cert/tls.crt",
    "--key", "/etc/chaos-daemon/cert/tls.key",
    "--runtime-socket-path", "/var/run/docker.sock"
]

# Fix the hostPath volume: /run/containerd -> /var/run (to include docker.sock)
ds["spec"]["template"]["spec"]["volumes"][0]["hostPath"]["path"] = "/var/run"

# Remove fields that cause conflicts on apply
for key in ["status", "metadata.resourceVersion", "metadata.uid", "metadata.generation"]:
    parts = key.split(".")
    obj = ds
    for p in parts[:-1]:
        obj = obj[p]
    del obj[parts[-1]]

# Remove problematic annotations
if "kubectl.kubernetes.io/restartedAt" in ds["metadata"]["annotations"]:
    del ds["metadata"]["annotations"]["kubectl.kubernetes.io/restartedAt"]
if "deprecated.daemon" in str(ds["metadata"].get("annotations", {})):
    del ds["metadata"]["annotations"]["deprecated.daemonset.template.generation"]

# Write fixed JSON
with open(r"d:\???\???????\Final\chaos-daemon-ds-fixed.json", "w", encoding="utf-8") as f:
    json.dump(ds, f, indent=2, ensure_ascii=False)

print("Fixed JSON written")

# Apply it
result2 = subprocess.run(
    ["kubectl", "apply", "-f", r"d:\???\???????\Final\chaos-daemon-ds-fixed.json"],
    capture_output=True, text=True, encoding="utf-8"
)
print("STDOUT:", result2.stdout)
print("STDERR:", result2.stderr)
print("Return code:", result2.returncode)
