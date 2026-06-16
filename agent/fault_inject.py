# -*- coding: utf-8 -*-
"""
Inject a CPU stress fault into cartservice to make SR-CNN actually fire,
then run the AIOps agent to diagnose + self-heal.
"""
import os
import sys
import io
import time
import json
import subprocess

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "buffer"):
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiops_agent import AIOpsAgent

def run(cmd, timeout=30, input_=None):
    print(f"$ {cmd}")
    try:
        out = subprocess.run(cmd, shell=True, stderr=subprocess.STDOUT, timeout=timeout,
                              input=input_, stdout=subprocess.PIPE)
        txt = out.stdout.decode("utf-8", errors="replace")
        print(txt[:1500])
        if out.returncode != 0:
            print(f"[exit={out.returncode}]")
        return txt
    except Exception as e:
        print(f"[exception] {e}")
        return ""


def get_pod():
    return run("kubectl get pod -n online-boutique -l app=cartservice "
               "-o jsonpath={.items[0].metadata.name}", 10).strip() or "cartservice-77f8cfdff-b94mn"


def inject_cpu_stress(duration_sec=60):
    pod = get_pod()
    node = run(f"kubectl get pod -n online-boutique {pod} -o jsonpath={{.spec.nodeName}}",
               10).strip()
    print(f"\n>>> inject CPU stress on node {node} (where {pod} runs) for {duration_sec}s <<<\n")
    # use a separate stress pod on the same node to compete for CPU
    yaml = f"""
apiVersion: v1
kind: Pod
metadata:
  name: cpu-stress-injector
  namespace: online-boutique
spec:
  nodeName: {node}
  restartPolicy: Never
  containers:
  - name: stress
    image: busybox
    command: ["sh", "-c", "dd if=/dev/urandom | bzip2 -9 > /dev/null"]
  tolerations:
  - key: node-role.kubernetes.io/control-plane
    operator: Exists
"""
    run("kubectl apply -f -", 5, input_=yaml) if False else None
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml)
        path = f.name
    run(f"kubectl apply -f {path}", 15)
    time.sleep(8)  # let it ramp up


def clear_stress():
    run("kubectl delete pod cpu-stress-injector -n online-boutique --ignore-not-found", 15)


if __name__ == "__main__":
    if sys.argv[1:2] == ["inject"]:
        inject_cpu_stress(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif sys.argv[1:2] == ["clear"]:
        clear_stress()
    else:
        print("usage: fault_inject.py inject [sec] | clear")
