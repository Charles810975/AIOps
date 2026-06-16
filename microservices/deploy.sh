#!/bin/bash
set -e
echo "==> Applying custom microservices manifests"
kubectl apply -f "$(dirname "$0")/k8s-manifests.yaml"
echo ""
echo "==> Waiting for deployments to become ready"
kubectl wait --for=condition=available --timeout=120s deployment/api-gateway
kubectl wait --for=condition=available --timeout=120s deployment/notification-service
echo ""
echo "==> Custom microservices status"
kubectl get deploy,svc,po -l 'tier=custom' -o wide
