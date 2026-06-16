#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Building Custom Microservices Docker Images"
echo "========================================="

cd "$PROJECT_ROOT/microservices"

echo ""
echo "==> Building api-gateway:v1.0.0"
docker build -t api-gateway:v1.0.0 -f api-gateway/Dockerfile api-gateway/
docker tag api-gateway:v1.0.0 api-gateway:latest

echo ""
echo "==> Building notification-service:v1.0.0"
docker build -t notification-service:v1.0.0 -f notification-service/Dockerfile notification-service/
docker tag notification-service:v1.0.0 notification-service:latest

echo ""
echo "==> Loading images into Minikube"
minikube -p online-boutique image load api-gateway:v1.0.0
minikube -p online-boutique image load api-gateway:latest
minikube -p online-boutique image load notification-service:v1.0.0
minikube -p online-boutique image load notification-service:latest

echo ""
echo "==> Images in Minikube:"
minikube -p online-boutique image list | grep -E "api-gateway|notification-service"

echo ""
echo "Build and load complete!"
