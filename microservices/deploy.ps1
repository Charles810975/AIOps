#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

$Manifest = Join-Path $PSScriptRoot "k8s-manifests.yaml"

Write-Host "==> Applying custom microservices manifests" -ForegroundColor Cyan
kubectl apply -f $Manifest

Write-Host "`n==> Waiting for deployments to become ready" -ForegroundColor Cyan
kubectl wait --for=condition=available --timeout=120s deployment/api-gateway
kubectl wait --for=condition=available --timeout=120s deployment/notification-service

Write-Host "`n==> Custom microservices status" -ForegroundColor Cyan
kubectl get deploy,svc,po -l 'tier=custom' -o wide
