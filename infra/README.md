# FinRL Platform: k3s/ArgoRollouts Deployment

## Overview
This directory contains production-grade Kubernetes (k3s) manifests for deploying the FinRL platform with:
- Blue-green deployments via ArgoRollouts
- Horizontal Pod Autoscaler (HPA) for Celery workers
- Persistent storage for Redis and Postgres
- Multi-cluster (colo + AWS Chicago DR) support

## Components
- `k3s-api.yaml`: FastAPI backend (blue-green, ArgoRollouts)
- `k3s-worker.yaml`: Celery worker (blue-green, HPA)
- `k3s-redis.yaml`: Redis (persistent, single-node)
- `k3s-postgres.yaml`: Postgres (persistent, single-node)

## Blue-Green Deployment (ArgoRollouts)
- Each core service uses an Argo Rollout for zero-downtime upgrades.
- `activeService` and `previewService` allow traffic shifting and instant rollback.

## Horizontal Pod Autoscaler (HPA)
- Celery worker auto-scales from 1 to 10 pods if CPU > 70% for 3 minutes.

## Multi-Cluster/DR
- Deploy the same manifests to both colo and AWS Chicago clusters.
- Use ArgoCD or `kubectl` for sync.
- For DR, ensure persistent volumes are available in both clusters.

## Quick Start
```sh
# Apply manifests in order
kubectl apply -f k3s-redis.yaml
kubectl apply -f k3s-postgres.yaml
kubectl apply -f k3s-api.yaml
kubectl apply -f k3s-worker.yaml

# (Optional) Install ArgoRollouts controller
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# Monitor rollout status
kubectl argo rollouts get rollout finrl-backend
kubectl argo rollouts get rollout finrl-worker

# HPA status
kubectl get hpa
```

## Notes
- Set image tags as needed for production.
- For DR, use cloud block storage for PVCs.
- All manifests are production-ready and tested for k3s v1.27+. 