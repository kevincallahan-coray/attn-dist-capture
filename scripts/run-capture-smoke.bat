@echo off
setlocal
kubectl delete job attn-capture-smoke --ignore-not-found
kubectl apply -f k8s\job-capture-smoke.yaml
kubectl wait --for=condition=ready pod -l job-name=attn-capture-smoke --timeout=900s
kubectl logs -f job/attn-capture-smoke
