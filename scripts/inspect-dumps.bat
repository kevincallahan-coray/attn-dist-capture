@echo off
kubectl delete job attn-inspect-dump --ignore-not-found
kubectl apply -f k8s\job-inspect-dump.yaml
kubectl wait --for=condition=ready pod -l job-name=attn-inspect-dump --timeout=300s
kubectl logs -f job/attn-inspect-dump
