@echo off
setlocal
kubectl delete job attn-capture-ruler-4k --ignore-not-found
kubectl apply -f k8s\job-capture-ruler-4k.yaml
kubectl wait --for=condition=ready pod -l job-name=attn-capture-ruler-4k --timeout=900s
kubectl logs -f job/attn-capture-ruler-4k
