@echo off
REM Tears down every job/pod this repo creates. Leaves PVC data alone.
kubectl delete job attn-capture-smoke --ignore-not-found
kubectl delete job attn-capture-ruler-4k --ignore-not-found
kubectl delete job attn-capture-ruler-32k --ignore-not-found
kubectl delete job attn-inspect-dump --ignore-not-found
kubectl delete job attn-export-dump --ignore-not-found
kubectl get pods
