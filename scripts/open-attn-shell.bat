@echo off
kubectl apply -f k8s\shell-attn-dist.yaml
kubectl wait --for=condition=ready pod/attn-dist-shell --timeout=300s
kubectl exec -it attn-dist-shell -- bash
echo Remove it with: kubectl delete pod attn-dist-shell
