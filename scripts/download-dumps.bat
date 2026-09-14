@echo off
setlocal
REM Copies dumps off the PVC into .\dumps. No capture Job may be running,
REM because kevin-workspace is ReadWriteOnce.
set SRC=%1
if "%SRC%"=="" set SRC=ruler_4096
kubectl apply -f k8s\shell-attn-dist.yaml
kubectl wait --for=condition=ready pod/attn-dist-shell --timeout=300s
if not exist dumps mkdir dumps
kubectl cp attn-dist-shell:/work/attn-dist/%SRC% dumps\%SRC%
echo Copied to dumps\%SRC%
echo Remove the shell pod with: kubectl delete pod attn-dist-shell
