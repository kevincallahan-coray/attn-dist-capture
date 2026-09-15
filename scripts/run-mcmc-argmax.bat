@echo off
setlocal
REM Best-start MCMC pass, then re-aggregate so the new plots include it.
kubectl delete job attn-eval-mcmc-argmax attn-aggregate-results --ignore-not-found
kubectl apply -f k8s\job-eval-mcmc-argmax.yaml
kubectl wait --for=condition=complete job/attn-eval-mcmc-argmax --timeout=14400s
if errorlevel 1 (echo eval failed & kubectl logs job/attn-eval-mcmc-argmax & exit /b 1)

kubectl apply -f k8s\job-aggregate-results.yaml
kubectl wait --for=condition=complete job/attn-aggregate-results --timeout=3600s
kubectl logs job/attn-aggregate-results
echo.
echo Now run scripts\fetch-results.bat to pull the new figures.
