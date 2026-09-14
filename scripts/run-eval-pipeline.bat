@echo off
setlocal
REM Full pipeline: stage to CephFS, evaluate 4 tasks in parallel, aggregate.
kubectl delete job attn-stage-to-shared attn-eval-samplers attn-aggregate-results --ignore-not-found

kubectl apply -f k8s\job-stage-to-shared.yaml
kubectl wait --for=condition=complete job/attn-stage-to-shared --timeout=7200s
if errorlevel 1 (echo staging failed & kubectl logs job/attn-stage-to-shared & exit /b 1)

kubectl apply -f k8s\job-eval-samplers.yaml
kubectl wait --for=condition=complete job/attn-eval-samplers --timeout=10800s
if errorlevel 1 (echo eval failed & kubectl logs job/attn-eval-samplers --all-containers & exit /b 1)

kubectl apply -f k8s\job-aggregate-results.yaml
kubectl wait --for=condition=complete job/attn-aggregate-results --timeout=3600s
kubectl logs job/attn-aggregate-results
