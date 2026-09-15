@echo off
setlocal enabledelayedexpansion
REM Fetch summary tables and figures into .\results
REM Uses only cmd builtins plus certutil and tar (both ship with Windows 10+).

kubectl delete job attn-publish-results --ignore-not-found >nul 2>&1
kubectl apply -f k8s\job-publish-results.yaml >nul
echo Waiting for the publish job...
kubectl wait --for=condition=complete job/attn-publish-results --timeout=900s
if errorlevel 1 goto fail

set POD=
for /f "tokens=*" %%p in ('kubectl get pods --selector=job-name=attn-publish-results --field-selector=status.phase=Succeeded -o name') do set POD=%%p
if "%POD%"=="" goto fail
echo Reading payload from %POD%

if exist attn-results.b64 del attn-results.b64
if exist attn-results.tar.gz del attn-results.tar.gz
kubectl logs %POD% > attn-results.b64
certutil -decode attn-results.b64 attn-results.tar.gz >nul
if errorlevel 1 (echo certutil could not decode the payload & goto fail)

if not exist results mkdir results
tar -xzf attn-results.tar.gz -C results --strip-components=1
if errorlevel 1 goto fail
del attn-results.b64 attn-results.tar.gz

echo.
type results\MANIFEST.txt
echo.
echo Unpacked into .\results
goto :eof

:fail
echo.
echo FAILED. Job output:
kubectl logs job/attn-publish-results
exit /b 1
