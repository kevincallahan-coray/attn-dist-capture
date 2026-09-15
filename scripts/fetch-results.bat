@echo off
setlocal
REM Fetch summary tables and figures into .\results
REM Uses only cmd builtins plus certutil and tar (both ship with Windows 10+).
REM Run from the repo root.

if not exist k8s\job-publish-results.yaml (
  echo Run this from the repo root ^(the folder containing k8s\^).
  exit /b 1
)

kubectl delete job attn-publish-results --ignore-not-found >nul 2>&1
kubectl apply -f k8s\job-publish-results.yaml >nul
if errorlevel 1 goto fail

echo Waiting for the publish job...
kubectl wait --for=condition=complete job/attn-publish-results --timeout=900s
if errorlevel 1 goto fail

if exist attn-results.b64 del attn-results.b64
if exist attn-results.tar.gz del attn-results.tar.gz

REM backoffLimit is 0, so the job has exactly one pod and job/ is unambiguous.
kubectl logs job/attn-publish-results --tail=-1 > attn-results.b64
if errorlevel 1 goto fail

set SIZE=0
for %%A in (attn-results.b64) do set SIZE=%%~zA
if "%SIZE%"=="0" (
  echo Payload was empty.
  goto fail
)
echo Payload: %SIZE% bytes

certutil -decode attn-results.b64 attn-results.tar.gz >nul
if errorlevel 1 (
  echo certutil could not decode the payload. First lines received:
  more /e +0 attn-results.b64 | findstr /n /r "." | findstr /b "1: 2: 3:"
  goto fail
)

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
kubectl logs job/attn-publish-results --tail=40
exit /b 1
