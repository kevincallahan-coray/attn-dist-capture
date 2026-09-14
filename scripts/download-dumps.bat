@echo off
setlocal
REM Pull a dump tarball off the PVC. Requires the export job to have run and
REM a pod that currently mounts kevin-workspace (RWO: one pod at a time).
REM Pass the running pod name as the first argument.
set POD=%1
set SRC=%2
if "%SRC%"=="" set SRC=ruler_4096
if "%POD%"=="" echo Usage: download-dumps.bat POD_NAME [DUMP_NAME] && exit /b 1
if not exist dumps mkdir dumps
kubectl cp %POD%:/work/attn-dist/%SRC%.tar.gz dumps\%SRC%.tar.gz
echo Wrote dumps\%SRC%.tar.gz
