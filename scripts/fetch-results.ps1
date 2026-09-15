# Runs the publish job, decodes its log, and unpacks the tables and figures
# into .\results locally.
$ErrorActionPreference = "Stop"

kubectl delete job attn-publish-results --ignore-not-found | Out-Null
kubectl apply -f k8s\job-publish-results.yaml | Out-Null
Write-Host "waiting for publish job..."
kubectl wait --for=condition=complete job/attn-publish-results --timeout=900s

$log = kubectl logs job/attn-publish-results
$start = $log | Select-String -Pattern "BEGIN ATTN RESULTS" | Select-Object -First 1
$end   = $log | Select-String -Pattern "END ATTN RESULTS"   | Select-Object -First 1
if (-not $start -or -not $end) {
    Write-Host "no payload in log:"; $log; exit 1
}
$b64 = ($log[($start.LineNumber)..($end.LineNumber - 2)]) -join ""
[IO.File]::WriteAllBytes("$PWD\attn-results.tar.gz",
                         [Convert]::FromBase64String($b64))

New-Item -ItemType Directory -Force -Path .\results | Out-Null
tar -xzf attn-results.tar.gz -C .\results --strip-components=1
Remove-Item attn-results.tar.gz
Write-Host "`nunpacked into .\results:"
Get-ChildItem -Recurse .\results | Select-Object FullName, Length | Format-Table
