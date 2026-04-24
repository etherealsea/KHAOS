$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$financeRoot = Join-Path $repoRoot 'Finance'
$dataDir = Join-Path $financeRoot '01_数据中心\03_研究数据\research_processed'

$runId = '20260423_iter12_final_20epoch'
$runRoot = Join-Path $repoRoot 'runs'
$runDir = Join-Path $runRoot $runId
$logPath = Join-Path $runDir 'live_console.log'
$runnerPath = Join-Path $repoRoot 'scripts\run_iter12_multiasset_closed_loop.py'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
Set-Location -LiteralPath $repoRoot

Write-Host "repoRoot: $repoRoot"
Write-Host "dataDir : $dataDir"
Write-Host "runDir  : $runDir"
Write-Host "logPath : $logPath"
Write-Host ''

& python $runnerPath `
    --phase formal `
    --data_dir $dataDir `
    --run_root $runRoot `
    --run_id $runId `
    --timeframes "5m,15m,60m,240m" `
    --score_timeframes "15m,60m,240m" 2>&1 |
    Tee-Object -FilePath $logPath

Write-Host ''
Write-Host "[process-exit] $LASTEXITCODE"
