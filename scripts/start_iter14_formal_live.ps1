$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $repoRoot 'Finance\01_数据中心\03_研究数据\research_processed'
$runRoot = Join-Path $repoRoot 'Finance\02_核心代码\模型权重备份\iter14_multiasset'
$runId = '20260423_iter14_multiasset_formal_live'
$runDir = Join-Path $runRoot $runId
$logPath = Join-Path $runDir 'live_console.log'
$runnerPath = Join-Path $repoRoot 'scripts\run_iter14_multiasset_closed_loop.py'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
Set-Location -LiteralPath $repoRoot

Write-Host "repoRoot: $repoRoot"
Write-Host "dataDir : $dataDir"
Write-Host "runRoot : $runRoot"
Write-Host "runId   : $runId"
Write-Host "logPath : $logPath"
Write-Host ''
Write-Host '说明:'
Write-Host '- 该脚本前台运行，终端会直接显示训练输出与 epoch 进度。'
Write-Host '- 再次运行同一脚本会复用同一个 run_id，并按 resume_mode=auto 自动续训。'
Write-Host ''

& python $runnerPath `
    --phase formal `
    --data_dir $dataDir `
    --run_root $runRoot `
    --run_id $runId `
    --assets "BTCUSD,ETHUSD,ESUSD,SPXUSD,UDXUSD,WTIUSD,XAUUSD" `
    --timeframes "5m,15m,60m,240m" `
    --resume_mode auto 2>&1 |
    Tee-Object -FilePath $logPath

Write-Host ''
Write-Host "[process-exit] $LASTEXITCODE"
