$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$financeRoot = Join-Path $repoRoot 'Finance'
$dataDir = Join-Path $financeRoot '01_数据中心\03_研究数据\research_processed'

$runId = '20260423_iter12_final_20epoch'
$runRoot = Join-Path $repoRoot 'runs'
$runDir = Join-Path $runRoot $runId
$logPath = Join-Path $runDir 'live_console.log'
$trainPy = Join-Path $financeRoot '02_核心代码\源代码\khaos\模型训练\train.py'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
Set-Location -LiteralPath $repoRoot

Write-Host "repoRoot: $repoRoot"
Write-Host "dataDir: $dataDir"
Write-Host "runDir: $runDir"
Write-Host "logPath: $logPath"
Write-Host ""

python -u $trainPy `
    --data_dir $dataDir `
    --save_dir $runDir `
    --market "legacy_multiasset" `
    --assets "BTCUSD,ETHUSD,ESUSD,SPXUSD,EURUSD,UDXUSD,WTIUSD,XAUUSD" `
    --timeframes "5m,15m,60m,240m" `
    --epochs 20 `
    --batch_size 256 `
    --lr 7e-4 `
    --window_size 24 `
    --horizon 10 `
    --hidden_dim 80 `
    --layers 4 `
    --grid_size 12 `
    --arch_version "iterA4_multiscale" `
    --dataset_profile "shortT_discovery_guarded_v2" `
    --loss_profile "shortT_discovery_guarded_v2" `
    --constraint_profile "teacher_feasible_discovery_v1" `
    --score_profile "iter12_guarded_precision_first" `
    --score_timeframes "15m,60m,240m" `
    --aux_timeframes "5m" `
    --split_scheme "rolling_recent_v1" `
    --early_stop_patience 8 `
    --early_stop_min_delta 0.001 `
    --grad_clip 0.8 `
    --per_timeframe_train_cap "5m=2048,15m=4096,60m=6144,240m=4096" `
    --num_workers 0 `
    --prefetch_factor 2 `
    --non_deterministic `
    --kill_keep_signal_frequency_max 0.40 `
    --kill_keep_public_violation_rate_max 0.14 `
    --kill_keep_timeframe_60m_composite_min 0.0 `
    --kill_keep_breakout_signal_space_min 0.95 `
    --kill_keep_reversion_signal_space_min 0.70 `
    --public_violation_cap 0.05 `
    --signal_frequency_cap_ratio 0.70

Write-Host ""
Write-Host "[process-exit] $LASTEXITCODE"
