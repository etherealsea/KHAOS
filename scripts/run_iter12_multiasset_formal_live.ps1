$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$financeRoot = Join-Path $repoRoot 'Finance'
$dataDir = Get-ChildItem -LiteralPath $financeRoot -Directory |
    Where-Object { $_.Name -like '01_*' } |
    Select-Object -First 1 |
    ForEach-Object {
        Get-ChildItem -LiteralPath $_.FullName -Directory |
            Where-Object { $_.Name -like '03_*' } |
            Select-Object -First 1 |
            ForEach-Object { Join-Path $_.FullName 'research_processed' }
    }

if (-not $dataDir -or -not (Test-Path -LiteralPath $dataDir)) {
    throw "Failed to resolve research_processed under $financeRoot"
}

$runId = '20260421_iter12_multiasset_formal'
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

& python $runnerPath --phase formal --data_dir $dataDir --run_root $runRoot --run_id $runId 2>&1 |
    Tee-Object -FilePath $logPath

Write-Host ''
Write-Host "[process-exit] $LASTEXITCODE"
