$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python "Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py" --pipeline mainline_20260407 @args

