#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py --pipeline mainline_20260407 "$@"

