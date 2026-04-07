# AI Handoff

This file is for a new AI collaborator who only has access to the GitHub repository.

## Read Order

1. [`README.md`](README.md)
2. [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
3. [`DATA_CONTRACT.md`](DATA_CONTRACT.md)
4. [`Finance/03_实验与验证/脚本/README.md`](Finance/03_实验与验证/脚本/README.md)
5. [`Finance/02_核心代码/源代码/khaos/README.md`](Finance/02_核心代码/源代码/khaos/README.md)

## What This Project Is

- A PI-KAN / KHAOS research codebase for financial phase transition detection
- Current practical focus is A-share data
- Outputs must remain compatible with a THS-style signal interface

## Current Reality

- The repository contains code, docs, and runner definitions
- The repository does not contain full datasets, logs, or trained checkpoints
- Therefore, code inspection and documentation reading come first; local reproduction comes second

## Current Mainline

Treat the current active direction as:

- `teacher-first`
- `shortT balanced`
- `iterA4_multiscale` stability + improved diagnostics + constraint-aware training

The most important active runner is:

- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

The most important legacy-but-still-relevant formal runner is:

- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py)

## Do Not Misread These Folders

- [`Finance/03_实验与验证/脚本/测试与临时脚本`](Finance/03_实验与验证/脚本/测试与临时脚本) contains active runners, not just throwaway experiments.
- [`Finance/02_核心代码/旧代码归档`](Finance/02_核心代码/旧代码归档) is historical reference, not the first place to edit.
- [`Finance/02_核心代码/模型权重备份`](Finance/02_核心代码/模型权重备份) is a local artifact area and should not be treated as source-of-truth code.

## Safe First Tasks For A New AI

1. Clarify docs and script intent without changing training behavior.
2. Trace dataset / loss / score profiles through:
   [`Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py`](Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py)
   [`Finance/02_核心代码/源代码/khaos/模型训练/loss.py`](Finance/02_核心代码/源代码/khaos/模型训练/loss.py)
   [`Finance/02_核心代码/源代码/khaos/模型训练/train.py`](Finance/02_核心代码/源代码/khaos/模型训练/train.py)
3. Improve reproducibility and config clarity before attempting architectural rewrites.
4. Keep THS compatibility as a hard constraint when proposing model changes.

## Things You Should Not Assume

- Do not assume local checkpoints exist.
- Do not assume the best model is `iterA5`.
- Do not assume `5m` always helps.
- Do not assume the GitHub repository contains enough evidence to promote a new checkpoint without regenerating local metrics.

## Promotion Logic

Current repository-level promotion logic is anchored to:

- `overall_model.composite >= 0.4187`
- `calibrated_ths.test_objective >= 0.4448`
- `60m composite >= 0.4032`

If a proposed change cannot plausibly improve both model-side quality and THS-aligned behavior, it is probably not a promotion candidate.

## Where To Look For Context

- Latest development conclusions:
  [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md)
  [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md)
- Core source package:
  [`Finance/02_核心代码/源代码/khaos`](Finance/02_核心代码/源代码/khaos)
- Script map:
  [`Finance/03_实验与验证/脚本/README.md`](Finance/03_实验与验证/脚本/README.md)

## Bottom Line

If you only have GitHub:

- treat documentation as the current memory of the project
- treat runner definitions as the operational truth
- treat missing logs and weights as expected, not as repository damage
