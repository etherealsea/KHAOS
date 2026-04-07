# KHAOS: Kinetic Hurst Adaptive Oscillator System

Core package for the PI-KAN based financial phase detection system.

## What lives here

- `模型定义/`: PI-KAN / multiscale architectures, attention blocks, RevIN
- `模型训练/`: generic trainer, loss definitions, checkpoint scoring
- `数据处理/`: A-share data discovery, normalization, dataset profiles, fetch helpers
- `核心引擎/`: differentiable physics features and indicator logic
- `工具箱/`: visualization and utility scripts
- `同花顺公式/`: THS formula files and proxy mapping helpers

## Current recommended entry pattern

The low-level trainer in [`模型训练/train.py`](模型训练/train.py) is reusable, but the practical workflow is driven by repository-level runners that assemble:

- data paths
- dataset / loss / score profiles
- smoke vs formal settings
- local artifact directories

For current A-share work, prefer these runners:

- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](../../../03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py`](../../../03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py)

## Key modules for new contributors

- Dataset weighting and A-share event labels:
  [`数据处理/ashare_dataset.py`](数据处理/ashare_dataset.py)
- A-share file discovery / BaoStock fetch / training-ready generation:
  [`数据处理/ashare_support.py`](数据处理/ashare_support.py)
- Generic trainer and checkpoint scoring:
  [`模型训练/train.py`](模型训练/train.py)
- Constraint-aware and profile-aware losses:
  [`模型训练/loss.py`](模型训练/loss.py)
- Active model definition:
  [`模型定义/kan.py`](模型定义/kan.py)

## Notes

- `旧代码归档/` is historical reference, not the default place for new work.
- Current GitHub workflow intentionally excludes local weights, logs, and full datasets.
- If you only have the GitHub repository, read the root-level handoff docs before diving into package internals.
