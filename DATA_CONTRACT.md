# Data Contract

This repository intentionally excludes large datasets. This file describes the local data layout expected by the active A-share pipeline.

## Local Data Root

The current A-share runners expect data under:

- [`Finance/01_数据中心/03_研究数据`](Finance/01_数据中心/03_研究数据)

Important subdirectories:

- raw imports:
  `Finance/01_数据中心/03_研究数据/research_raw/ashare/imports`
- normalized raw:
  `Finance/01_数据中心/03_研究数据/research_raw/ashare/normalized`
- training-ready files:
  `Finance/01_数据中心/03_研究数据/research_processed/training_ready/ashare`

## File Naming

Training-ready files are discovered by suffix. Current canonical mapping:

- `5m -> *_5m.csv`
- `15m -> *_15m.csv`
- `60m -> *_1h.csv`
- `240m -> *_4h.csv`
- `1d -> *_1d.csv`

Example:

- `600036_5m.csv`
- `600036_1h.csv`
- `600036_1d.csv`

## Required Columns

Each normalized OHLCV file must contain:

- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`

Optional columns supported by the current pipeline:

- `amount`
- `turnover`
- `adj_factor`

The loader can normalize several Chinese and English aliases into these standard names.

## A-share Split Boundaries

Current default boundaries in the active A-share support module are:

- train end: `2023-12-31`
- validation end: `2024-12-31`
- test start: `2025-01-01`

These are defined in
[`Finance/02_核心代码/源代码/khaos/数据处理/ashare_support.py`](Finance/02_核心代码/源代码/khaos/数据处理/ashare_support.py)

## Current Active Timeframes

The repository currently uses two main timeframe sets:

- core set: `15m`, `60m`, `240m`, `1d`
- short-term enhanced set: `5m`, `15m`, `60m`, `240m`, `1d`

`iterA5` and teacher-first ablations both rely on the second set, but they do not use `5m` in exactly the same way.

## Data Acquisition Behavior

The active A-share runners can fetch missing public data through BaoStock when local CSV imports are absent or incomplete.

Relevant entry points:

- [`Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py`](Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

## Outputs That Stay Local

The following are expected to exist locally during real runs but are intentionally excluded from GitHub:

- coverage reports
- log files
- checkpoint files
- epoch / per-timeframe metric snapshots
- THS alignment outputs generated during evaluation

Typical local-only locations:

- `logs/`
- `Finance/02_核心代码/模型权重备份/`
- `Finance/models/`

## Practical Rule

If a collaborator only has GitHub access, they should understand the schema and directory expectations from this file, but they should not expect the actual market data to be present in the repository.
