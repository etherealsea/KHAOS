# Project Status

Last updated: `2026-04-07`

This file is the GitHub-side status snapshot for collaborators who do not have access to local logs, weights, or datasets.

## Snapshot

- Research target: PI-KAN / KHAOS based financial phase transition detection, with A-share deployment constraints and THS-compatible outputs.
- Current stable promoted baseline: `iterA3_ashare`
- `iterA5_ashare` is an important engineering iteration, but as of `2026-04-07` it should not be treated as a promoted successor.
- Current recommended follow-up direction: `teacher-first + shortT balanced` experiments defined in
  [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

## Confirmed Baseline Numbers

Based on the development notes dated `2026-04-07`:

- `iterA3 overall_model.composite = 0.4187`
- `iterA4 overall_model.composite = 0.3906`
- `iterA5` mid-run best was inferred at about `0.3766`

Interpretation:

- `iterA3` remains the strongest confirmed public baseline in the repository narrative.
- `iterA5` improved observability and diagnostics, but did not yet justify promotion.

## What iterA5 Added

- More detailed per-timeframe logging
- Gate / constraint diagnostics during training
- Explicit `5m` integration into the A-share pipeline
- Better separation of dataset profile, loss profile, and score profile choices

These improvements matter even though `iterA5` itself is not the promoted winner.

## Current Mainline Hypothesis

The repository currently points toward this hypothesis:

- keep the more stable `iterA4_multiscale` backbone
- preserve the new diagnostics and teacher-feasible constraints
- treat `5m` as a controlled short-term signal source instead of a blind overall gain source
- optimize `breakout` and `reversion` together instead of overfitting to breakout-only improvements

This direction is expressed through:

- `shortT_balanced_v1`
- `shortT_balanced_v2`

inside the ablation runner.

## Promotion Gates

The currently documented promotion gates are:

- `overall_model.composite >= 0.4187`
- `calibrated_ths.test_objective >= 0.4448`
- `60m composite >= 0.4032`

For teacher-first smoke experiments, there are also explicit guardrails for:

- `direction_macro_f1`
- `public_below_directional_violation_rate`

See the runner for exact thresholds.

## Recommended Next Actions

1. Run or review `shortT_balanced_v1` and `shortT_balanced_v2` smoke experiments first.
2. Compare per-timeframe metrics, especially `60m`, against the `iterA3` and `iterA4` narrative baseline.
3. Keep THS proxy validation in the loop; do not treat teacher-side score alone as enough for promotion.
4. Avoid spending time on deeper `iterA5` epoch accumulation unless a targeted ablation suggests a recoverable issue.

## Primary Files To Read Next

- Root overview:
  [`README.md`](README.md)
- AI-oriented handoff:
  [`AI_HANDOFF.md`](AI_HANDOFF.md)
- Data layout contract:
  [`DATA_CONTRACT.md`](DATA_CONTRACT.md)
- Script map:
  [`Finance/03_实验与验证/脚本/README.md`](Finance/03_实验与验证/脚本/README.md)
- Latest development notes:
  [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md)
  [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md)

## Important Limitation

This repository intentionally excludes:

- the full A-share dataset
- local training logs
- weight checkpoints
- most experiment artifacts

So any collaborator or AI reading GitHub alone must treat this file and the development logs as the canonical project snapshot.
