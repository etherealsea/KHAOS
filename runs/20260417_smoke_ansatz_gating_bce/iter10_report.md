# iter10 多资产闭环报告（run_dir=/workspace/runs/20260417_smoke_ansatz_gating_bce）

- best_epoch: 1
- best_score: 0.2251
- epochs: 1
- per_asset_records: 1

## Epoch 概览
| epoch | val_loss | composite_score | public_violation | score_veto_passed | avg_signal_frequency |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.5790 | 0.2251 | 0.0000 | True | 0.0153 |

## Best Epoch：分周期汇总
| timeframe | sample_count | breakout_f1(avg) | reversion_f1(avg) | public_violation(avg) | signal_frequency(avg) |
| --- | --- | --- | --- | --- | --- |
| 60m | 1011 | 0.1081 | 0.0000 | 0.0000 | 0.0307 |

## Best Epoch：分资产×分周期明细
| asset | timeframe | breakout_f1 | reversion_f1 | breakout_precision | reversion_precision | public_violation | signal_frequency | thresholds_frozen | frozen_thresholds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000333 | 60m | 0.1081 | 0.0000 | 0.1429 | 0.0000 | 0.0000 | 0.0307 | True | {"breakout": 0.45139414234594866, "reversion": 0.45021594762802125} |

## 每 epoch 的冻结阈值（timeframe 级，来自 per_asset_metrics）
| epoch | timeframe | thresholds_frozen | frozen_thresholds |
| --- | --- | --- | --- |
| 1 | 60m | True | {"breakout": 0.45139414234594866, "reversion": 0.45021594762802125} |
