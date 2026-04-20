# iter10 多资产闭环报告（run_dir=/workspace/runs/20260416_iter10_multiasset_113053）

- best_epoch: 1
- best_score: -839.1325
- epochs: 2
- per_asset_records: 18

## Epoch 概览
| epoch | val_loss | composite_score | public_violation | score_veto_passed | avg_signal_frequency |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.6804 | -839.1325 | 0.9914 | False | 0.0060 |
| 2 | 7.6479 | -839.1356 | 0.9914 | False | 0.0065 |

## Best Epoch：分周期汇总
| timeframe | sample_count | breakout_f1(avg) | reversion_f1(avg) | public_violation(avg) | signal_frequency(avg) |
| --- | --- | --- | --- | --- | --- |
| 15m | 11988 | 0.0032 | 0.0072 | 0.9893 | 0.0051 |
| 1d | 1188 | 0.0000 | 0.0000 | 0.9899 | 0.0227 |
| 60m | 3033 | 0.0000 | 0.0091 | 1.0000 | 0.0082 |

## Best Epoch：分资产×分周期明细
| asset | timeframe | breakout_f1 | reversion_f1 | breakout_precision | reversion_precision | public_violation | signal_frequency | thresholds_frozen | frozen_thresholds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000333 | 15m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0048 | True | {"breakout": 0.7468628871440888, "reversion": 0.2587502238154411} |
| 000333 | 1d | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0429 | True | {"breakout": 0.7480934605002403, "reversion": 0.25743874582377346} |
| 000333 | 60m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0069 | True | {"breakout": 0.7480408585071564, "reversion": 0.2594827648997307} |
| 000651 | 15m | 0.0096 | 0.0000 | 0.2000 | 0.0000 | 1.0000 | 0.0045 | True | {"breakout": 0.7468628871440888, "reversion": 0.2587502238154411} |
| 000651 | 1d | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0126 | True | {"breakout": 0.7480934605002403, "reversion": 0.25743874582377346} |
| 000651 | 60m | 0.0000 | 0.0274 | 0.0000 | 0.1667 | 1.0000 | 0.0119 | True | {"breakout": 0.7480408585071564, "reversion": 0.2594827648997307} |
| 000725 | 15m | 0.0000 | 0.0216 | 0.0000 | 0.1333 | 0.9680 | 0.0060 | True | {"breakout": 0.7468628871440888, "reversion": 0.2587502238154411} |
| 000725 | 1d | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9697 | 0.0126 | True | {"breakout": 0.7480934605002403, "reversion": 0.25743874582377346} |
| 000725 | 60m | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0059 | True | {"breakout": 0.7480408585071564, "reversion": 0.2594827648997307} |

## 每 epoch 的冻结阈值（timeframe 级，来自 per_asset_metrics）
| epoch | timeframe | thresholds_frozen | frozen_thresholds |
| --- | --- | --- | --- |
| 1 | 15m | True | {"breakout": 0.7468628871440888, "reversion": 0.2587502238154411} |
| 1 | 1d | True | {"breakout": 0.7480934605002403, "reversion": 0.25743874582377346} |
| 1 | 60m | True | {"breakout": 0.7480408585071564, "reversion": 0.2594827648997307} |
| 2 | 15m | True | {"breakout": 0.6317874670028687, "reversion": 0.524596315026283} |
| 2 | 1d | True | {"breakout": 0.6420365840196609, "reversion": 0.589396768212318} |
| 2 | 60m | True | {"breakout": 0.6415081959962845, "reversion": 0.5579913473129273} |
