# KHAOS-KAN 模型训练评估报告

**模型路径**: `d:\Finance\Finance\models\khaos_kan_best.pth`  
**评估时间**: 2025-12-13 13:43:58.603577

## 1. 总体表现摘要
- **平均 R2 Score**: -0.2967
- **平均 Correlation**: 0.0925
- **平均 MSE**: 0.7043

## 2. 分资产类别详细评估
| Category | Asset | Samples | MSE | R2 Score | MAE | Correlation |
| --- | --- | --- | --- | --- | --- | --- |
| Crypto | BTCUSD_1h.csv | 13137 | 0.4332 | -0.6514 | 0.5036 | -0.0137 |
| Forex | EURUSD_1h.csv | 9089 | 0.7110 | -0.0868 | 0.6809 | 0.1856 |
| Commodity | XAUUSD_1h.csv | 8626 | 0.7387 | -0.2023 | 0.6869 | 0.0463 |
| Index | SPXUSD_1h.csv | 8612 | 0.9342 | -0.2464 | 0.7858 | 0.1517 |

## 3. 结果分析
- **R2 Score**: 衡量模型对未来波动率变化（Vol Return）的解释程度。
- **Correlation**: 预测变化与真实变化的相关性。正相关表示能捕捉到波动率扩张/收缩的趋势。
- **Target**: Log(FutureVol / CurrentVol)。预测波动率的相对变化。

## 4. 后续建议
- 如果 **Correlation** > 0.1，说明模型具有预测波动率方向的能力（Gamma Trading）。
- 此版本已针对多资产（BTC, SPX, EUR）进行了自适应标准化（Target = Vol Change）。
