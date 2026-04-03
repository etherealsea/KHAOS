# Fix AMP NaN Training Spec

## Why
在进行近十年高频数据的混合精度（AMP）全量训练时，偶尔会出现张量维度崩溃或 `NaN/Inf` 梯度爆炸导致训练中断。用户要求不降低精度（继续保留混合精度训练以确保效率），因此需要从数值稳定性的角度优化前向传播和正则化计算。

## What Changes
- 修改 `kan.py` 中的正则化损失计算，将极小值 `1e-8` 替换为在 FP16 下安全的 `1e-4`，防止除零导致的 `NaN`。
- 修改 `revin.py`，在计算均值和方差等统计量时强制转换至 FP32 计算，避免在 FP16 动态范围内由于异常值引起的溢出（Overflow/Underflow）。
- 修改 `train.py`，在支持的设备上优先使用 `bfloat16` 替代 `float16`，并优化梯度检查逻辑，确保在进行梯度裁剪时能够安全跳过无效的 `NaN/Inf` 批次。

## Impact
- Affected specs: 训练稳定性与混合精度加速
- Affected code:
  - `Finance/02_核心代码/源代码/khaos/模型定义/kan.py`
  - `Finance/02_核心代码/源代码/khaos/模型定义/revin.py`
  - `Finance/02_核心代码/源代码/khaos/模型训练/train.py`

## ADDED Requirements
### Requirement: AMP Stability
系统应在保持混合精度（AMP）训练效率的同时，避免任何 `NaN/Inf` 引发的训练崩溃。

#### Scenario: Success case
- **WHEN** 模型进行高频数据的长时间序列前向传播与 Loss 计算时
- **THEN** 各网络层计算出的统计特征、激活值以及梯度均应在合法数值范围内，遇到潜在的下溢情况时能够被安全的 epsilon 值保护。

## MODIFIED Requirements
### Requirement: Numerical Precision in Normalization
- RevIN 模块在进行分布标准化（Normalization）和反标准化（Denormalization）时，内部统计量计算必须以 FP32 精度进行，以防数据极值导致 FP16 下的溢出。
