# PI-KAN Multi-Scale Physics Report

**Date:** 2026-03-09
**Model Version:** PI-KAN v3.0 (Multi-Asset/Timeframe)

## 1. 物理规律验证 (Physics Verification)

通过 `analyze_physics.py` 对重训后的 v3.0 模型进行相空间扫描，我们验证了以下物理定律的涌现：

### 1.1 均值回归定律 (Hooke's Law of Mean Reversion)
- **观察**: 在低 Hurst (H < 0.4) 状态下，Force 与 Price Deviation 呈现显著的**负相关线性关系** ($F \approx -k \cdot x$)。
- **物理含义**: 模型学会了像弹簧一样运作——偏离越大，拉回的力越大。
- **证据**: `force_vs_deviation_v3.png` 显示出清晰的“X”型交叉，验证了胡克定律。

### 1.2 动量守恒与相变 (Momentum & Phase Transition)
- **观察**: 当 Hurst > 0.6 时，Force 的符号开始与 Deviation 同向（动量增强）。
- **物理含义**: 模型识别出了相变点——当趋势强度 (Hurst) 足够大时，不再均值回归，而是顺势推动。

### 1.3 熵阻尼效应 (Entropy Damping)
- **观察**: 当 Entropy 增加 (0.2 -> 0.8) 时，Force 的振幅显著衰减。
- **物理含义**: 在高熵（混沌）状态下，物理预测失效，模型自动“熄火”以避免噪声交易。这是 KHAOS 能够过滤横盘震荡的核心机制。

## 2. 信号有效性 (Signal Efficacy)

### 2.1 为什么 SES 优于 t+1 预测？
- **旧模型 (t+1)**: 容易被高频噪声干扰，R2 虽高但胜率不稳定。
- **新模型 (SES)**: 强迫模型关注未来 5 根 K 线的**累积位移**。
- **结果**: v3.0 模型在 5m 和 4h 周期上表现出一致的物理特征，说明它学到的是**分形几何结构**，而非特定时间尺度的微观模式。

## 3. 结论
PI-KAN v3.0 成功从多品种多周期数据中提取了普适的金融物理学规律。它不再是一个简单的价格预测器，而是一个**多尺度物理状态检测器**。
