# KHAOS 中文使用与判读指南

## 1. 指标定位
- KHAOS 是“反转预警 + 潜在交易信号筛选器”。
- 先判断当前市场是否可交易，再判断方向，再看触发。

## 2. 快速上手
- `View Mode` 选 `Trader`：适合实盘，信息更精简。
- `View Mode` 选 `Research`：适合研究，显示背景、能量柱、触发线。
- `Language` 选 `中文`：面板与原因码切换为中文。

## 3. 你最需要看的 4 个字段
- `状态`：可交易/屏蔽，先看它；屏蔽时尽量不交易。
- `主线`：当前偏向，多为正偏多，负偏空。
- `置信度`：建议高于 55% 再重点关注。
- `原因`：解释为什么触发或为什么被屏蔽。

## 4. 信号判读顺序
1. 看 `状态` 是否为 `可交易`。  
2. 看 `原因` 是否是“门+驱+能:多/空”。  
3. 看 `置信度` 是否达到你的阈值。  
4. 再看图上的 `买/卖` 标签执行。  

## 5. 参数建议（默认起步）
- `Min Confidence`: 0.58
- `Min Main Bias`: 0.08
- `Signal Cooldown Bars`: 8
- `Strict Precision Mode`: 开启
- `Require Bar Close`: 开启

## 6. 解决“中线压缩看不清”
- `Main Scale Mode` 选 `自适应`（推荐）。
- 如果仍偏小，调大 `Adaptive Target Amp`（例如 1.2 -> 1.8）。
- 需要固定视觉时，改 `Main Scale Mode=固定`，再调 `Fixed Scale`。

## 7. 配色与透明度建议
- 日间图：`Color Theme=高对比`，`Panel Opacity=52~62`。
- 夜间图：`Color Theme=柔和`，`Panel Opacity=58~72`。
- 背景过重时，调高 `Background Opacity` 数值（更透明）。

## 8. 告警设置
- `KHAOS Buy Signal`：多头触发提醒。
- `KHAOS Sell Signal`：空头触发提醒。
- `KHAOS Regime Blocked`：状态屏蔽提醒。
