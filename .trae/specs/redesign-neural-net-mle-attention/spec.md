# 基于 MLE 与 Kimi 注意力残差机制的神经网络重构 Spec

## Why
当前的 PI-KAN 架构在处理时序数据时，直接将多时间步的物理特征展平（Flatten）输入给 KAN 网络。这种方式缺乏对长程时序依赖（Long-term Dependency）和历史上下文的动态权重分配。
同时，最大李雅普诺夫指数（MLE）作为衡量金融动力系统混沌程度和相变边缘的核心物理量，尚未在网络中得到显式应用。
结合最新的 Kimi 注意力残差机制（Gated/Scaled Attention Residuals），可以有效缓解深层网络在处理长序列时的梯度衰减和注意力坍塌问题。将该机制与 MLE 物理约束及 KAN 的非线性样条拟合能力相结合，能够大幅提升系统对金融时序相变（趋势爆发、均值回归、混沌震荡）的探测精度与鲁棒性。

## What Changes
- **引入可微的最大李雅普诺夫指数（MLE）代理**：在物理引擎层新增计算模块，提取时序数据的局部混沌程度作为新特征。
- **设计注意力残差模块（Attention Residual Block）**：基于 Kimi 最新论文思路，引入带有门控/缩放残差连接的多头自注意力机制（Multi-Head Self-Attention with Scaled Residuals），用于在时序维度上提取特征。
- **重构 PI-KAN 主干架构 (RA-PI-KAN)**：将网络结构升级为 `[输入序列] -> [注意力残差模块 (时序建模)] -> [时序池化/降维] -> [KAN 层 (非线性物理映射)] -> [预测输出]`。
- **新增 P6 物理损失约束**：在损失函数中增加基于 MLE 的物理正则化项，包括混沌-置信度一致性约束和相变爆发预警约束。
- **BREAKING**: 模型输入维度和内部传递形状将发生改变，由原本的直接展平变为保留 `(Batch, Sequence, Feature)` 形状经过注意力层后再送入 KAN 层。

## Impact
- Affected specs: 神经网络架构设计、物理特征提取逻辑、物理信息损失函数。
- Affected code: 
  - `Finance/02_核心代码/源代码/khaos/核心引擎/physics.py`
  - `Finance/02_核心代码/源代码/khaos/模型定义/kan.py` (可能新增 `attention.py`)
  - `Finance/02_核心代码/源代码/khaos/模型训练/loss.py`
  - `Finance/02_核心代码/源代码/khaos/模型训练/train.py`

## ADDED Requirements
### Requirement: 融合注意力残差的 KAN 架构 (RA-PI-KAN)
系统必须能够使用带有门控残差连接的自注意力机制处理时序特征序列，并将输出传递给 KAN 线性层进行最终的样条拟合。

#### Scenario: 时序特征前向传播
- **WHEN** 形状为 `(Batch, Sequence_Length, Features)` 的物理+价格特征输入模型
- **THEN** 注意力模块对序列进行上下文加权，残差连接防止退化，随后聚合为 `(Batch, Hidden_Dim)` 送入 KAN 网络并输出波动率变化预测。

### Requirement: MLE 物理约束损失
系统必须在训练时计算基于 MLE 代理的物理正则化损失。

#### Scenario: 高混沌状态下的惩罚
- **WHEN** 当前时间步的 MLE 代理值 $\lambda > 0$（高度混沌）且模型输出了高置信度（极低方差或极确定的预测）
- **THEN** 损失函数中 P6 项产生额外惩罚，迫使模型在混沌区域保持“不确定性”。

## MODIFIED Requirements
### Requirement: 物理特征提取 (PhysicsLayer)
在现有的 `[Hurst, 波动率, 预测残差, 排列熵]` 特征基础上，新增并输出平滑后的 `最大李雅普诺夫指数代理 (MLE)` 特征。

## REMOVED Requirements
### Requirement: 简单的时序展平 (Temporal Flattening)
**Reason**: 直接展平丢失了时间步之间的相对关系和长期依赖的动态权重。
**Migration**: 替换为通过 Attention 模块处理后提取上下文向量（如取最后一步的输出或做加权池化）。
