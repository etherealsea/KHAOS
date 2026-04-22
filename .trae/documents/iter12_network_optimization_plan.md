# Iter12 神经网络深度优化方案：基于数学特性的损失函数重构

## 1. 严谨的数学与科学分析

当前 KHAOS_KAN 模型在本地训练中表现出“高准确率（95%），但极低召回率（<1%）与极低信号频率（0.5%）”，其本质是模型陷入了**极端类别不平衡（Class Imbalance）下的局部最优解（Trivial Solution）**。

### 1.1 BCE Loss 在 4% vs 96% 不平衡下的梯度主导效应
二元交叉熵（BCE）的损失函数对 Logits $x$ 的梯度为：$\frac{\partial L}{\partial x} = p - y$。
- 在 4% 的正样本（事件区，$y=1$）上，梯度为 $p - 1$。
- 在 96% 的负样本（无事件区，$y=0$）上，梯度为 $p$。

由于负样本数量是正样本的 24 倍，在每一个 Batch 的反向传播中，即使模型对负样本的预测概率 $p$ 已经很小（如 0.1），累积的负向梯度（$24 \times 0.1 = 2.4$）依然会压倒正样本传来的微弱正向梯度（$1 \times (0.1 - 1) = -0.9$）。这迫使模型在所有区域都尽可能输出 0，从而导致召回率和信号频率锐减。

### 1.2 辅助任务梯度喧宾夺主 (Gradient Domination)
在训练日志中，主损失（BCE）为 `0.8729`，而辅助损失（Aux Loss，预测连续值的 Smooth L1）高达 `2.5902`。
从数学优化角度看，多任务学习中共享层的参数更新方向 $\Delta \theta$ 主要由范数（Norm）最大的梯度决定。当 Aux 梯度远大于主任务梯度时，网络 75% 的容量都在做“回归曲线拟合”，而真正用于捕捉离散交易信号的分类能力被严重削弱。

### 1.3 方向门控的模式坍塌 (Mode Collapse in Mixture of Experts)
`direction_mix_gate` 均值为 0.01（即 100% 偏向空头分支），这是混合专家模型（MoE）中典型的**赢者通吃（Winner-takes-all）坍塌**。在训练早期，如果空头分支偶然获得了略优的损失下降，网络会通过 Sigmoid 门控迅速将权重 $g$ 推向 0，切断多头分支的梯度流（因为 $\frac{\partial L}{\partial w_{bull}} \propto (1-g)$）。多头分支变成死神经元，导致模型宏观方向 F1 极度偏置。

---

## 2. 彻底的优化与修改方案

为了从根源上解决上述数学陷阱，建议对 `loss.py` 和 `kan.py` 进行以下三项核心重构：

### 2.1 非对称负样本降权 (Asymmetric Negative Downweighting)
**科学依据**：与其在整个 96% 的负样本空间中艰难寻优，不如引入非对称掩码（Mask），让模型“忽略”那些预测分数已经很低的无聊波动，只惩罚那些错报高分的“硬负样本（Hard Negatives）”。
**代码修改（`loss.py`）**：
在 BCE Loss 计算后，对真实标签为 0 且预测概率 $< 0.2$ 的样本，将其 Loss 权重强制乘以 $0.1$（降权 90%）。
```python
# 示例伪代码
neg_mask = (event_target < 0.5)
weight = torch.ones_like(bce_loss)
weight[neg_mask & (prob_pred < 0.2)] = 0.1
bce_loss = bce_loss * weight
```
*预期效果*：释放模型的“胆量”，信号频率将迅速从 0.5% 恢复到健康的 4-5%。

### 2.2 辅助损失钳制与局部梯度截断 (Aux Loss Clamping & Hook)
**科学依据**：确保分类主任务对共享时序特征提取器具有绝对的梯度主导权。
**代码修改（`loss.py`）**：
1.  **钳制 Loss**：在将 `aux_loss` 加入总损失前，强制限制其上限（例如 `aux_loss = torch.clamp(aux_loss, max=1.0)`）。
2.  **梯度挂钩（Gradient Hook）**：为防止局部极端值引发梯度雪崩，在 `aux_pred` 上注册 Hook，将反向传播的梯度截断在 `[-1.5, 1.5]`。
```python
if aux_pred_act.requires_grad:
    aux_pred_act.register_hook(lambda grad: torch.clamp(grad, min=-1.5, max=1.5))
```

### 2.3 门控平衡正则化 (Gate Balance Regularization)
**科学依据**：通过显式惩罚偏离最大熵状态的门控分布，强制网络保持多分支（多头/空头）的活性，打破模式坍塌。
**代码修改（`loss.py`）**：
在 `PhysicsLoss.forward` 中提取 `direction_gate_mean`，如果偏离 0.5，则施加 MSE 惩罚。
```python
gate_mean = debug_info.get('direction_gate_mean', 0.5)
balance_penalty = (gate_mean - 0.5) ** 2
main_loss += 0.5 * balance_penalty  # 强迫门控回到中间地带
```

## 3. 总结
本方案摒弃了简单的参数微调，而是从损失函数曲面、梯度流向和正则化约束的底层数学逻辑出发，通过**非对称降权**恢复召回率，通过**梯度截断**保卫主任务，通过**平衡正则化**拯救死神经元。这套组合拳将彻底重塑 KHAOS_KAN 的训练轨迹。