# KHAOS-PIKAN 架构级硬接线 (Ansatz Hard-Wiring) 改造方案

## 1. 任务背景与目标
**痛点**：金融时间序列高噪、刚性（stiff）。在模型训练中，使用 Loss 惩罚（如 EDL、硬/软惩罚项）来约束网络，会导致极端的梯度冲突。这些冲突不仅使得网络输出假阳性信号（乱发信号），还会撕裂 KAN 网络的 B样条控制网格，引发训练崩溃或认知坍缩（不敢发信号）。
**目标**：彻底废除基于 Loss 的物理惩罚，转向 **Ansatz 架构级物理硬接线（Physics-Informed Ansatz Hard-Wiring）**。通过引入不可学习的物理距离门控函数 $D(t)$，在网络拓扑结构中直接阻断高噪数据引起的错误信号。

## 2. 改造方案：平滑微分距离函数门控 (Soft Differentiable Gating)
### 2.1 `kan.py` 前向传播架构调整
在 `breakout_head` 和 `reversion_head` 产生原始输出（raw logits）后，不直接返回，而是引入物理先验计算出的门控权重进行过滤。

- **门控 1：方向一致性门控 (Directional Gate)**
  - **逻辑**：只有当 `bull_score` 和 `bear_score` 存在显著差异（即市场具有明确趋势底座）时，门控开启；如果两者都在震荡，门控趋近 0。
  - **实现**：`directional_gate = torch.sigmoid(alpha * (abs(bull_score - bear_score) - threshold)).detach()`

- **门控 2：物理状态门控 (Physics State Gate - Compression/Volatility)**
  - **逻辑**：从输入的 `physics_state` 中提取压缩率/波动率特征。
  - **实现**：`compression_gate = torch.sigmoid(beta * physics_state[..., IDX]).detach()`

- **输出组装**：
  `final_breakout_logits = raw_breakout_logits * directional_gate * compression_gate`
  *(注意：所有的门控必须调用 `.detach()` 切断梯度，以确保它们是不可学习的纯物理先验)*

### 2.2 `loss.py` 目标函数洗净
由于错误信号已被 `kan.py` 的物理门控阻断，不再需要事后惩罚。需要执行大规模的“删减”操作，将网络恢复为纯粹的数据拟合：
- 移除 `directional_violation` 惩罚。
- 移除 `public_below_directional_violation` 惩罚。
- 移除 `direction_consistency_loss` 惩罚。
- 移除复杂的 Curriculum Training（`lambda_phys`、`epsilon` 等软容差逻辑）。
- 仅保留基于预测 logits 与真实事件标签（`breakout_event`, `reversion_event`）的基础分类/回归 Loss。

## 3. 验收标准
1. **梯度健康**：KAN 网络的 B样条参数在训练过程中不再出现极端的梯度飙升。
2. **假阳性下降**：由于物理门控的存在，模型在震荡市、低压缩比等不满足物理先验的时期，发出的交易信号将直接被阻断（接近于 0）。
3. **胜率提升**：模型能够在符合物理先验的窗口期，更有效、更精准地拟合突破/反转形态，准确率（Precision）显著回升。