# Iter12 神经网络深度优化方案：废除物理惩罚与修复 KAN 坍塌

## 1. 摘要
根据前期分析与业务诉求，模型不需要在所有物理状态下都被强制学习和区分，只需专注于捕捉交易信号。本方案将彻底移除损失函数中导致梯度冲突的物理惩罚项（P3/P4/P6等），完全依托 Iter12 引入的前向软门控（Soft Gating）来进行物理约束；同时修复 KAN 网络内部因 `Tanh` 激活函数导致的 B样条坍塌问题。

## 2. 现状分析
- **损失函数撕裂**：`loss.py` 中的 `p3, p4, p6, p7` 等物理惩罚项，强制要求模型在排列熵（Ent）或李雅普诺夫指数（MLE）较高时预测无波动。这使得主损失（捕捉交易机会）与物理损失发生严重拔河，污染了正常交易信号，导致假阳性（Hard Negative）极高。
- **B样条坍塌（B-Spline Collapse）**：`kan.py` 的 `KANHead` 在层与层之间使用了 `torch.tanh(x)` 进行激活。在遇到极端物理脉冲时，`Tanh` 会将特征挤压到 `[-1, 1]` 的边界极值处，落入 B样条的梯度死区，导致权重无法更新，最终引发 `catastrophic_output_collapse`（输出变为一条直线）。

## 3. 具体修改方案

### 3.1 净化目标函数 (`Finance/02_核心代码/源代码/khaos/模型训练/loss.py`)
- **移除冗余物理惩罚计算**：删除 `p3`、`p4`、`p6_lyapunov`、`p7_csd`、`p7_false_reversion` 以及 `continuation_bias` 等变量的计算逻辑。
- **清理日志与汇总**：将上述移除的变量从 `logs` 字典中剔除。
- **清理权重预设**：从 `LOSS_WEIGHT_PRESETS` 的所有配置字典中删除 `p3`, `p4`, `p6`, `p7` 的权重定义。
- **保留核心**：仅保留主损失（BCE）、辅助任务损失（Aux）、事件间隔（Event Gap）、强负样本（Hard Negative）以及门控相关的结构化损失。

### 3.2 修复 KAN 结构死区 (`Finance/02_核心代码/源代码/khaos/模型定义/kan.py`)
- **替换/移除 Tanh 激活**：在 `KANHead.forward` 方法中，去掉 `x = torch.tanh(x)`。由于 KAN 本身的 B样条已经具备强大的非线性映射能力，额外的 `Tanh` 截断反而有害。可将其替换为 `x = torch.nn.functional.silu(x)` 或直接移除（推荐替换为 SiLU 以保留平滑的非线性且无硬性边界）。
- **确认 Soft Gating**：确认前向传播末端的门控（`compression_gate`, `directional_gate`）作为唯一的物理先验拦截器，接管被删除的损失函数惩罚职能。

## 4. 假设与决策
- **决策**：完全信任 Ansatz Hard-Wiring（架构级硬接线），即通过前向传播时的物理门控来阻断错误信号，而不是通过 Loss 惩罚来纠正。
- **假设**：去除 `Tanh` 后，物理特征的极端值能够顺畅通过 B样条的基底，梯度流恢复正常，从而解决输出坍塌问题。

## 5. 验证步骤
1. 修改完成后，运行 `python test_loss.py`（或类似本地测试脚本）确保损失函数剔除无误且维度对齐。
2. 运行 `python test_model_gating.py` 确保 KAN 前向传播正常。
3. 使用 `run_iter12_multiasset_closed_loop.py --phase smoke` 跑 1-2 个 Epoch，观察训练日志，确认不再触发 `catastrophic_output_collapse`，且 `main` loss 能够正常下降。