# 任务列表 (Tasks)

- [x] 任务 1：重构多维连续目标标签 (Data Loader)
  - [x] 子任务 1.1：修改 `Finance/02_核心代码/源代码/khaos/数据处理/data_loader.py`，彻底移除旧的单维度有效收益率目标逻辑。
  - [x] 子任务 1.2：计算 `Target 0`：使用 pandas 滚动窗口计算未来 `forecast_horizon` 周期的真实标准差 `forward_vol`，并生成 `np.log(forward_vol / (self.sigma + 1e-8))`。
  - [x] 子任务 1.3：计算 `Target 1`：基于当前均线偏离度 `(close - ema) / sigma` 和未来标准化收益率 `forward_return / sigma` 的负向交叉乘积，生成均值回归强度目标。
  - [x] 子任务 1.4：将两者组合为 `[N, 2]` 维度的 `self.targets`。
  - [x] 子任务 1.5：调整 `sample_weights`，将两个目标在正向区间的幅度叠加作为动态权重放大因子，消除硬阈值。
- [x] 任务 2：更新模型结构与损失函数 (Training & Loss)
  - [x] 子任务 2.1：修改 `Finance/02_核心代码/源代码/khaos/模型训练/train.py`，将 `KHAOS_KAN` 的实例化参数 `output_dim` 改为 2，并保持主损失函数为 `MSELoss()`。
  - [x] 子任务 2.2：修改 `Finance/02_核心代码/源代码/khaos/模型训练/loss.py`。解包 2D 预测值 `pred`，移除旧的绝对值逻辑。
  - [x] 子任务 2.3：在 `loss.py` 中重构物理惩罚项。例如针对 `pred[..., 0]`（波动率）应用高熵惩罚；针对 `pred[..., 1]`（反转强度）应用残差与极值回归惩罚。
- [x] 任务 3：适配规则提取与验证
  - [x] 子任务 3.1：检查 `Finance/02_核心代码/源代码/khaos/回测模块/symbolic_extractor.py`，适配 `output_dim=2` 的模型结构（KAN 网络输出节点数量变为2），确保梯度计算不会因形状不匹配而报错。
  - [x] 子任务 3.2：运行 `python Finance/02_核心代码/源代码/khaos/模型训练/train.py --test_mode`，验证全流程计算图和多维张量形状是否正确对齐。

# 任务依赖 (Task Dependencies)
- [任务 2] 依赖于 [任务 1]
- [任务 3] 依赖于 [任务 1], [任务 2]