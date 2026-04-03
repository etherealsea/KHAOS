# Tasks

- [ ] Task 1: 实现可微的最大李雅普诺夫指数 (MLE) 代理
  - [ ] SubTask 1.1: 在 `khaos/核心引擎/physics.py` 中编写 `calculate_lyapunov_proxy` 函数，使用对数收益率散度估算局部混沌度。
  - [ ] SubTask 1.2: 在 `PhysicsLayer` 中集成 MLE 特征计算，更新输出特征维度（15维增加至16维或对应调整）。

- [ ] Task 2: 实现基于 Kimi 论文思路的注意力残差模块
  - [ ] SubTask 2.1: 在 `khaos/模型定义/` 下创建 `attention.py`。
  - [ ] SubTask 2.2: 实现带有门控机制/缩放因子的残差连接模块（Scaled/Gated Residual Connection）。
  - [ ] SubTask 2.3: 实现多头自注意力块（Multi-Head Self Attention），并与残差机制结合。

- [ ] Task 3: 重构 PI-KAN 主干架构 (RA-PI-KAN)
  - [ ] SubTask 3.1: 修改 `khaos/模型定义/kan.py`，将输入从 Flatten 后的向量改为 `(Batch, Seq, Feature)`。
  - [ ] SubTask 3.2: 将 Attention Residual Block 串联在 KANLinear 层之前。
  - [ ] SubTask 3.3: 增加时序池化层（如提取序列最后一个时间步的特征，或进行 Global Average Pooling），以匹配后续 KAN 层的输入要求。

- [ ] Task 4: 更新物理信息损失函数 (P6 Constraint)
  - [ ] SubTask 4.1: 修改 `khaos/模型训练/loss.py` 中的 `PhysicsLoss` 类。
  - [ ] SubTask 4.2: 从 `physics_state` 中提取 MLE 变量，计算“混沌-置信度一致性”与“相变爆发预警”损失。
  - [ ] SubTask 4.3: 将新计算的 `p6_lyapunov` 惩罚项加入总损失函数。

- [ ] Task 5: 适配并测试训练流程
  - [ ] SubTask 5.1: 修改 `khaos/模型训练/train.py` 中的数据切片逻辑，将 `(Batch, Seq, Features)` 直接送入网络，而不再进行 reshape 展平。
  - [ ] SubTask 5.2: 调整模型初始化参数（`input_dim` 变更为单步特征维度，而非序列展平维度）。
  - [ ] SubTask 5.3: 运行小批量训练（Dummy Data 或截取的小数据集），验证前向传播与反向传播是否无报错，Loss 是否正常下降。

# Task Dependencies
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 1], [Task 3], [Task 4]
