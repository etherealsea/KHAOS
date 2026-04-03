# Tasks

- [x] Task 1: 统一 Python 与 Pine Script 特征及自适应归一化
  - [x] SubTask 1.1: 审查并重写 `khaos/核心引擎/physics.py`，实现与 Pine Script 中 `sum_tr / rng` 及对数收益率 EMA 散度完全一致的代理逻辑。
  - [x] SubTask 1.2: 在模型输入端实现并集成 RevIN (Reversible Instance Normalization) 模块，消除多周期与流式数据带来的漂移。

- [x] Task 2: 改造 KAN 与 Attention 架构以支持强可解释性
  - [x] SubTask 2.1: 修改 `khaos/模型定义/kan.py`，为 KANLinear 层的样条系数（Spline Coefficients）增加 L1 稀疏正则化接口。
  - [x] SubTask 2.2: 修改 `khaos/模型定义/attention.py`，使注意力模块在前向传播时能够返回注意力权重矩阵（Attention Weights），用于时序重要性分析。

- [x] Task 3: 引入物理先验与优化目标
  - [x] SubTask 3.1: 修正目标标签 `Efficient Return`，严格排查前视偏差。
  - [x] SubTask 3.2: 在 `khaos/模型训练/loss.py` 中引入物理先验损失项（如要求输出信号的导数符合临界慢化阶段的方差激增特性），约束 KAN 的学习方向。

- [x] Task 4: 开发基于 KAN 的符号回归公式提取器
  - [x] SubTask 4.1: 新增 `khaos/回测模块/symbolic_extractor.py`，实现从剪枝后的 KAN 层提取近似解析公式的逻辑（如利用 SymPy 将样条函数拟合为多项式或基础函数）。
  - [x] SubTask 4.2: 编写基于注意力权重的时序延迟规则提取，确定 Pine Script 中的动态 Lookback 周期。
  - [x] SubTask 4.3: 验证提取出的解析公式与注意力规则在 TradingView 模板中的直接可用性。

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 2], [Task 3]