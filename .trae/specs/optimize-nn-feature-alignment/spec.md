# Optimize NN Feature Alignment & Interpretable Training Spec

## Why
在早期的 PI-KAN V3 (RA-PI-KAN) 实践中，模型虽然在训练集表现优异，但实盘映射完全失效。这不仅是因为“物理代理指标的语义鸿沟”和“多周期归一化差异”，更深层的原因在于**我们没有真正发挥 KAN（柯尔莫哥洛夫-阿诺德网络）和注意力机制的可解释性优势**。如果仅仅把 RA-PI-KAN 当作黑盒，然后用决策树去拟合其特征输出，这与传统的“量化因子挖掘”毫无二致。我们必须让神经网络自身学习到具有强物理意义且可直接转化为解析公式（Symbolic Formula）的规律。

## What Changes
- **统一特征工程与自适应归一化**：在 Python 端完全对齐 Pine Script 的简易代理指标计算逻辑，并在网络前置层引入流式自适应归一化（如 RevIN），消除实盘流式数据的基准漂移。
- **KAN 样条稀疏化与符号回归 (Symbolic Extraction)**：在 KAN 层的损失函数中引入 L1/Entropy 稀疏正则化，迫使网络仅依赖少数核心物理特征的非线性组合。训练完成后，通过符号回归（Symbolic Regression）直接从 KAN 的样条函数中提取出解析数学公式，作为 TradingView 指标的核心算法，彻底取代原有的黑盒决策树提取。
- **注意力权重的物理释义 (Interpretable Attention)**：修改 `AttentionResidualBlock` 使其输出注意力权重图。通过分析注意力在时序上的分布，动态确定相变发生前的“有效回溯窗口（Lookback Window）”，并将此时序延迟规律硬编码或动态映射到 Pine Script 中。
- **物理先验损失 (Physics-Informed Constraint)**：在预测“有效收益率”的基础上，增加物理约束损失（如符合 Langevin 动力学或临界慢化特征的导数惩罚项），强制网络学习真正的物理相变规律，而非拟合市场噪音。

## Impact
- Affected specs: 物理特征预处理、模型架构定义（引入稀疏性与注意力输出）、损失函数、回测与规则提取模块（符号回归替代决策树）。
- Affected code: `khaos/核心引擎/physics.py`, `khaos/模型定义/kan.py`, `khaos/模型定义/attention.py`, `khaos/模型训练/loss.py`, 新增 `khaos/回测模块/symbolic_extractor.py`。

## MODIFIED Requirements
### Requirement: 强可解释性的 KAN 规则提取
不再使用决策树在模型外部提取“死板的阈值”。必须利用 KAN 的数学特性，将其学到的非线性映射直接转化为形如 `Signal = a * f(MLE) + b * g(Entropy)` 的明确解析公式，用于 Pine Script。

### Requirement: 物理特征对齐与归一化
Python 训练端的指标必须与 Pine Script 的流式计算逻辑 1:1 对齐，并通过 RevIN 等机制自适应不同时间周期的波动率。