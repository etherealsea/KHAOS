# KHAOS-KAN 架构升级设计文档：证据深度学习 (EDL)

## 1. 简介

本设计文档记录了 KHAOS-KAN 在 Iter11 针对多资产高噪声金融时间序列训练中的重大架构升级。我们将网络输出层与损失函数从传统的 Softmax 二分类体系升级为 **证据深度学习 (Evidential Deep Learning, EDL)** 体系。

## 2. 设计动机与现状痛点

在处理金融时间序列（特别是全球多资产、多周期数据）时，最突出的特征是 **低信噪比（Aleatoric Uncertainty 极高）**。市场在 90% 的时间内处于随机游走状态（震荡市），仅有 10% 的时间存在真正的结构性机会（信号）。

**原架构的缺陷：**
1. **过度自信的分类器**：传统的二分类网络被迫在“突破”与“非突破”之间做出概率分配。即使输入特征毫无规律，网络也会输出一个虚高的概率（如 0.6），导致大量假阳性（Hard Negatives）。
2. **生硬的后处理约束**：为了解决上述问题，我们在 `train.py` 的评估阶段人为设置了极高的门槛（如 `min_precision=0.7` 和 `max_frequency=0.05`）。这种“标量化后处理”不仅破坏了反向传播梯度的平滑性，还导致模型在早期 Epoch 由于找不到及格阈值而获得极端惩罚（-999），搜索器被迫退保至极端分位数。

## 3. 证据深度学习 (EDL) 核心机制

为了让模型自然地学出高置信度的决策面，我们引入了 EDL 框架。其核心思想是：不输出绝对概率，而是输出支持每个类别的 **证据 (Evidence)**，并在此基础上计算不确定性。

### 3.1 数学基础

在 EDL 中，分类概率不再是一个固定的值，而是服从一个 **Dirichlet 分布**。
假设我们有 $K$ 个类别：
1. **模型输出**：神经网络不再通过 Softmax 输出概率 $p_k$，而是通过非负激活函数（如 ReLU 或 Softplus）输出每个类别的证据 $e_k \ge 0$。
2. **Dirichlet 参数**：我们将证据转化为 Dirichlet 分布的参数 $\alpha_k = e_k + 1$。
3. **Dirichlet 总和**：$S = \sum_{k=1}^K \alpha_k = \sum_{k=1}^K e_k + K$。
4. **预期概率**：类别 $k$ 的预测概率为 $\hat{p}_k = \frac{\alpha_k}{S}$。
5. **不确定性 (Uncertainty)**：模型整体的不确定性被定义为 $u = \frac{K}{S}$。
   - 当所有证据 $e_k \to 0$ 时，$\alpha_k \to 1$，$S \to K$，此时 $u \to 1$（模型处于 100% “懵逼”状态）。
   - 当某个类别的证据 $e_k \to \infty$ 时，$S \to \infty$，此时 $u \to 0$（模型极度自信）。

### 3.2 损失函数重构 (Evidential Loss)

我们将现有的交叉熵或 Margin Loss 替换为基于 Dirichlet 分布的损失函数，主要由两部分组成：
$$ \mathcal{L}(\Theta) = \mathcal{L}_{err}(\Theta) + \lambda \mathcal{L}_{KL}(\Theta) $$

1. **数据拟合损失 ($\mathcal{L}_{err}$)**：
   采用预测分布与独热标签的负对数似然（Negative Log-Likelihood, NLL），或采用更鲁棒的期望交叉熵（Expected Cross Entropy）。
   $$ \mathcal{L}_{err} = \sum_{k=1}^K y_k (\log(S) - \log(\alpha_k)) $$
2. **KL 散度惩罚 ($\mathcal{L}_{KL}$)**：
   **这是过滤震荡市的关键！** 当模型预测错误时（特别是对于引发巨大回撤的 Hard Negatives），我们施加 KL 散度惩罚，强行剥夺模型对该样本的误导性“证据”，将其分布推向均匀分布（即完全不确定，$\alpha \to 1$）。

## 4. 架构改造方案

### 4.1 网络结构改造 (`kan.py`)
- 保留现有的 KHAOS 物理特征提取与 KAN 网络主干。
- **输出层激活**：将最终的 `event_score` 头（如 `breakout_score`、`reversion_score`）的输出取消原本无界的 Logits 形式，通过 `ReLU` 激活，确保输出的证据非负（$e \ge 0$）。

### 4.2 损失函数改造 (`loss.py`)
- 引入新的 `EvidentialLoss` 模块。
- 在计算 `hard_negative_penalty` 时，将其作为 KL 散度的核心放大器。对于产生巨大回撤的错误样本，施加重罚，迫使模型在该类形态下输出 $e = 0$。

### 4.3 评估与交易决策改造 (`train.py`)
- **移除硬性门槛**：彻底删除 `min_precision=0.7` 和 `max_frequency=0.05` 等后处理死规定。
- **双重过滤发信**：
  在回测与评分排序时，使用以下条件过滤信号：
  `IF Probability > threshold AND Uncertainty < uncertainty_threshold THEN Trade`
- 这种双重过滤不仅能精确捕捉胜率，还能让模型通过 `Uncertainty` 指标直接剔除 90% 的无规律震荡市，极大地提升真实盈亏比。

## 5. 预期收益

- **更快的收敛**：模型不再因为极端的后处理惩罚而在早期 Epoch 撞墙，梯度下降将变得自然平滑。
- **更高的胜率上限**：通过明确区分“数据噪声（Aleatoric）”与“模型无知（Epistemic）”，我们能在低信噪比的金融市场中，精准定位到那 10% 高置信度的结构性行情。
- **鲁棒的泛化能力**：由于取消了针对特定数据集（如 A 股）微调的硬性阈值，基于 EDL 的网络将天然具备多资产（Crypto, FX, Indices）的泛化能力。
