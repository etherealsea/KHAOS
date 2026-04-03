# PI-KAN Technical Implementation & Training Report

**Date:** 2026-03-08
**Model Version:** PI-KAN v2.1 (Distilled)
**Training Device:** NVIDIA GPU (CUDA)

## 1. 核心质疑回应 (Executive Summary)

针对用户关于 **"阈值 1.65 是否经过训练"** 及 **"具体训练结果"** 的疑问，本报告基于实际训练日志 (`train_pi_kan_v2_progress.csv`, `pikan_opt_results_gpu.txt`) 给出确切证据。

### 1.1 阈值 1.65 的来源
- **来源**: 并非随机设定，而是通过 **逻辑优化 (Logic Optimization)** 搜索得出的统计最优解。
- **过程**: 在 `optimize_logic.py` 中，我们在区间 `[1.5, 1.6, 1.65, 1.7, 1.8]` 内对 `sigma_trig` 参数进行了网格搜索。
- **统计意义**: `1.65` 对应标准正态分布的 **95% 置信区间 (Z-Score)**。在 PI-KAN 的输出分布中，超过 1.65 的 Force 值代表发生了 **5% 小概率的强相变事件**。
- **验证**: 在 `pikan_opt_results_gpu.txt` 的 GPU 进化算法结果中，该参数配合 `alpha=0.10` 和 `calib_len=107` 达到了最佳适应度得分 **649.22**。

### 1.2 训练成果概览
- **方向准确率 (Directional Accuracy)**: 验证集达到 **91.98%** (Epoch 50)。
- **预测周期**: t+1 (未来1小时)。信号持续性通常为 3-8 小时。详见 [预测周期与R2分析报告](PIKAN_HORIZON_AND_R2_REPORT.md)。
- **模型拟合度 (R2 Score)**: 蒸馏后的 Force 模型 R2 达到 **0.88**，说明 Pine Script 公式能 88% 还原深度神经网络的决策逻辑。
- **损失收敛**: Total Loss 从初始的 6684.14 降至 1806.95，收敛性极佳。

---

## 2. 训练过程详解 (Training Process)

### 2.1 模型架构 (Architecture)
- **Core**: PI-KAN (Physics-Informed Kolmogorov-Arnold Network)。
- **Features**: 
  - Hurst Exponent (分形维)
  - Permutation Entropy (熵/复杂度)
  - Volatility (波动率)
- **Physics Loss**:
  - $L_{total} = L_{MSE} + 0.1 \cdot L_{Momentum} + 0.01 \cdot L_{MeanRev} + 0.01 \cdot L_{Entropy} + 0.1 \cdot L_{EKF}$
  - 引入物理约束防止过拟合，确保模型符合金融物理学规律（如均值回归属性）。

### 2.2 训练曲线 (Training Curves)
数据来源: `Finance/train_pi_kan_v2_progress.csv`

| Epoch | Train Loss | Val Loss | Val Directional Accuracy (DA) |
| :--- | :--- | :--- | :--- |
| 1 | 6684.14 | 6325.85 | 61.13% |
| 10 | 3836.82 | 3871.36 | 81.11% |
| 30 | 2433.91 | 2550.86 | 88.95% |
| **50** | **1806.95** | **1969.71** | **91.98%** |

> **结论**: 模型在第 30 Epoch 后进入高准确率区间 (>88%)，并在第 50 Epoch 达到峰值，未出现明显过拟合。

---

## 3. 实现细节：从 PyTorch 到 Pine Script (Distillation)

为了在 TradingView 上实时运行复杂的神经网络，我们采用了 **符号回归蒸馏 (Symbolic Regression Distillation)** 技术。

### 3.1 蒸馏原理
1.  **Teacher Model**: 训练好的 PyTorch PI-KAN 模型（参数量大，无法直接在 Pine 运行）。
2.  **Student Model**: 多项式回归模型 (Polynomial Regression)。
3.  **Process**: 使用 Teacher 的输出 (Force) 作为目标，训练 Student 拟合 Teacher 的行为。

### 3.2 蒸馏结果
数据来源: `Finance/training_results.txt`

- **Force Module (M)**: R2 Score = **0.88** (极高拟合度)
- **Chaos Module (R)**: R2 Score = 0.80
- **Trend Module (Q)**: R2 Score = 0.59

### 3.3 公式推导 (The "Magic Numbers")
Pine Script 中的系数正是来自 Student Model 的权重。例如 `generated_force_model.pine` 中的：

```pine
float force = -0.007978318259
force += 0.039860282093 * h_norm  // Hurst 权重
force += 0.029056455940 * e_norm  // Entropy 权重
...
```

这些数字不是随机的，而是 Python `sklearn.linear_model.LinearRegression.coef_` 的直接输出。

---

## 4. 最终参数优化结果 (Final Parameters)

通过 GPU 遗传算法 (`optimize_pikan_gpu.py`) 对不可导参数进行进化搜索，得到最优组：

- **Alpha (EKF Noise)**: `0.1000` (控制能量反应速度)
- **Calibration Window**: `107` (最佳回溯周期)
- **Volatility Gate**: `-0.64` (过滤低波动的死盘)
- **Hurst Gate**: `0.90` (防止在极强趋势中逆势)
- **Trigger Sigma**: `1.65` (置信度阈值)

**Best Fitness Score**: 649.22 (综合考虑了胜率、盈亏比和交易频率)。
