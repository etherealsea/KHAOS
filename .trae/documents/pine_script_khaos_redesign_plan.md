# Pine Script KHAOS 指标重构方案 (PI-KAN V3)

## 1. 背景与目标
根据最新成功跑通的 PI-KAN 混合精度模型训练结果（见 `2026-03-23.md` 开发日志），我们通过 `symbolic_extractor.py` 提取到了全新的、基于真实多周期大样本数据训练出的非线性规则和物理阈值。现有的 `KHAOS.pine` 脚本使用的是旧版的拍脑袋阈值（如 MLE > 0.27, Entropy > 0.70），这存在严重的“语义鸿沟”。

**目标**：将最新的物理规则、阈值以及长短周期注意力机制融合进 `KHAOS.pine` 中，使其判定逻辑与真实的神经网络输出对齐，并在 TradingView 副图中实现优雅的可视化。

---

## 2. 核心规则的精确数学复刻 (Model to Pine Script)

由于 Python 端 (PI-KAN) 输出的是经过 `RevIN` 归一化后的阈值（本质上是类似 Z-Score 的相对分布值），为了精确复刻神经网络学到的规律，我们在 Pine Script 中不能采用拍脑袋的“百分位”定性映射，而是必须**严格重现 Z-Score 归一化计算过程**，并直接使用网络提取出的浮点阈值。

### 2.1 物理特征的严格 Z-Score 映射
我们将为每个核心特征引入滚动 Z-Score 转换，公式为：`Z = (X - SMA(X, 20)) / Stdev(X, 20)`，其中 20 对应 PI-KAN 的训练窗口 `lookback`。这使得 Pine 算出的特征在分布上无限逼近 Python 端的 RevIN 预处理结果。

*   **Hurst (趋势基准)**：
    *   *网络规律*：Hurst 归一化值 > -0.13 时，系统脱离震荡进入趋势态，此时物理特征作用被放大。
    *   *数学复刻*：计算 `Hurst` 的 Z-Score，设定条件 `z_hurst > -0.13`。
*   **突破/混沌相变驱动 (Breakout Phase)**：
    *   *网络规律*：当系统在趋势态中，若 Volatility > -1.00，或 MLE > -0.09，或 Entropy > 0.07 时，系统进入强烈的非线性激增（突破）。
    *   *数学复刻*：计算各自的 Z-Score。当 `(z_mle > -0.09 or z_entropy > 0.07)` 且 `z_volatility > -1.00` 时，触发突破相变。
*   **极值反转惩罚 (Reversion Phase)**：
    *   *网络规律*：当残差冲击与均值偏离度达到极端情况时（abs(EKF_Res) > 0.86 且 abs(EMA_Div) > 0.64），触发均值回归。
    *   *数学复刻*：计算 `z_ekf_res` 和 `z_ema_div`。判定条件为 `math.abs(z_ekf_res) > 0.86` 且 `math.abs(z_ema_div) > 0.64`。

### 2.2 时序注意力锚点 (Trigger Anchor) 的精确实现
*   *网络规律*：网络并非在当前时刻 (t-0) 立即对异动作出反应，而是将最高权重 (9.5%) 放在了 `t-10` 位置，同时远期 (t-8 之前) 占了 57.9%。这代表模型是在**事后确认**：即“不久前（过去10根K线内）曾发生过物理异动，且这种异动的能量一直维持至今，所以现在确认相变成立”。
*   *数学复刻*：
    使用 Pine 的 `ta.barssince` 或 `ta.highestbars`。对于突破条件，我们不要求当前时刻必须大于阈值，而是要求：
    1. 在过去的 10 根 K 线内（`anchor_lookback = 10`），**曾经发生过** `z_mle > -0.09` 等核心异动（Trigger）。
    2. 当前的动能（`EKF_Velocity`）与异动方向一致，证明能量未消散。
    只有满足这两个条件，才点亮主线输出信号。

---

## 3. Pine Script 代码结构重构

### Step 1: 引入标准化模块
编写一个通用的 Z-Score 函数：
```pine
f_zscore(src, length) =>
    mean = ta.sma(src, length)
    std = ta.stdev(src, length)
    std == 0 ? 0 : (src - mean) / std
```

### Step 2: 重新定义硬性输入参数 (Inputs)
将用户界面中的参数默认值严格锁定为神经网络提取的值：
```pine
float thres_hurst   = input.float(-0.13, "Hurst Z-Threshold")
float thres_mle     = input.float(-0.09, "MLE Z-Threshold")
float thres_entropy = input.float(0.07,  "Entropy Z-Threshold")
float thres_res     = input.float(0.86,  "EKF Residual Z-Threshold")
float thres_div     = input.float(0.64,  "EMA Div Z-Threshold")
```

### Step 3: 状态机与锚点逻辑重写
```pine
// 1. 计算所有原始特征的 Z-Score
z_hurst = f_zscore(hurst, 20)
z_mle = f_zscore(mle, 20)
// ... 其他特征

// 2. 检查过去 10 根 K 线内是否触发过混沌突变 (Anchor)
bool raw_breakout_trigger = (z_mle > thres_mle or z_entropy > thres_entropy) and z_hurst > thres_hurst
bool has_breakout_anchor = ta.barssince(raw_breakout_trigger) <= 10

// 3. 检查反转极值
bool is_reversion = math.abs(z_ekf_res) > thres_res and math.abs(z_ema_div) > thres_div
```

### Step 4: 可视化层 (Visualization)
保持现有的副图设计风格，因为其“底层能量网格 + 主线阶梯”的表达非常清晰。
- **底层直方图**：展示 `EKF_Res` 的强度。
- **主线阶梯**：展示 `EKF_Velocity`，并在进入相变时（Phase 1/2 为红，Phase 3/4 为绿）加粗高亮。

---

## 4. 下一步行动
等待您的确认。如果您同意此方案，我将直接开始编辑 `KHAOS.pine` 文件，完成代码层面的重构。