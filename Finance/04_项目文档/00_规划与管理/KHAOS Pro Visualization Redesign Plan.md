# KHAOS 指标系统开发与优化计划

## 1. 核心指标算法实现 (Core Implementation)

直接利用现有的 `PhysicsLayer` 提取物理场数据，构建两个核心复合指标，不再依赖神经网络的黑盒输出：

* **文件**: 创建 `src/khaos_kan/indicator.py`

* **指标 A: KHAOS Instability (不稳定性/趋势终结)**

  * **逻辑**: 融合 `Permutation Entropy` (熵) 和 `Hurst Exponent`。

  * **作用**: 数值越低，趋势越强；数值飙升至阈值（如 0.8），代表趋势结构瓦解（Trend Exhaustion），即**趋势终结信号**。

* **指标 B: KHAOS Anomaly (异常度/波动爆发)**

  * **逻辑**: 融合 `EKF Residual` (残差) 和 `Energy` (能量)。

  * **作用**: 数值突然脉冲，代表物理模型失效，通常对应**大行情启动**或**剧烈变盘**。

## 2. 信号逻辑与可视化 (Signal & Visualization)

开发可视化脚本，直观展示指标如何捕捉“趋势终结”和“波动爆发”。

* **文件**: 创建 `src/khaos_kan/visualize_indicator.py`

* **图表设计**:

  * **主图**: K线图 + **KHAOS Ribbon (彩带)**。彩带颜色代表“物理状态”：

    * 🟢 **绿色**: 稳定流 (低熵，趋势安全)。

    * 🔴 **红色**: 混沌流 (高熵，趋势终结/震荡)。

    * 🟡 **黄色高亮**: 异常脉冲 (波动率预警)。

  * **副图**: `Instability` 曲线与 `Anomaly` 柱状图。

## 3. 调试与参数优化 (Tuning)

* **目标**: 针对 **BTC (高噪)** 和 **SPX (低噪)** 分别调整熵计算的窗口参数和阈值，确保信号不滞后且不频繁误报。

* **交付**: 生成 HTML 格式的指标分析报告，让您直接看到 KHAOS 指标在最近行情中的表现。

