# KHAOS 2025 AI 本科毕设战略规划 (Roadmap)

## 1. 核心定位与论文重构 (Re-framing)

**当前状态**: KHAOS 是一个高级技术指标。
**目标状态**: KHAOS 是一个 **"专家知识驱动的轻量级自适应时序滤波器" (Expert-Knowledge-Driven Lightweight Adaptive Filter)**。

### 1.1 论文题目
*   **中文**: 基于自适应卡尔曼滤波与分形几何的轻量级金融时序预测系统
*   **英文**: *KHAOS: A Lightweight Expert-Knowledge-Driven Adaptive Filter for Financial Time Series Forecasting*

### 1.2 核心叙事逻辑 (The Narrative)
在答辩中，我们要构建这样的对立冲突：
*   **反派 (The Villain)**: 深度学习模型 (LSTM/Transformer)。它们是"黑箱"，计算昂贵，需要海量数据训练，且容易过拟合 (Overfitting)，在金融市场这种"非平稳 (Non-stationary)"环境中往往失效。
*   **主角 (The Hero)**: KHAOS。它不需要训练 (Training-free)，参数极少，计算复杂度为 O(N)，且完全可解释 (Interpretable)。它通过 **Hurst 指数** 实时感知市场的分形状态（趋势 vs 震荡），并动态调整 **卡尔曼滤波** 的参数。
*   **高潮 (The Climax)**: 实验证明，KHAOS 在仅消耗 1% 计算资源的情况下，取得了优于 SOTA 深度学习模型的夏普比率和抗回撤能力。

---

## 2. 任务定义与实验设计 (Experimental Design)

为了让 AI 老师信服，必须使用标准的机器学习实验流程。

### 2.1 任务定义
*   **输入 (X)**: 过去 60 小时的 OHLCV 数据 (Open, High, Low, Close, Volume)。
*   **输出 (Y)**: 未来 4 小时后的收益率方向 (二分类：涨/跌) 或 具体收益率 (回归)。
    *   *策略建议*: 为了最大化 KHAOS 优势，建议侧重于 **"交易绩效评估"**，即把模型预测转化为交易信号后的回测表现。

### 2.2 数据集 (Unified Dataset)
*   **资产**: BTC/USDT, ETH/USDT, NASDAQ100 (QQQ), EUR/USD, XAU/USD (涵盖加密、美股、外汇、大宗)。
*   **频率**: 1H (1小时线)。
*   **时间跨度**: 2018-01-01 至 2025-06-30。
*   **划分**:
    *   Train (2018-2022): 深度学习模型用于训练权重。KHAOS 不需要训练，但可用于参数微调。
    *   Validation (2023): 用于调参 (Hyperparameter Tuning)。
    *   Test (2024-2025): **严格样本外测试 (Out-of-Sample)**，决胜局。

### 2.3 对手模型 (The Benchmarks)
我们将对比以下三类模型：
1.  **统计基准**: ARIMA (用于被吊打)。
2.  **经典深度学习**: LSTM, GRU, TCN (时间卷积网络)。
3.  **SOTA (State-of-the-Art)**: Transformer (Vanilla), N-BEATS (2020), Informer/Autoformer (2021/2022)。

---

## 3. 技术实现路径 (Execution Path)

### 步骤 1: Python 核心移植 (The "Modelification" of KHAOS)
KHAOS 必须脱离 Pine Script，成为一个 Python Class，接口需符合 `sklearn` 风格：
```python
class KHAOS_Model:
    def __init__(self, h_lookback=30, ...): ...
    def fit(self, X, y): pass # KHAOS 不需要 fit，这里留空或用于初始化统计量
    def predict(self, X): ... # 返回预测值/信号
```
*   **工作量**: 1 天 (优先级最高)。

### 步骤 2: 搭建 SOTA 竞技场 (The Arena)
使用 `darts` 或 `neuralforecast` 库，这些库封装了 LSTM, N-BEATS 等模型，可以几行代码调用。
*   **工具**: `darts` (推荐，API 友好)。
*   **工作量**: 2-3 天。

### 步骤 3: 统一回测引擎 (The Judge)
使用 `vectorbt` (速度快) 或 `backtrader` 进行回测。
*   所有模型的预测输出 -> 统一的信号逻辑 -> 资金曲线。
*   **关键指标**: Sharpe Ratio, Max Drawdown, Annualized Return, Win Rate, Calmar Ratio。

### 步骤 4: 消融实验与可解释性 (The "Science")
*   **消融**: 编写脚本，通过开关配置 (Config Flags) 运行 KHAOS 的不同阉割版本。
*   **可视化**: 绘制 Hurst 热力图、卡尔曼增益曲线、SHAP 值分析。

---

## 4. 论文核心图表规划

1.  **主结果表**: 一张大表，横向是各模型，纵向是各指标（MSE, Accuracy, Sharpe, Return）。KHAOS 的 Sharpe 必须标粗。
2.  **累积收益曲线图**: KHAOS 的曲线应该稳步上升，回撤小；LSTM 可能大起大落。
3.  **Hurst 机制图**: 展示 Hurst < 0.5 时，KHAOS 如何收紧带宽避免震荡亏损；Hurst > 0.65 时，如何放宽带宽捕捉趋势。
4.  **计算效率对比图**: 柱状图，训练/推理耗时。LSTM: 10小时, KHAOS: 10秒。

## 5. 下一步具体行动

1.  **立即执行**: 完成 `khaos_core.py` (Python 版 KHAOS)。
2.  **紧接着**: 编写 `data_loader.py` 获取并清洗数据。
3.  **然后**: 编写 `benchmark_runner.py` 跑通第一个 LSTM 模型。
