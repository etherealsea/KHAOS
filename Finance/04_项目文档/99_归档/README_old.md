# PI-KAN Finance Project

基于物理信息增强神经网络 (PI-KAN) 的金融时间序列相变探测系统。

## 目录结构 (Directory Structure)

### 1. 核心代码 (`src/`)
- `khaos/`: 核心模型逻辑。
    - `model/`: PI-KAN 网络定义 (KAN + Physics Loss)。
    - `ekf/`: 扩展卡尔曼滤波 (EKF) 状态估计。
    - `pine/`: **KHAOS TradingView 指标脚本** (核心产出)。

### 2. 文档 (`docs/`)
- **[PIKAN_TO_INDICATOR_MAPPING.md](docs/PIKAN_TO_INDICATOR_MAPPING.md)**: **必读**。解释 PI-KAN 训练结果如何映射为 KHAOS 指标的“红蓝警示线”。
- `reports/`: 模型训练报告与回测结果。
- `thesis/`: 毕业论文相关资料与开题报告。
- `archive/`: 归档的旧开发日志、已废弃的设计文档。

### 3. 脚本 (`scripts/`)
- `pipeline/`: 数据预处理、模型训练、参数优化全流程脚本。

## 快速开始 (Quick Start)

1.  **查看指标逻辑**: 阅读 `docs/PIKAN_TO_INDICATOR_MAPPING.md`。
2.  **获取指标代码**: 复制 `src/khaos/pine/KHAOS.pine` 到 TradingView。
3.  **运行训练**: 使用 `scripts/pipeline/train_pi_kan_v2.py` (需配置 Python 环境)。
