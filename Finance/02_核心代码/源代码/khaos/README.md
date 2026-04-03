# KHAOS: Kinetic Hurst Adaptive Oscillator System

Core package for the PI-KAN based financial phase detection system.

## Structure
- `模型定义/`: PyTorch 模型定义 (PI-KAN 等)
- `模型训练/`: 训练脚本与损失函数
- `数据处理/`: 数据加载、处理与获取脚本
- `核心引擎/`: 核心物理引擎逻辑 (EKF, Hurst, Entropy) 和指标计算
- `工具箱/`: 可视化和报告生成工具
- `Pine Script代码/`: Pine Script 代码

## Usage
Run training:
```bash
python -m khaos.模型训练.train
```
