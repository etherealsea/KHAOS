# 基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究

本仓库用于维护一个面向金融时间序列相变探测的 PI-KAN / KHAOS 研究工程，目标是同时保留：

- 可继续训练和分析的核心代码
- 可供论文与研发复盘使用的开发文档
- 能让新的 AI 或协作者只读 GitHub 也能接手推进的上下文

## 当前主线

- 当前最稳定的公开基线是 `iterA3_ashare`
- `iterA5_ashare` 完成了多尺度诊断能力补齐，但截至 `2026-04-07` 的阶段复盘，尚未达到升级门槛
- 当前建议继续推进的方向是 `teacher-first + shortT balanced` 系列实验，主入口为：
  [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

如果你是新接手的 AI，请先读：

1. [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
2. [`AI_HANDOFF.md`](AI_HANDOFF.md)
3. [`DATA_CONTRACT.md`](DATA_CONTRACT.md)
4. [`Finance/03_实验与验证/脚本/README.md`](Finance/03_实验与验证/脚本/README.md)
5. [`Finance/02_核心代码/源代码/khaos/README.md`](Finance/02_核心代码/源代码/khaos/README.md)

## 快速开始

### 1. 准备环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

说明：

- 本仓库没有上传原始数据、研究处理数据、训练日志和权重文件
- 若要实际训练，需要先在本地准备数据目录，规则见 [`DATA_CONTRACT.md`](DATA_CONTRACT.md)
- `torch` 的 CPU / CUDA 安装方式可能因机器不同需要单独调整

### 2. 典型工作流

准备 A 股数据：

```powershell
python Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py
```

运行当前推荐的 smoke 实验：

```powershell
python Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py --experiments shortT_balanced_v1 --smoke-only
```

运行 `iterA5` 正式训练参考入口：

```powershell
python Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py
```

分析 `iterA5` 模型输出：

```powershell
python Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iterA5_ashare_results.py
```

校验同花顺代理输出：

```powershell
python Finance/03_实验与验证/脚本/测试与临时脚本/validate_iterA5_ths_proxy.py
```

## 仓库地图

- [`Finance/02_核心代码`](Finance/02_核心代码)
  核心源码、工具函数、同花顺映射、历史归档
- [`Finance/03_实验与验证`](Finance/03_实验与验证)
  数据准备脚本、训练 runner、分析与验证脚本
- [`Finance/04_项目文档`](Finance/04_项目文档)
  开发日志、技术文档、实验报告和论文相关材料
- [`.trae`](.trae)
  AI 协作协议、计划和技能说明

## 当前关键入口

- 核心训练器：
  [`Finance/02_核心代码/源代码/khaos/模型训练/train.py`](Finance/02_核心代码/源代码/khaos/模型训练/train.py)
- A 股数据支持：
  [`Finance/02_核心代码/源代码/khaos/数据处理/ashare_support.py`](Finance/02_核心代码/源代码/khaos/数据处理/ashare_support.py)
- A 股样本权重与数据集 profile：
  [`Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py`](Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py)
- 当前主线 runner：
  [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
- `iterA5` 训练参考 runner：
  [`Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py)

## 升级门槛

当前仓库内已经明确记录的版本升级门槛包括：

- `overall_model.composite >= 0.4187`
- `calibrated_ths.test_objective >= 0.4448`
- `60m composite >= 0.4032`

这些门槛目前主要出现在：

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

## GitHub 中有意缺失的内容

为了保持仓库轻量、可读、便于 AI 接手，以下内容默认不上 GitHub：

- 原始行情数据与研究处理数据
- 模型权重、断点、局部实验输出
- 训练日志和中间统计文件
- 根目录媒体素材、音频、分镜图
- 本地上下文数据库和缓存目录

如果未来必须共享大文件，请优先考虑 Git LFS 或独立对象存储，而不是直接把产物塞进源码仓库。
