# KHAOS / PI-KAN：金融时间序列相变探测研究工程

本仓库用于维护一个围绕金融时间序列相变探测构建的研究工程。项目以 PI-KAN 为神经网络主体，以 KHAOS 为物理与符号语义框架，持续联通以下几条主线：

- 物理特征与符号解释层
- PI-KAN / teacher 网络训练层
- TradingView Pine 与同花顺等终端表达层
- 实验、验证与蒸馏映射层
- 毕业设计与论文写作材料

这里的 A 股训练线很重要，但它只是当前最活跃的一条工程分支，不是整个项目的全部定义。

## 研究目标

- 用物理信息增强的方式刻画金融时间序列中的相变、机制切换与极端状态。
- 让 `breakout / reversion` 双核输出在多周期场景下保持清晰语义，而不是退化为普通涨跌预测。
- 在 Python 研究母体、Pine/同花顺终端表达和论文材料之间保持统一叙事。
- 为后续跨平台 truth-engine 演化保留明确的特征、版本与映射边界。

## 系统分层

### 1. 数据层

- 管理本地原始数据、研究处理数据和 `training_ready` 文件。
- 当前活跃工程分支已形成 A 股专用数据支持，但整体目录设计并不局限于 A 股。

### 2. 物理与符号层

- 通过 `Hurst`、`EKF`、`Entropy`、`MLE`、`Compression` 等特征刻画市场状态。
- 负责把“相变”定义为可计算、可验证、可蒸馏的结构，而不是纯粹的经验标签。

### 3. 神经网络层

- 以 PI-KAN / KAN 为核心，训练 `breakout` 与 `reversion` 双核输出。
- **(New in Iter11)** 引入 **证据深度学习 (Evidential Deep Learning, EDL)** 架构，将传统的二分类概率升级为基于 Dirichlet 分布的“证据”与“不确定性 (Uncertainty)”量化。通过自然学习过滤高噪声金融时间序列中的震荡市假阳性，彻底废除人为硬性胜率阈值。
- 当前主干强调多周期输入、局部触发锚点、共享状态门控与约束型训练。

### 4. 终端表达层

- TradingView Pine 与同花顺公式承担终端消费与可视化职责。
- 它们是表达层，不是项目唯一真值来源；复杂推理仍以 Python 侧研究母体为准。

### 5. 实验与验证层

- 负责 runner、ablation、分析脚本、THS 代理校验和阶段性报告。
- 当前 A 股工程主线主要从这里组织训练与回看。

### 6. 文档与论文层

- 沉淀技术方案、开发日志、实验报告、论文路线图和阶段复盘。
- 用于保持研究叙事、工程演进和学术写作之间的一致性。

## 仓库结构

- [`Finance/01_数据中心`](Finance/01_数据中心)
  本地数据目录约定与研究数据入口。
- [`Finance/02_核心代码`](Finance/02_核心代码)
  当前源代码、终端公式、工具箱以及历史归档。
- [`Finance/03_实验与验证`](Finance/03_实验与验证)
  数据准备、训练 runner、结果分析与校验脚本。
- [`Finance/04_项目文档`](Finance/04_项目文档)
  规划、技术文档、开发日志、实验报告和论文材料。
更细的目录说明见 [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md)。

## 当前主要工作线

截至 `2026-04-17`，仓库中最值得关注的工作线包括：

- **(Iter11)** 引入证据深度学习 (EDL) 进行 KHAOS-KAN 的重大架构升级，以自适应不确定性量化取代后处理硬约束。
- A 股与多资产 (Multi-asset) 的模型训练、`shortT balanced` 系列实验及泛化性验证。
- 同花顺公式作为终端消费层的落地与代理验证。
- “Python truth-engine + Pine/THS 消费层”的跨平台架构方案。
- 毕设论文与研究材料的持续整理。

具体状态快照见 [`CURRENT_WORKSTREAMS.md`](CURRENT_WORKSTREAMS.md)。

## 建议阅读路径

1. [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md)
2. [`Finance/04_项目文档/02_开发文档/02_技术文档/项目架构.md`](Finance/04_项目文档/02_开发文档/02_技术文档/项目架构.md)
3. [`Finance/04_项目文档/02_开发文档/02_技术文档/跨平台指标引擎架构方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/跨平台指标引擎架构方案.md)
4. [`Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_A股训练数据选取与构建报告.md`](Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_A股训练数据选取与构建报告.md)
5. [`Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_A股专用网络训练方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_A股专用网络训练方案.md)
6. [`Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_同花顺落地方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_同花顺落地方案.md)
7. [`Finance/03_实验与验证/脚本/README.md`](Finance/03_实验与验证/脚本/README.md)
8. [`Finance/02_核心代码/源代码/khaos/README.md`](Finance/02_核心代码/源代码/khaos/README.md)

## 环境与运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

当前 A 股数据准备入口：

```powershell
python Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py
```

当前推荐的 smoke 入口：

```powershell
python Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py --experiments shortT_balanced_v1 --smoke-only
```

本地数据目录、命名规范与输出边界见 [`LOCAL_DATA_LAYOUT.md`](LOCAL_DATA_LAYOUT.md)。

## GitHub 中有意缺失的内容

本仓库默认不上传以下本地产物：

- 原始行情数据与研究处理数据
- 模型权重、checkpoint 与临时导出图像
- 训练日志与中间统计文件
- 本地实验结果目录
- `.trae`、`.context_db` 等本地 AI / 编辑器协作元数据
- 根目录媒体素材、音频与分镜资源

如果未来必须共享大文件，优先考虑 Git LFS 或独立对象存储，而不是把实验产物直接并入源码仓库。
