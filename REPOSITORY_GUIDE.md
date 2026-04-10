# 仓库导览

这份文档用于说明仓库内部各目录的职责、推荐阅读顺序，以及哪些内容属于源码仓库、哪些内容只保留在本地。

## 推荐阅读路径

### 理解项目整体

1. [`README.md`](README.md)
2. [`Finance/04_项目文档/00_规划与管理/Graduation_Project_Roadmap.md`](Finance/04_项目文档/00_规划与管理/Graduation_Project_Roadmap.md)
3. [`Finance/04_项目文档/02_开发文档/02_技术文档/项目架构.md`](Finance/04_项目文档/02_开发文档/02_技术文档/项目架构.md)
4. [`Finance/04_项目文档/01_学术论文/README_AI_THESIS.md`](Finance/04_项目文档/01_学术论文/README_AI_THESIS.md)

### 理解跨平台设计

1. [`Finance/04_项目文档/02_开发文档/02_技术文档/跨平台指标引擎架构方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/跨平台指标引擎架构方案.md)
2. [`Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_同花顺落地方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_同花顺落地方案.md)
3. [`Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_TO_INDICATOR_MAPPING.md`](Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_TO_INDICATOR_MAPPING.md)

### 理解当前 A 股主线

1. [`CURRENT_WORKSTREAMS.md`](CURRENT_WORKSTREAMS.md)
2. [`Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_A股训练数据选取与构建报告.md`](Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_A股训练数据选取与构建报告.md)
3. [`Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_A股专用网络训练方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_A股专用网络训练方案.md)
4. [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md)
5. [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md)
6. [`Finance/03_实验与验证/脚本/README.md`](Finance/03_实验与验证/脚本/README.md)

### 理解代码主体

1. [`Finance/02_核心代码/源代码/khaos/README.md`](Finance/02_核心代码/源代码/khaos/README.md)
2. [`Finance/02_核心代码/源代码/khaos/核心引擎/physics.py`](Finance/02_核心代码/源代码/khaos/核心引擎/physics.py)
3. [`Finance/02_核心代码/源代码/khaos/模型定义/kan.py`](Finance/02_核心代码/源代码/khaos/模型定义/kan.py)
4. [`Finance/02_核心代码/源代码/khaos/模型训练/train.py`](Finance/02_核心代码/源代码/khaos/模型训练/train.py)
5. [`Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py`](Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py)

## 目录地图

### `Finance/01_数据中心`

- 定义本地数据布局，而不是直接承载完整数据资产。
- 仓库中保留结构约定与说明文档，真实数据默认保留在本地。

### `Finance/02_核心代码`

- `源代码/khaos/` 是当前维护中的主代码包。
- `旧代码归档/` 保留历史实现与旧框架，不是默认修改入口。
- `模型权重备份/` 是本地实验产物区，不属于 GitHub 源码仓库内容。

### `Finance/03_实验与验证`

- `脚本/` 是训练、分析、验证与数据准备的入口层。
- `测试与临时脚本/` 名字带有历史痕迹，但其中仍包含当前主线 runner。
- `pipeline/` 更多承担通用研究、旧管线与历史实验职责。

### `Finance/04_项目文档`

- `00_规划与管理/` 记录路线图、方案和设计演化。
- `01_学术论文/` 保存毕设与论文相关材料。
- `02_开发文档/` 包含开发日志与技术文档，是理解项目演化的关键区域。
- `04_实验报告/` 保存阶段性结果汇总与分析。
- `99_归档/` 保存旧资料与历史说明。

## 使用仓库时需要特别注意

- `测试与临时脚本/` 不等于“废弃脚本区”，当前主线训练入口仍在其中。
- `同花顺公式/` 和 Pine Script 代码属于终端表达层，不应被误读为项目唯一真值来源。
- 当前最活跃的工作线是 A 股 teacher 训练，但项目整体仍包含跨平台输出与论文研究两条长期主线。
- 历史文档中的术语和路径可能存在阶段性差异，阅读时应结合当前文件结构交叉校验。

## 本地保留、不进入 GitHub 的内容

- 行情原始数据与研究处理数据。
- 训练日志、coverage 报告和大部分中间统计文件。
- checkpoint、模型权重和大部分实验输出。
- `.trae`、`.context_db` 等本地 AI / 编辑器协作元数据。
- 根目录媒体素材与临时音视频资源。

本地数据与产物布局见 [`LOCAL_DATA_LAYOUT.md`](LOCAL_DATA_LAYOUT.md)。
