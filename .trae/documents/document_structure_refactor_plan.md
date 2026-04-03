# 文档体系规整计划

## 目标
统一整理 `Finance` 目录下的文档结构，解决 `04_项目文档` 与 `04_开发文档` 功能重叠的问题，建立清晰的分类体系。

## 新的目录结构
所有文档将统一存放在 `Finance/04_项目文档/` 下，不再保留 `Finance/04_开发文档`。

结构如下：
- **00_规划与管理/**: 存放项目路线图、任务规划、待办事项等。
- **01_学术论文/**: 存放**开题报告**、论文草稿、参考文献整理等学术相关内容。
- **02_开发日志/**: 存放每日开发日志。
- **03_技术文档/**: 存放系统架构、算法原理、API说明、用户指南等。
- **04_实验报告/**: 存放模型训练报告、回测报告、特征验证报告等。
- **99_归档/**: 存放过时或废弃的文档。

## 执行步骤

### 1. 创建新目录
在 `Finance/04_项目文档/` 下创建以下子目录：
- `00_规划与管理`
- `01_学术论文`
- `02_开发日志`
- `03_技术文档`
- `04_实验报告`
- `99_归档`

### 2. 迁移“学术论文/开题报告”相关文档
将以下文件移动到 `Finance/04_项目文档/01_学术论文/`：
- `Finance/04_开发文档/文档归档/thesis/` 下的所有内容
- `Finance/04_项目文档/2026-03-14_论文提案修改计划.md`
- `Finance/04_开发文档/文档归档/archive/AI_THESIS_ROADMAP.md`
- `Finance/04_开发文档/文档归档/archive/README_AI_THESIS.md`

### 3. 迁移“规划与管理”相关文档
将以下文件移动到 `Finance/04_项目文档/00_规划与管理/`：
- `Finance/04_开发文档/规划文档/` 下的所有内容
- `Finance/04_项目文档/2026-03-14_JessieOBS模式剖析与改造计划.md`
- `Finance/04_项目文档/2026-03-14_指标设计后续任务规划.md`
- `Finance/04_开发文档/文档归档/archive/Graduation_Project_Roadmap.md`

### 4. 迁移“开发日志”
将以下文件移动到 `Finance/04_项目文档/02_开发日志/`：
- `Finance/04_项目文档/2026-03-14_开发日志.md`
- `Finance/04_项目文档/旧文档归档/开发日志_旧.md`
- `Finance/04_开发文档/开发日志/` 下的所有内容
- `Finance/04_开发文档/文档归档/archive/开发工作日志.md`

### 5. 迁移“实验报告”
将以下文件移动到 `Finance/04_项目文档/04_实验报告/`：
- `Finance/04_开发文档/实验报告/` 下的所有内容
- `Finance/04_开发文档/文档归档/reports/` 下的所有内容 (除开题报告外)
- `Finance/04_开发文档/文档归档/PIKAN_MULTI_SCALE_PHYSICS_REPORT.md`
- `Finance/04_开发文档/文档归档/PIKAN_HORIZON_AND_R2_REPORT.md`
- `Finance/04_开发文档/文档归档/PIKAN_TECHNICAL_REPORT.md`

### 6. 迁移“技术文档”
将以下文件移动到 `Finance/04_项目文档/03_技术文档/`：
- `Finance/04_开发文档/文档归档/KHAOS_USER_GUIDE.md`
- `Finance/04_开发文档/文档归档/PIKAN_TO_INDICATOR_MAPPING.md`
- `Finance/04_开发文档/文档归档/architecture/` 下的所有内容
- `Finance/04_开发文档/文档归档/archive/KHAOS_Mathematical_Model_Reference.md`
- `Finance/04_开发文档/文档归档/archive/KHAOS_PIKAN_能力与指标设计指南.md`
- `Finance/04_开发文档/文档归档/archive/KHAOS_TradingView手动验证模板.md`
- `Finance/04_开发文档/文档归档/archive/KHAOS_中文使用与判读指南.md`
- `Finance/项目架构.md`

### 7. 清理与归档
- 将 `Finance/04_开发文档` 下剩余的杂项移动到 `Finance/04_项目文档/99_归档/`。
- 删除空的 `Finance/04_开发文档` 目录。
- 删除空的 `Finance/04_项目文档/旧文档归档` 目录 (内容已迁移)。

## 最终确认
告知用户开题报告位于：`Finance/04_项目文档/01_学术论文/`
