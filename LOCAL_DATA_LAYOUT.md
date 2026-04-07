# 本地数据布局说明

本仓库默认不上传完整数据集。本文档说明当前项目在本地运行时采用的数据目录约定、文件命名和输出边界。

## 本地数据根目录

当前 A 股相关 runner 预期数据位于：

- [`Finance/01_数据中心/03_研究数据`](Finance/01_数据中心/03_研究数据)

其中常用子目录包括：

- `research_raw/ashare/imports`
- `research_raw/ashare/normalized`
- `research_processed/training_ready/ashare`

## `training_ready` 文件命名

当前训练流程通过文件后缀识别周期，对应关系为：

- `5m -> *_5m.csv`
- `15m -> *_15m.csv`
- `60m -> *_1h.csv`
- `240m -> *_4h.csv`
- `1d -> *_1d.csv`

例如：

- `600036_5m.csv`
- `600036_1h.csv`
- `600036_1d.csv`

## 必备字段

标准化后的 OHLCV 文件至少需要包含：

- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`

可选保留字段：

- `amount`
- `turnover`
- `adj_factor`

当前加载器已支持部分中英文别名向上述标准字段归一。

## 当前默认时间切分

活跃 A 股支持模块中的默认边界为：

- train end: `2023-12-31`
- validation end: `2024-12-31`
- test start: `2025-01-01`

对应实现见：

- [`Finance/02_核心代码/源代码/khaos/数据处理/ashare_support.py`](Finance/02_核心代码/源代码/khaos/数据处理/ashare_support.py)

## 当前活跃周期集合

仓库内正在使用的周期集合主要有两组：

- 核心集合：`15m`, `60m`, `240m`, `1d`
- 短周期增强集合：`5m`, `15m`, `60m`, `240m`, `1d`

其中：

- `iterA5_ashare` 使用了包含 `5m` 的增强集合。
- `teacher-first` 系列实验也会用到 `5m`，但不一定以相同方式参与选模。

## 数据获取入口

当前活跃脚本支持在本地 CSV 不完整时，通过公开接口补齐部分 A 股数据。常用入口包括：

- [`Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py`](Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

## 保留在本地的输出

真实训练或校验时，以下内容通常会在本地生成，但默认不进入 Git：

- coverage 报告
- 训练日志
- checkpoint 与模型权重
- epoch 级与分周期统计快照
- THS 对齐与代理评估输出

典型本地目录包括：

- `logs/`
- `Finance/02_核心代码/模型权重备份/`
- `Finance/models/`

## 实用边界

GitHub 上保留的是目录结构、脚本、文档和约定；真实数据资产、训练产物与本地实验输出默认仍留在本地管理。
