# 2026-04-07 teacher-first mainline 训练执行计划（A股）

更新日期：`2026-04-07`

> 目的：在**不把大体量数据提交到 Git** 的前提下，让具备本地权限的执行方（Codex/你本地机器）按统一口径完成：
> **数据准备/覆盖校验 → smoke gate → formal 训练 → THS 可映射硬验收 → 产物汇总**。

本计划与以下仓库文档一致：
- `KHAOS_A股训练数据选取与构建报告.md`
- `CURRENT_WORKSTREAMS.md`

---

## 0. 关键结论（你需要先记住的硬约束）

### 0.1 数据口径（来自《数据选取与构建报告》）
- 市场限定：**纯 A 股**
- 标的池：**24 主标的 + 6 备选**
- 周期集合（当前活跃工程口径）：**`5m / 15m / 60m / 240m / 1d`**
- 时间切分：严格顺序切分  
  - Train：`<= 2023-12-31`  
  - Val：`2024-01-01 ~ 2024-12-31`  
  - Test：`>= 2025-01-01`
- 文件命名映射（训练逻辑识别规则）：
  - `5m  -> *_5m.csv`
  - `15m -> *_15m.csv`
  - `60m -> *_1h.csv`
  - `240m -> *_4h.csv`
  - `1d  -> *_1d.csv`
- 必需字段：`time, open, high, low, close, volume`

> 注意：`240m` 不是“自然时钟 4 小时分桶”，而是**按交易日会话聚合**（避免午休断档切裂日内结构）。

### 0.2 升级硬门槛（来自 CURRENT_WORKSTREAMS）
任何候选版本要被视为“可升级”，必须同时满足：
- `overall_model.composite >= 0.4187`
- `calibrated_ths.test_objective >= 0.4448`
- `60m composite >= 0.4032`

并且 **THS proxy 校验必须通过**（本仓库已将其固化为 formal 阶段硬验收）。

---

## 1. 数据放置与目录结构（不走 Git 提交）

本训练流程默认以“仓库内相对路径”读取数据，执行方需要在本地把数据放到这些目录（没有就创建）：

### 1.1 原始导入目录（可选）
如果你有同花顺/券商导出的原始 CSV，可放到：

`Finance/01_数据中心/03_研究数据/research_raw/ashare/imports/`

脚本会尝试标准化并生成最终训练目录。

### 1.2 最终训练目录（推荐直接提供）
如果你已经生成了 `training_ready`（推荐），请直接放到：

`Finance/01_数据中心/03_研究数据/research_processed/training_ready/ashare/`

该目录下应包含（至少）：
- 24 主标的完整的 `*_5m.csv *_15m.csv *_1h.csv *_4h.csv *_1d.csv`
- 若主标的覆盖不足，可用 6 备选补位（但不能静默残缺主池）

---

## 2. 环境准备

### 2.1 Python 依赖安装

Linux/macOS（bash）：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows（PowerShell）：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 如果你不需要 GPU，可安装 CPU torch（可选）：  
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 3. 一键主线执行（推荐）

### 3.1 主线 pipeline（smoke → formal）

在仓库根目录运行：
```bash
python Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py --pipeline mainline_20260407
```

该 pipeline 会按顺序执行：
1) 覆盖校验（coverage report，不足则直接失败）
2) `iterA4_refresh` smoke
3) `shortT_balanced_v1` smoke
4) `shortT_balanced_v2` smoke
5) `iterA4_refresh` formal（baseline）
6) 对通过 smoke gate 的 balanced 候选进入 formal
7) 每个 formal 结束后：**THS proxy 校验（必须通过）**

> 可选参数：
> - `--resume`：断点续训
> - `--skip-itera5-freeze`：跳过 iterA5 快照冻结（节省时间）
> - `--skip-ths-validation`：跳过 THS 校验（只用于调试，不建议用于正式候选）

---

## 4. 训练产物与日志位置

### 4.1 Pipeline 汇总
`logs/teacher_first_ashare/_pipeline/mainline_20260407_summary.json`

### 4.2 Smoke / Formal 权重目录
- Smoke：`Finance/02_核心代码/模型权重备份/teacher_first_ashare_smoke/<experiment>/`
- Formal：`Finance/02_核心代码/模型权重备份/teacher_first_ashare/<experiment>/`

### 4.3 每轮训练的核心诊断文件
每个实验目录（smoke/formal）应包含：
- `epoch_metrics.jsonl`（含 promotion scoreboard、violation、分支指标）
- `per_timeframe_metrics.jsonl`（按周期拆分的指标）

### 4.4 THS 校验（硬验收）
formal 阶段会输出类似：
```json
{"ths_proxy_validation": {"all_passed": true, "mismatches": {}, ...}}
```
若 `all_passed=false`，该次训练会被视为不可升级候选（pipeline 直接报错或 gate 失败）。

---

## 5. 验收（执行方需要给出的最终结论）

执行完成后，请在提交报告/同步信息里明确给出：

1) coverage 是否足够（24 主标的是否完整；若有补位，补位清单是什么）
2) `iterA4_refresh` baseline 的 overall / 60m / THS objective
3) `shortT_balanced_v1` 与 `v2` 各自的：
   - 是否通过 smoke gate
   - formal best checkpoint 的 three-threshold 是否全部达标
   - THS proxy 是否通过
4) 如果未达标：最主要拖累项（优先看 `60m` 与 violation 指标）

---

## 6. 常见失败与排查

### 6.1 coverage 不足
- 直接看 coverage 报告（`logs/teacher_first_ashare/teacher_first_ashare_coverage.md`）
- 按报告补齐缺失组合（缺失的标的/周期/时间范围）

### 6.2 周期命名不对导致无法识别
- 确认 `*_1h.csv` 会被识别为 `60m`，`*_4h.csv` 会被识别为 `240m`
- 文件至少包含：`time/open/high/low/close/volume`

### 6.3 THS proxy 不通过
- 先运行：
```bash
python Finance/03_实验与验证/脚本/测试与临时脚本/validate_iterA2_ths_proxy.py
```
- 若常量 mismatch，先对齐 `khaos/同花顺公式/KHAOS_THS_CORE.txt` 与 `khaos/同花顺公式/params.json`

