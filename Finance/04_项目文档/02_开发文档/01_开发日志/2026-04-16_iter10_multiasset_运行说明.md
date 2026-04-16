# 2026-04-16 iter10 多资产闭环运行说明（训练 → 指标 → 回测 → 报告）

## 1. 数据目录约定

iter10 多资产训练默认从 `<DATA_DIR>/training_ready/` 读取训练文件。

### 1.1 训练文件格式

- 文件类型：CSV
- 必需列：`time, open, high, low, close, volume`
- 推荐命名：`<ASSET>_<SUFFIX>.csv`
  - `SUFFIX` 支持：`5m, 15m, 1h, 4h, 1d`
  - 示例：`BTC_1h.csv`、`SPX_1d.csv`

### 1.2 放置位置

```
<DATA_DIR>/
  training_ready/
    BTC_15m.csv
    BTC_1h.csv
    ETH_15m.csv
    SPX_1d.csv
```

> 若 `training_ready/` 为空，训练脚本会尝试从 `<DATA_DIR>` 下递归扫描 `.csv` 原始文件并自动生成多周期版本写入 `training_ready/`。

## 2. 训练与产物目录

建议把训练产物集中写入 `<RUN_ROOT>/<RUN_ID>/`，其中：

- `<RUN_ID>`：建议使用 `YYYYMMDD_iter10_<market>_<profile>` 的格式
- 产物包含：
  - `epoch_metrics.jsonl`
  - `per_timeframe_metrics.jsonl`
  - `per_asset_metrics.jsonl`
  - `khaos_kan_best.pth` / `khaos_kan_best_gate.pth` / `khaos_kan_resume.pth`
  - `iter10_report.md`

## 3. 关键参数（多资产闭环推荐）

- `--market legacy_multiasset`
- `--assets`：例如 `BTC,ETH,SPX`
- `--timeframes`：例如 `15m,60m,1d`
- `--constraint_profile teacher_feasible_discovery_v1`
- `--score_profile short_t_discovery_guarded_focus`（若想把 space/quality 纳入评分）
- `--kill_keep_public_violation_rate_max 0.25`（可按阶段收紧）

## 4. 口径说明（必须对齐）

### 4.1 public-directional 一致性（可行域约束）

训练过程中会记录：

- `public_below_directional_violation_rate`

该指标用于衡量 public 信号是否在 directional 证据不足时“越界输出”，是迭代的硬门槛之一。

### 4.2 分周期阈值冻结（可复现评估）

- 训练会在 train split 上按 timeframe 拟合 breakout/reversion 阈值
- val/test 汇总指标时固定复用该阈值，避免阈值漂移导致“看起来变好/变差”
- 如需关闭：`--disable_threshold_fit`

## 5. 输出与验收

iter10 的闭环验收分两层：

1) 信号层：多资产 × 多周期的 precision/recall/f1、hard negative、public violation、信号频次、冻结阈值
2) 策略层：基于信号的最小策略回测对比（固定成本假设），报告中与信号层同时展示

