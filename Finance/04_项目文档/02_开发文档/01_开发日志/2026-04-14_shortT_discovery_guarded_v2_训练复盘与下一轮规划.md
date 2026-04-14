# 2026-04-14 shortT_discovery_guarded_v2 训练复盘与下一轮规划

## 1. 本轮已完成的工程推进

这一轮不是只改了 loss 权重，而是把短周期 teacher 训练链路做了一次系统级修正，并在此基础上推进了 discovery guarded 方向。

### 1.1 训练流水线修正已经落地

- `train.py` 已切换为全局混合训练：
  - train split 不再按 runtime record 串行跑完一个文件再跑下一个；
  - 改为把所有 train dataset 组装进全局 `ConcatDataset`；
  - 通过 sampler 维持时间框架 cap 与配额平衡；
  - batch 内开始混合不同标的与时间周期。
- train / validation loss 聚合已经改为按样本数加权，而不是按 batch 数做简单平均。
- scheduler stepping、checkpoint 比较、日志写出，已经统一使用修正后的验证 loss 语义。
- 训练与评估阶段的非有限值处理已补齐：
  - 非有限预测直接跳过；
  - 非有限 loss / rank / reg / logging dict 直接跳过；
  - 非有限梯度与非有限裁剪结果会跳过 step，并累积 skip counter；
  - gradient clipping 保持显式开启。

### 1.2 因果数据链路已修正

- `ashare_dataset.py` 中 sigma / volatility 预处理不再使用 `.bfill()`。
- 现统一为严格因果链：
  - `rolling std`
  - `ffill`
  - `fillna(0.0)`
  - `minimum clamp`
- 同一套因果语义已经覆盖训练数据和缓存生成链路，减少了边界样本的前视偏差风险。

### 1.3 discovery guarded 方向已形成可运行方案

- `data_loader.py` 新增了 discovery target 生成与 train-fit threshold 机制：
  - `build_breakout_discovery_targets`
  - `fit_breakout_discovery_thresholds`
  - `build_reversion_discovery_targets`
  - `fit_reversion_discovery_thresholds`
- `ashare_dataset.py` 中已加入：
  - `shortT_discovery_v1`
  - `shortT_discovery_guarded_v1`
  - `shortT_discovery_guarded_v2`
- guarded 版本的核心思路不是回退到死板事件标签，而是让模型优先学习：
  - 未来位移空间；
  - 路径持续性与 terminal confirmation；
  - continuation excursion；
  - imbalance alignment；
  - signal quality / signal space。
- 同时又通过 train split 拟合阈值并复用于 val/test，避免验证阶段阈值跟着样本漂移。

### 1.4 checkpoint 与评估逻辑继续前移到“精度优先 + 空间质量”

- `train.py` 已加入 precision-first / discovery score profile：
  - `short_t_precision_focus`
  - `short_t_discovery_focus`
  - `short_t_discovery_guarded_focus`
- guarded scoring 已支持 aux-head-aware 组合打分，而不是只看主头原始分数。
- 训练日志中已补入：
  - `breakout_oversignal`
  - `reversion_oversignal`
  - `breakout_signal_space_mean`
  - `reversion_signal_space_mean`
  - `breakout_signal_quality_mean`
  - `reversion_signal_quality_mean`
- runner 已补齐：
  - `shortT_balanced_v3`
  - `shortT_dual_precision_v1`
  - `shortT_discovery_v1`
  - `shortT_discovery_guarded_v1`
  - `shortT_discovery_guarded_v2`

## 2. 已完成的验证

### 2.1 静态检查与测试

- 核心训练文件 `py_compile` 已通过。
- 现有针对重构链路的测试已通过：
  - `Finance/03_实验与验证/脚本/test_horizon_train_refactor.py`
  - `Finance/03_实验与验证/脚本/test_runtime_dataset_cache.py`

### 2.2 smoke 训练

- `shortT_discovery_guarded_v1` smoke 已通过。
- `shortT_discovery_guarded_v2` smoke 已通过。

smoke 给出的第一手结论是：

- discovery guarded 链路在工程上是稳定的；
- `15m/60m` 短周期偏置确实生效；
- 但 `public_below_directional_violation_rate` 从一开始就偏高，说明 public 头与 directional floor 的耦合问题仍然在。

## 3. `shortT_discovery_guarded_v2` formal 结果

### 3.1 执行状态

- formal 训练已经完成，不在运行中。
- 本轮以 early stop 提前结束在 `epoch 9`。
- 输出目录：
  - `Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_discovery_guarded_v2`
- 训练日志：
  - `logs/teacher_first_ashare/shortT_discovery_guarded_v2/shortT_discovery_guarded_v2.log`

### 3.2 关键结果

- best composite 出现在 `epoch 4`
  - `composite_score = 0.393952`
  - `val_loss = 7.748262`
- final `epoch 9`
  - `train_loss = 6.738619`
  - `val_loss = 7.889984`
  - `composite_score = 0.387722`
  - `breakout_precision = 0.148109`
  - `reversion_precision = 0.149773`
  - `signal_frequency.breakout = 0.090003`
  - `signal_frequency.reversion = 0.090046`
  - `60m composite = 0.364410`
  - `public_below_directional_violation_rate = 0.525431`
  - `score_veto.passed = false`

### 3.3 训练曲线给出的结论

- 最好分数出现在 `epoch 4`，之后没有再实质提升。
- `val_loss` 在 `epoch 1` 最低，之后整体没有形成稳定收敛。
- `public_below_directional_violation_rate` 反而随着训练推进持续恶化：
  - `epoch 1 = 0.2588`
  - `epoch 4 = 0.3632`
  - `epoch 5 = 0.3477`
  - `epoch 6 = 0.4340`
  - `epoch 7 = 0.4799`
  - `epoch 8 = 0.5013`
  - `epoch 9 = 0.5254`

## 4. 当前失败原因判断

### 4.1 这轮已经不是“信号发太多”导致的低准确率

- score timeframe 上的平均信号频率大约在 `0.07 ~ 0.09`，并不高。
- `breakout_oversignal` 和 `reversion_oversignal` 基本为 `0.0`。
- 因此当前 precision 很差，不是传统 balanced/F1 版本里那种“阈值太松、信号过发”的主问题。

### 4.2 短周期偏置确实修了一部分 `60m`，但没有修到“公共信号质量”

- `60m composite` 已经能稳定站到 `0.33` 以上，本轮 final 达到 `0.3644`。
- 这说明把模型做得更偏短期并不是完全无效。
- 但 `60m` 的 breakout / reversion precision 仍然没有进入可用区间，说明模型虽然抓到了一些“有波动空间”的点位，却没有把这些点位稳定转化成高精度公共信号。

### 4.3 最大瓶颈已经收敛到 directional-public 耦合失效

- 当前最糟糕的指标不是 `signal_frequency`，而是 `public_below_directional_violation_rate`。
- 这表示 public 头越来越多地在 directional floor 不足时发出信号。
- 也就是说：
  - discovery 头在学“位移空间”；
  - 但方向子头没有同步提供足够强的可解释支撑；
  - 结果是模型学会了一部分空间感，却没有学会把它锚定到更可靠的方向结构上。

### 4.4 loss 不收敛，本质上不是训练管线坏了，而是目标之间开始打架

- 当前 loss 链路已经比之前更干净：
  - 有 sample-weighted loss；
  - 有非有限值保护；
  - 有全局 mixed train；
  - 有因果 sigma；
  - 有 curriculum。
- 但 formal 结果仍然说明：
  - discovery / space 目标在继续被优化；
  - directional consistency 与 public feasibility 没有同步跟上；
  - 所以 loss 和最终精度开始出现明显错位。

## 5. 对下一轮的规划

下一轮不应继续只做“更短周期、更自由”的加码，而应直接修 directional-public 的结构耦合。

### 5.1 主方向

- 保留 discovery / guarded 的自发现路线，不退回纯手工事件体系。
- 继续坚持“模型学到的点位必须对应足够未来位移空间”。
- 但必须把 public 信号重新压回 directional 证据之上，避免空间感和公共输出脱节。

### 5.2 最优先的工程动作

- 在 loss 与 constraint 上继续加强 public 相对 directional floor 的一致性。
- 在 guarded score / checkpoint veto 中进一步提高 `public_below_directional_violation_rate` 的惩罚权重。
- 必要时让 public 头在 directional 子头确认不足时被显式抑制，而不是自由放行。
- 保留短周期主评分聚焦 `15m/60m` 的设定，但不再单独依赖时间框架偏置去解释低准确率。

### 5.3 当前不建议优先做的事

- 不建议马上继续单纯加大 `15m/60m` 的权重。
- 不建议在还没修 directional-public 耦合前，再追加更激进的 discovery 自由度。
- 不建议现在就把问题简单归咎于“训练没跑够 epoch”。

## 6. 本次记录对应代码

- 数据目标与 discovery 阈值：
  - `Finance/02_核心代码/源代码/khaos/数据处理/data_loader.py`
- 数据集、因果 sigma、profile 权重：
  - `Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py`
- 训练主流程、mixed train、指标与 veto：
  - `Finance/02_核心代码/源代码/khaos/模型训练/train.py`
- loss preset、constraint 与 curriculum：
  - `Finance/02_核心代码/源代码/khaos/模型训练/loss.py`
- 实验入口：
  - `Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`

## 一句话结论

- 训练流水线修正已经完成，discovery guarded 方向也已经真正跑起来了。
- 但 `shortT_discovery_guarded_v2` 证明：当前瓶颈不再是“能不能让模型发现有位移空间的点”，而是“如何让这些点被 directional 结构确认后再变成高精度公共信号”。
