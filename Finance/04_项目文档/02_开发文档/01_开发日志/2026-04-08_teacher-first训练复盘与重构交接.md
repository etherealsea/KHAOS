# 2026-04-08 teacher-first 训练复盘与重构交接

## 背景

- 用户要求继续训练，并把目标明确切换为“提升信号准确度”，尤其要降低当前 `reversion` 信号的误报。
- 用户同时要求给出直接结论：当前信号是否真的能识别波段爆发，以及均值回归的底和顶。
- 本轮工作不是继续解释旧分数，而是实际落地一轮新的精度优先训练，并据结果判断现有训练设计是否需要大范围重构。

## 本轮实际完成的工作

### 1. 先完成了 `shortT_balanced_v1` 正式训练并停在 v1

- 主线仓库同步到 `origin/main`，提交为 `b96a041`。
- 已确认 A 股 teacher-first 主线数据覆盖正常，24 个主标的、5 个周期训练集可用。
- `shortT_balanced_v1` 正式训练完成，并通过 watcher 在 `v1` 结束后阻止了 `v2` 自动继续。

关键结论：

- `shortT_balanced_v1` 是当前仍然最强的已知主线结果。
- 但它并没有解决“高质量信号”问题，尤其没有解决 `reversion` 的过度发信号。

### 2. 新增了一轮精度优先实验 `shortT_precision_v1`

本轮为了响应“提高信号准确度”的要求，新增了一个精度优先实验：

- `Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py`
- `Finance/02_核心代码/源代码/khaos/模型训练/loss.py`
- `Finance/02_核心代码/源代码/khaos/模型训练/train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`

新增内容包括：

- `dataset_profile='shortT_precision_v1'`
- `loss_profile='shortT_precision_v1'`
- `score_profile='short_t_precision_focus'`
- `constraint_profile='teacher_feasible_precision_v1'`
- 自动从 `shortT_balanced_v1_best.pth` 热启动

设计意图：

- 降低 `5m` 的主导权重，进一步抬高 `60m`
- 对 `reversion_hard_negative`、`reversion_event_gap`、`continuation_suppression`、`direction_consistency` 施加更强惩罚
- 在 checkpoint 评分中显式提高 precision 权重，并额外惩罚 oversignal

### 3. `shortT_precision_v1` smoke 已通过

- smoke 能正常出完整日志、`epoch_metrics.jsonl` 和 `per_timeframe_metrics.jsonl`
- 说明代码链路是通的，训练脚本没有语法或运行级别错误

### 4. `shortT_precision_v1` 正式训练已完成

- 正式训练实际跑到 `epoch 4`
- 因连续 4 个 epoch 未达到最小改进阈值而 early stop
- THS 代理校验通过，`all_passed = true`
- 产物已写出到：
  - `Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_precision_v1/`
  - `logs/teacher_first_ashare/shortT_precision_v1/shortT_precision_v1.log`

## 当前训练结果

### A. `shortT_balanced_v1` 仍是当前更强基线

此前已确认的最佳 checkpoint 指标：

- best checkpoint epoch: `15`
- composite: `0.4197367285974938`
- breakout accuracy / precision / recall / f1: `0.5295 / 0.4633 / 0.4793 / 0.4712`
- reversion accuracy / precision / recall / f1: `0.6114 / 0.3963 / 0.6077 / 0.4798`
- direction macro f1: `0.9718`
- `public_below_directional_violation_rate`: `0.7436`

从后续 epoch 17-20 的正式日志看，`v1` 的表现大致稳定在：

- breakout precision: `0.460 ~ 0.463`
- reversion precision: `0.394 ~ 0.396`
- 60m composite: `0.373 ~ 0.380`

这说明：

- `v1` 不是一个“高纯度抓点”的模型
- 它更像是能识别阶段切换区域，而不是精确起爆点或顶部/底部

### B. `shortT_precision_v1` 没有达到预期

正式训练最后一个 epoch 的结果：

- epoch: `4`
- composite: `0.3520387182853421`
- breakout accuracy / precision / recall / f1: `0.5291 / 0.4631 / 0.4821 / 0.4724`
- reversion accuracy / precision / recall / f1: `0.5940 / 0.3774 / 0.5807 / 0.4575`
- direction macro f1: `0.9559`
- 60m composite: `0.3090`
- breakout signal frequency / label frequency: `0.4553 / 0.4373`
- reversion signal frequency / label frequency: `0.4537 / 0.2948`
- `public_below_directional_violation_rate`: `0.9811`

对比 `v1` 可以直接得出：

- breakout precision 几乎没有提升
- reversion precision 明显更差
- reversion 仍严重过度发信号
- 60m 表现明显退化
- 约束相关指标没有收紧，反而更糟

## 这轮训练得出的直接业务结论

### 1. 当前信号仍然不能被视为“可靠识别波段爆发和均值回归底顶”

更准确的说法是：

- `breakout` 只能算中等纯度、中等召回的阶段切换提示
- `reversion` 仍然是高召回但低纯度、且明显过度发信号的噪声型分支
- 模型当前更像在报“这里可能处于转折区间”，而不是“这里就是精确底部/顶部/起爆点”

### 2. 当前主要瓶颈仍然是 `60m`

主线 formal success threshold 对 `60m composite` 的要求是 `0.4032`。

当前已知情况：

- `shortT_balanced_v1` 最好也只到约 `0.3796`
- `shortT_precision_v1` 只有 `0.3090`

这说明目前训练设计并没有真正解决中枢周期的判别质量问题。

### 3. 当前评分机制与用户目标没有完全对齐

虽然本轮新增了 `short_t_precision_focus`，但整体机制仍有结构性问题：

- 训练最优 checkpoint 仍由训练时内部 score 决定
- formal success 又看另一套门槛
- 用户真正关心的是“信号能不能少而准、能不能抓到关键位置”
- 这三者还没有真正统一成同一套优化目标

## 当前确认的关键设计问题

### 1. warm-start 逻辑污染了新实验的 early stop 与 best-checkpoint 判断

这是本轮最关键的问题。

`shortT_precision_v1` 是从 `shortT_balanced_v1_best.pth` 热启动的，但当前训练代码的恢复逻辑会把旧实验里的：

- `best_score`
- `epoch`
- `no_improve_epochs`

一起带入新实验。

而新实验已经更换了：

- `score_profile`
- `loss_profile`
- `constraint_profile`

也就是说：

- 旧实验的 `best_score≈0.4197` 来自旧评分口径
- 新实验自己的 score 只有 `0.352~0.354`
- 它从开始就不可能“超过历史 best”

直接后果：

- 新实验目录中没有生成新的 `shortT_precision_v1_best.pth`
- early stop 很可能不是在一个公平条件下触发的
- 本轮正式训练结果不能被当作一次完全有效的 precision-first 对比实验

这说明下一轮重构时，必须优先修复：

- “跨实验 warm-start 只加载模型权重，不继承旧 score/early-stop 状态”
- 或者至少在 score profile 变化时自动重置 `best_score`、`no_improve_epochs`

### 2. `reversion` 过度发信号不是简单加惩罚就能解决

本轮把 loss、score、sample weight 都往 precision 方向推了，但结果没有改善，说明问题很可能不只在损失系数，而在更上游的设计：

- 标签定义是否过宽
- `reversion` 正负样本边界是否和真实“顶底”概念不一致
- 共享 trunk 与双头输出的耦合方式是否让 `reversion` 天生被趋势信息污染
- 训练时的 threshold/校准流程是否缺失

### 3. 当前约束体系没有把“公共 reversion 输出”真正压到合理区域

`public_below_directional_violation_rate` 在 `shortT_precision_v1` 中约为 `0.9811`，非常高。

这说明：

- 当前 direction branch 与 public reversion branch 的关系式虽然被写进 loss
- 但实际优化后并没有形成有效约束
- 这些约束目前更像“写在目标函数里”，而不是“真正改变了输出结构”

### 4. “训练分数更高”与“信号更好用”仍不是同一件事

当前模型评估还不够直观，缺少用户最关心的信号质量验证：

- 爆发信号提前多少根 K 线出现
- 顶底信号距离局部 extrema 多远
- 不同阈值下 precision / recall 如何变化
- top-k 信号是否更纯
- 连续重复发信号是否被压制

如果只盯着 composite，容易把“仍然噪声很大但还能拿到一般分数”的模型误认为有效。

## 对下一轮 AI 的重构建议

建议下一轮不要继续做小修小补，而是把训练设计当成一轮真正的结构重构。

建议优先级如下：

### 第一优先级：修训练流程正确性

1. 修复 warm-start 逻辑
2. 跨实验热启动时只加载 `model_state_dict`
3. 若 `score_profile` 或 `loss_profile` 变化，必须重置 best score 和 early-stop 状态
4. 让新实验一定能生成自己的 `best checkpoint`

### 第二优先级：重写“真正服务于信号质量”的评估体系

1. 增加 threshold sweep
2. 增加 precision-first 的 best-checkpoint 选择逻辑
3. 单独统计：
   - breakout top-k precision
   - reversion top-k precision
   - 事件提前量
   - 距局部底顶的偏移
   - 去重后信号准确率
4. 把 `60m` 作为单独主目标，而不是仅作为附属时间框架

### 第三优先级：重审标签与多头结构

1. 重新检查 breakout / reversion 的标签定义
2. 评估是否需要：
   - precision-first label
   - 更窄的 event 窗口
   - 更严格的 hard negative 定义
3. 评估是否需要把：
   - “阶段切换提示”
   - “精确顶底/起爆点”
   拆成不同头或不同任务，而不是继续混在同一输出里

### 第四优先级：把“能不能抓到真实关键点”做成可读报告

下一轮 AI 不应只报 composite/F1，而应产出更直观的结论：

- 爆发信号在多少案例中出现在真正放量启动前后若干根内
- 回归信号在多少案例中出现在局部顶底附近
- 不同阈值下每日/每周大致会报多少个信号
- 这些信号的命中率、漏报率和重复率分别如何

## 当前代码状态

本轮仍处于未提交状态，工作区里有以下修改：

- `Finance/02_核心代码/源代码/khaos/数据处理/ashare_dataset.py`
- `Finance/02_核心代码/源代码/khaos/模型训练/loss.py`
- `Finance/02_核心代码/源代码/khaos/模型训练/train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`

这些改动主要用于：

- 新增 `shortT_precision_v1`
- 新增精度优先 score
- 加强 precision 相关 loss / sample weight
- 自动从 `shortT_balanced_v1` 热启动

下一轮 AI 需要自行判断：

- 是在此基础上继续重构
- 还是保留部分改动、重做实验命名与训练入口

## 关键产物路径

- `logs/teacher_first_ashare/shortT_balanced_v1/shortT_balanced_v1.log`
- `Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_balanced_v1/epoch_metrics.jsonl`
- `Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_balanced_v1/per_timeframe_metrics.jsonl`
- `logs/teacher_first_ashare/shortT_precision_v1/shortT_precision_v1.log`
- `Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_precision_v1/epoch_metrics.jsonl`
- `Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_precision_v1/per_timeframe_metrics.jsonl`
- `logs/teacher_first_ashare/_pipeline/mainline_20260407_summary.json`

## 一句话交接结论

当前训练已经证明：单纯通过调权重、调惩罚、调时间框架采样，不能把模型变成一个可靠的“波段爆发 / 顶底识别器”；同时本轮 precision-first 实验还被 warm-start 评分继承问题污染，因此下一轮应该以“重构训练设计与评估体系”为主，而不是继续在旧框架上微调系数。
