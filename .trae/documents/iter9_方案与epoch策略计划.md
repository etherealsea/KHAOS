# iter9 方案与 epoch 策略计划

## 1. 现状判断

- iter8 已完整结束，最终最优 checkpoint 出现在 Epoch 9，而不是最后一轮；Epoch 10 仍接近最优，但没有进一步刷新综合评分。
- 从 epoch 汇总轨迹看，模型在 Epoch 1 到 Epoch 9 持续改进，但 Epoch 6 之后增幅已经明显放缓，说明当前状态更接近“仍有小幅上升空间，但已进入平台区边缘”，而不是“10 轮远远不够”。
- 训练主程序已经内置 `ReduceLROnPlateau` 调度器，因此后续是否增加 epoch，不应理解为盲目拉长训练，而应理解为：是否给降学习率后的后期细化阶段更多生存空间。

## 2. 对“是否继续增加 epoch”的结论

### 结论

- **需要适度增加，但不建议大幅增加。**
- iter9 建议把最大 epoch 从 `10` 调整到 **14~16**，同时保留早停，并把 `early_stop_patience` 从 `3` 调整到 **4**。

### 原因

- 如果完全不增加 epoch，可能会截断掉降学习率后仍能带来小幅改善的尾段。
- 如果直接大幅增加到 20 轮以上，当前证据并不支持，因为 iter8 在 Epoch 6 之后已经呈现缓慢收敛，收益很可能明显递减。
- 因此更合理的策略是：
  - 给后期收敛多留 4~6 轮空间；
  - 用早停限制无效训练；
  - 用 epoch 汇总 best checkpoint 继续防止被单文件局部高分劫持。

## 3. iter9 的核心目标

- 不再把“先救活 reversion”当成主目标，因为 iter8 已经证明 `horizon=10` 下 reversion 可训练性已恢复。
- iter9 应把重点转向两个更明确的问题：
  - **提升 breakout 在中期 horizon 下的干净度**，尽量压低伪信号率；
  - **继续提升 reversion 的确认能力**，尤其是 Precision，而不是只维持高召回。

## 4. iter9 的具体实施步骤

### Step 1：锁定 iter8 为对照基线

- 固化 iter8 的关键对照指标，至少包含：
  - best epoch
  - val loss / composite
  - breakout / reversion corr
  - breakout / reversion gap
  - breakout / reversion 的 Accuracy / Precision / Recall / F1 / 事件命中率 / 伪信号率
- 后续 iter9 的所有判断，都必须以 iter8 为直接对照，而不是只看单轮绝对值。

### Step 2：调整训练轮数与早停策略

- 将 iter9 的最大训练轮数改为 **14~16**。
- 将 `early_stop_patience` 提升到 **4**，`early_stop_min_delta` 暂时维持不变，先观察是否能让学习率衰减后的尾段产生稳定收益。
- 保留当前全局 epoch 汇总选模逻辑，不改为按单文件即时保存。

### Step 3：优化 breakout 的伪信号约束

- 围绕 breakout 的 hard negative 继续做定向约束，而不是继续抬 breakout target 本身。
- 优先检查：
  - 中期 horizon 下 `continuation_release` 对真假 breakout 的区分是否仍偏弱；
  - hard negative 规则是否对“先冲一下但不能持续”的样本惩罚还不够；
  - breakout 的事件阈值和 hard negative 阈值之间是否还存在过宽重叠带。
- 若需要改动，优先做“小幅收紧真假 breakout 分离边界”，避免再次破坏当前已恢复的双核平衡。

### Step 4：优化 reversion 的确认能力

- iter8 已恢复 reversion 可训练性，因此 iter9 不应再主要通过“继续放松标签”来换取更高 recall。
- 应重点检查：
  - reversion event 与 reversion hard negative 的分离边界；
  - terminal confirmation 与 reversal quality 的联合门槛是否还能更偏向“确认后再计入正例”；
  - 当前高 recall 是否有一部分来自过宽的准入范围。
- 优化方向应优先服务于 Precision 的提升，并保持 Recall 不出现大幅回落。

### Step 5：单独评估“更多 epoch 是否真的有效”

- iter9 完整训练结束后，必须专门回答一个问题：**新增的 4~6 轮训练空间，究竟有没有带来稳定收益。**
- 判断标准不是“最后一轮数值是否更高”，而是：
  - best checkpoint 是否出现在 10 轮之后；
  - composite 是否明显超过 iter8；
  - breakout / reversion 的 F1 是否至少有一侧出现清晰改善；
  - 伪信号率是否没有因为训练更久而重新恶化。

## 5. 本轮暂不建议做的事

- 不建议在 iter9 同时大改模型结构、损失函数主框架与标签体系，否则将无法判断“收益来自更多 epoch 还是来自别的改动”。
- 不建议把 epoch 一次性拉到很大，因为当前日志更像是“后期缓慢收敛”，不是“明显未训练够”。
- 不建议再次走 iter7 那种靠显著收紧标签来换纯度的路线，避免重新触发正例塌缩。

## 6. 预期交付

- 一个以 iter8 为基线、以“适度增加 epoch + 定向优化双核确认边界”为核心的 iter9 训练方案。
- 一次能明确回答“10 轮是否真的不够”的对照实验，而不是继续依赖直觉判断。
