# iter6 优化与评估计划

## 1. 先解释当前汇报里几个关键名词

### Accuracy

- 含义：整体判断正确的比例。
- 公式：`(TP + TN) / 全部样本`
- 在当前任务里，它表示模型把“事件 / 非事件”总体上分对了多少。
- 局限：如果非事件样本很多，Accuracy 可能看起来不错，但不代表真正擅长抓事件。

### Precision

- 含义：模型报出来的正样本里，有多少是真的。
- 公式：`TP / (TP + FP)`
- 对 breakout 核最直观的解释是：模型一旦提示“这是爆发候选”，它有多大概率不是误报。
- Precision 高，说明信号更“干净”。

### Recall

- 含义：所有真实正样本里，模型抓到了多少。
- 公式：`TP / (TP + FN)`
- 对 reversion 核最直观的解释是：所有真正发生有效回归的地方，模型漏掉了多少。
- Recall 高，说明模型更“不容易漏报”。

### F1

- 含义：Precision 和 Recall 的综合平衡指标。
- 公式：`2 * Precision * Recall / (Precision + Recall)`
- 适合用来回答“这个核综合上到底好不好”，因为它不会只奖励高 Precision，也不会只奖励高 Recall。
- 如果一个核 Precision 很高但 Recall 很低，或者反过来，F1 都不会特别高。

### 事件命中率

- 含义：在真实事件样本里，被模型判为正样本的比例。
- 在当前实现里，它和基于选定阈值计算出的 Recall 非常接近，可以理解成“真实事件抓到了多少”。

### 伪信号率

- 含义：在 hard negative 样本里，被模型判为正样本的比例。
- 对 breakout 核尤其重要，因为这类样本最像“假启动 / 假突破”。
- 这个值越低越好。

## 2. 对 iter5 当前结果的解释

- breakout 核当前更偏“高质量筛选器”：
  - Precision 更高
  - F1 更高
  - 伪信号率更低
- reversion 核当前更偏“广覆盖探测器”：
  - Recall 更高
  - 更容易抓到真实回归
  - 但误报仍偏多，Precision 还不够高
- 这说明两个核已经形成可解释分工：
  - breakout 更适合做高质量爆发提示
  - reversion 更适合做先提示、再确认的回归监测

## 3. 是否需要使用更多 epoch

我的判断：**需要尝试更多 epoch，但不能只是盲目从 3 提到更大值，而要和更稳健的选模/早停一起做。**

理由如下：

- iter5 的 best checkpoint 出现在第 3 个 epoch，而不是第 1 或第 2 个 epoch。
- 当前验证指标仍是持续改善的：
  - `Val Loss: 3.6513 -> 3.6047 -> 3.5915`
  - `Composite: 0.3370 -> 0.3590 -> 0.3653`
- 这说明模型在第 3 个 epoch 还没有明显进入“继续训练只会变差”的阶段。

因此，iter6 不建议继续固定 3 个 epoch，而建议改为：

- 默认训练上限提高到 **5~6 个 epoch**
- 同时引入 **early stopping / patience**
- best checkpoint 仍按全局 epoch 汇总指标保存

这样做的好处是：

- 如果第 4~5 个 epoch 继续变好，就能继续吃到收益
- 如果开始过拟合，也能及时停住，而不是硬跑完

## 4. iter6 的核心优化目标

iter6 不再以“继续抬高单个 corr”为主目标，而是明确追求下面三件事：

### 目标 A：继续压低 breakout 的伪信号率

- 重点看 breakout 的 hard negative 误报是否继续下降
- 目标不是让 breakout 变得极度激进，而是保持其“更干净”的优势

### 目标 B：提升 reversion 的 Precision

- 当前 reversion 的主要问题不是抓不到，而是报得偏多、确认不够
- iter6 应把重点放在“让回归核更像确认器”，而不是只做宽松召回

### 目标 C：让最终汇报更接近实战语言

- 训练日志继续保留 corr / gap
- 但结果汇报优先展示：
  - Accuracy
  - Precision
  - Recall
  - F1
  - 事件命中率
  - 伪信号率
- 并对 breakout / reversion 给出直白判断：
  - 谁更适合做高质量提示
  - 谁更适合做广覆盖监测

## 5. iter6 计划执行步骤

### 步骤 1：训练配置升级

- 将 epoch 上限从 3 提高到 5~6。
- 增加 early stopping / patience 机制。
- 保留现有全局 epoch 汇总选模。

### 步骤 2：reversion 进一步确认化

- 检查是否把 reversion 的阈值选择与 hard negative 区分进一步拉开。
- 重点加强“确认后再报”的倾向，减少高召回带来的低 Precision。

### 步骤 3：保留 breakout 的干净特性

- 不让 breakout 为了追 Recall 而明显牺牲 Precision。
- 持续观察 breakout 的：
  - Precision
  - F1
  - 伪信号率

### 步骤 4：增强结果分析脚本

- 在 iter6 结果分析脚本中，直接输出 breakout / reversion 的直观评估表。
- 增加“哪个核更像高质量提示器 / 广覆盖监测器”的自动文字总结。

### 步骤 5：执行 iter6 全量训练

- 新建独立 iter6 训练入口、日志文件、权重目录。
- 不覆盖 iter5 产物。

### 步骤 6：训练结束后汇报

- 汇报内容固定包括：
  - 训练了几个 epoch
  - 是否有效收敛
  - breakout 核表现
  - reversion 核表现
  - 与 iter5 的对比
  - 下一步是否应该转向 Pine 映射

## 6. 预期结果

- 如果 iter6 成功：
  - breakout 的 Precision / F1 继续维持或略升
  - breakout 伪信号率继续下降
  - reversion 的 Precision 提升，Recall 尽量不明显下滑
  - 全局 composite 继续高于 iter5
- 如果 iter6 失败：
  - 可能表现为第 4~5 个 epoch 指标不再提升甚至退化
  - 这种情况下应保留 best checkpoint，并停止继续加 epoch

## 7. 我的建议

- 我建议继续做 iter6。
- 但这次的关键不是“更暴力地训练”，而是：
  - **适度增加 epoch**
  - **加早停**
  - **重点提升 reversion 的确认能力**
  - **继续用更直观的指标判断两个核是否真的变好**
