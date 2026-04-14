# 当前工作线与研究快照

更新日期：`2026-04-14`

## 项目整体定位

KHAOS / PI-KAN 是一个研究工程，而不是单一训练仓库。当前仓库中并行维护的内容包括：

- 物理特征与相变语义建模
- PI-KAN / KAN teacher 网络训练
- Pine / 同花顺终端表达
- 跨平台 truth-engine 设计
- 毕设论文与开发材料

当前最活跃的工程分支是 A 股专用训练与终端落地，但它只是整体体系中的一个当前主线。

## 已确认的公开基线

基于 `2026-04-07` 的阶段复盘，当前可明确引用的公开基线为：

- `iterA3_ashare overall_model.composite = 0.4187`
- `iterA4_ashare overall_model.composite = 0.3906`
- `iterA5_ashare` 在中途复盘时推断 best 约为 `0.3766`
- **`shortT_balanced_v2` 全量训练结果 `overall_model.composite = 0.4429`** (当前最新基线)

这意味着：

- `shortT_balanced_v2` 已经超越了 `iterA3_ashare`，成为当前公开的新基线。
- 尽管超越了旧基线，但 `0.4429` 距离实盘可靠阈值（>0.50）仍有较大差距，准确率极低的核心问题已从“工程无法跑通”转移到了“模型上限与数据信噪比”。

## 2026-04-14 最新开发进度

### 1. 训练基础设施已完成一轮系统级修正

- `train.py` 已从“按运行记录串行逐文件训练”切到“全局混合训练”：
  - train split 统一进入 `ConcatDataset`；
  - 通过 sampler 维持分周期 cap 与时间框架平衡；
  - batch 内开始真正混合不同标的与周期，而不是按文件顺序学完一个再学下一个。
- 验证与调度逻辑已切换为**按样本数加权**的 loss 聚合，不再使用简单 batch 平均。
- 训练与评估流程已补上：
  - `NaN / Inf` 预测、loss、梯度的安全跳过；
  - 显式 gradient clipping；
  - skip counter 记录；
  - 分周期与分 fold 的稳定汇总。
- `ashare_dataset.py` 中波动率链路已切换为严格因果：
  - `.rolling().std().bfill()` 已替换为 `ffill -> fillna(0) -> clamp` 风格；
  - 训练与缓存数据生成语义对齐。

### 2. 已从“平衡/F1 导向”推进到“发现式但受约束”的短周期迭代

- 已加入 discovery target 族：
  - `shortT_discovery_v1`
  - `shortT_discovery_guarded_v1`
  - `shortT_discovery_guarded_v2`
- discovery 版本不再完全死板依赖旧事件定义，而是引入：
  - 未来位移空间、
  - 路径持续性、
  - terminal confirmation、
  - continuation excursion、
  - imbalance alignment
  等结构，鼓励模型学习“会走出一段空间”的点位。
- 为了避免 discovery 彻底漂移，当前 guarded 版本采用：
  - train split 自拟合阈值；
  - val/test 复用 train 阈值；
  - aux-head-aware 的组合打分；
  - signal space / signal quality 指标；
  - 更严格的 checkpoint veto。

### 3. 当前最新 formal 结果：`shortT_discovery_guarded_v2` 已结束，但未形成新基线

- smoke 已完成：
  - `shortT_discovery_guarded_v1`
  - `shortT_discovery_guarded_v2`
- 核心测试已通过：
  - `test_horizon_train_refactor.py`
  - `test_runtime_dataset_cache.py`
- `shortT_discovery_guarded_v2` formal 已提前停止在 `epoch 9`：
  - best composite 出现在 `epoch 4`，约 `0.3940`
  - final `epoch 9 composite = 0.3877`
  - `breakout_precision = 0.1481`
  - `reversion_precision = 0.1498`
  - `60m composite = 0.3644`
  - `public_below_directional_violation_rate = 0.5254`
- 结论：
  - 短周期偏置对 `60m` 有一定修复；
  - 但模型仍远低于 `shortT_balanced_v2` 基线；
  - 当前低准确率的主因已不再是“信号发得太多”，而是**public 头没有被 directional floor 稳定支撑**。

### 4. 当前工程判断已经更新

- 问题重心已经从：
  - “60m 是否该继续降权”
  转移到：
  - “public reversion / breakout 输出如何与方向子头重新耦合”。
- 目前最该修的，不是继续一味强化短周期权重，而是：
  - 强化 directional-public 一致性约束；
  - 让 public 信号在方向证据不足时被抑制；
  - 避免 discovery 头学到只会产生位移分数、却不能落成高精度公共信号的捷径。

## 当前工程判断

### 1. `shortT_balanced_v2` 暴露出数据信噪比与周期撕裂问题

- **回归信号 (Reversion) 坍塌**：在 A 股复杂生态中，左侧抄底逃顶信噪比极低，导致整体假阳性率高达 65%，严重拖累模型分数。
- **60m 周期成为显著噪音陷阱**：日线 (1d) 与 4小时 (240m) 表现良好（突破准确率 >53%），但 1 小时级别 (60m) 充满无序震荡，回归准确率仅 33%。
- **标的特性差异未隔离**：模型用同一套逻辑处理高波段成长股（如比亚迪）与低波段金融/长庄股（如长江电力、招商银行），导致严重偏科。

### 2. 下一步的演进方向（模型上限突破）

- **剥离或降权 60m 训练**：防止盘中无序震荡噪音污染其他周期的特征表达。
- **非对称的信号损失重构**：大幅抑制 A 股 Reversion 侧的预测输出或提高触发阈值，转而专攻 Breakout 顺势突破。
- **引入资产分类标识 (Asset Embedding)**：让模型区分股票的波动率或行业属性，自适应调节均值回归策略的权重。

## 当前并行工作线

### A 股专用 teacher 训练

- 目标是训练 A 股专用 PI-KAN teacher，并作为后续终端蒸馏母体。
- 当前主要入口：
  [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

### 同花顺终端落地

- 目标是把 KHAOS 作为终端消费层稳定表达，而不是在终端环境里重建全部真值逻辑。
- 相关目录：
  [`Finance/02_核心代码/源代码/khaos/同花顺公式`](Finance/02_核心代码/源代码/khaos/同花顺公式)

### 跨平台 truth-engine 设计

- 目标是形成“Python 真值引擎 + Pine/THS 消费层”的长期架构。
- 核心文档：
  [`Finance/04_项目文档/02_开发文档/02_技术文档/跨平台指标引擎架构方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/跨平台指标引擎架构方案.md)

### 毕设与论文材料

- 目标是保持研究目标、技术路线和工程实现之间的叙事一致。
- 核心文档：
  [`Finance/04_项目文档/00_规划与管理/Graduation_Project_Roadmap.md`](Finance/04_项目文档/00_规划与管理/Graduation_Project_Roadmap.md)
  [`Finance/04_项目文档/01_学术论文/README_AI_THESIS.md`](Finance/04_项目文档/01_学术论文/README_AI_THESIS.md)

## 推荐跟进的文档与日志

- [`Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_A股专用网络训练方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_A股专用网络训练方案.md)
- [`Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_同花顺落地方案.md`](Finance/04_项目文档/02_开发文档/02_技术文档/KHAOS_同花顺落地方案.md)
- [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-14_shortT_discovery_guarded_v2_训练复盘与下一轮规划.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-14_shortT_discovery_guarded_v2_训练复盘与下一轮规划.md)
- [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md)
- [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md)

## 当前版本门槛

仓库中已明确写入的升级门槛包括：

- `overall_model.composite >= 0.4187`
- `calibrated_ths.test_objective >= 0.4448`
- `60m composite >= 0.4032`

这些门槛仍需要和 teacher 侧训练结果、分周期表现及 THS 对齐验证一起看待，而不能只凭单一模型分数决定。
