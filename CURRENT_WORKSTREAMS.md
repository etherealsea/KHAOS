# 当前工作线与研究快照

更新日期：`2026-04-07`

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

这意味着：

- `iterA3_ashare` 仍是当前最稳的公开基线。
- `iterA5_ashare` 是重要工程迭代，但截至 `2026-04-07` 还不应视为升级成功。

## 当前工程判断

### 1. `iterA5` 的价值在于暴露问题，而不是直接完成升级

- 补齐了分周期日志、约束统计和更细的诊断能力。
- 证明了把 `5m` 真正接入主训练后，并不会自动带来公共指标提升。
- 暴露出 `public reversion` 与方向分支耦合方式仍需修正。

### 2. 当前阻塞点仍集中在多周期协调，尤其是 `60m`

- 历史分析已反复指出 `60m` 是关键短板之一。
- 新增 `5m` 和更复杂结构后，整体分数仍未超过旧基线，说明问题并非简单由训练轮数不足造成。

### 3. 当前更合理的下一步是 `teacher-first + shortT balanced`

根据 `2026-04-07_shortT训练纠偏.md` 的结论，后续更值得继续验证的是：

- `shortT_balanced_v1`
- `shortT_balanced_v2`

核心思想是：

- 不以牺牲 `reversion` 为代价换取 `breakout` 表面提升。
- 不以牺牲长周期稳定性为代价换取 `5m` 敏感度。
- 继续把 THS 可映射性保留为硬约束。

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
- [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_iterA5阶段复盘.md)
- [`Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md`](Finance/04_项目文档/02_开发文档/01_开发日志/2026-04-07_shortT训练纠偏.md)

## 当前版本门槛

仓库中已明确写入的升级门槛包括：

- `overall_model.composite >= 0.4187`
- `calibrated_ths.test_objective >= 0.4448`
- `60m composite >= 0.4032`

这些门槛仍需要和 teacher 侧训练结果、分周期表现及 THS 对齐验证一起看待，而不能只凭单一模型分数决定。
