# Tasks

- [x] 任务 1：重定义 Pine 双核状态语义
  - [x] 子任务 1.1：梳理 `KHAOS.pine` 现有状态机中的 breakout、reversion、phase 与方向结算逻辑
  - [x] 子任务 1.2：将外部可读语义统一改写为 iter9 的 breakout / reversion 双核状态
  - [x] 子任务 1.3：明确 neutral/filter 区的进入条件与输出方式

- [x] 任务 2：加入 breakout 的有限周期感知表达
  - [x] 子任务 2.1：在不拆分多套周期逻辑的前提下设计 breakout 的输出映射或视觉提示
  - [x] 子任务 2.2：确保高周期 breakout 被诚实表达，而不是被人为增强
  - [x] 子任务 2.3：确保 reversion 继续保持跨周期稳定核的主定位

- [x] 任务 3：重组 Pine 可视化与用户读取入口
  - [x] 子任务 3.1：让图表优先表达 breakout / reversion、bull / bear、neutral/filter
  - [x] 子任务 3.2：压低旧版抽象能量表达的阅读优先级
  - [x] 子任务 3.3：让用户能直观看出当前周期更值得关注哪一类状态

- [x] 任务 4：同步技术文档与开发记录
  - [x] 子任务 4.1：重写 `PIKAN_TO_INDICATOR_MAPPING.md` 中的过时 Force / 红蓝粗线叙事
  - [x] 子任务 4.2：补充双核语义、方向判断、中性过滤与周期适用差异说明
  - [x] 子任务 4.3：将本轮确认的关键观点沉淀到当日开发日志

- [x] 任务 5：完成 Pine 语法与语义验证
  - [x] 子任务 5.1：检查脚本是否符合 Pine Script v6 语法约束
  - [x] 子任务 5.2：核查是否仍残留旧版 Force 叙事或误导性命名
  - [x] 子任务 5.3：确认实现结果与 iter9 的全周期统一约束一致

# Task Dependencies
- [任务 2] depends on [任务 1]
- [任务 3] depends on [任务 1], [任务 2]
- [任务 4] depends on [任务 1], [任务 3]
- [任务 5] depends on [任务 1], [任务 2], [任务 3], [任务 4]
