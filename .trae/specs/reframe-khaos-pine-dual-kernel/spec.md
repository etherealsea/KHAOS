# KHAOS Pine 双核语义重构 Spec

## 为什么 (Why)
iter9 的训练结论已经表明，当前主线不再是继续围绕 epoch 做训练试探，而是把 breakout / reversion 双核结果诚实、稳定、统一地映射到 Pine Script。现有 `KHAOS.pine` 虽然已有状态机骨架，但整体语义仍偏旧版阈值状态机与抽象能量表达，尚未完整对齐 iter9 的双核叙事，也没有清晰处理“全局统一逻辑下高周期 breakout 稀疏”这一现实约束。

## 做了哪些更改 (What Changes)
- **BREAKING**: 将 `KHAOS.pine` 的对外核心语义正式切换为 iter9 的 breakout / reversion 双核状态，而不是旧版单一 Force 或抽象能量叙事。
- **BREAKING**: 指标输出与可视化必须优先表达“当前属于 breakout 还是 reversion、方向是 bull 还是 bear、是否处于 neutral/filter 区”，而不是继续围绕神秘主线值组织阅读方式。
- 保留“全周期统一”的单一核心逻辑，不为不同周期维护独立策略分支。
- 在不破坏统一逻辑的前提下，为 breakout 增加有限的周期感知输出映射或视觉提示，用于诚实表达其在高周期上的天然稀疏性。
- 重写 Pine 对应技术文档，删除或替换 Force / 红蓝粗线等过时描述，使文档与当前脚本语义一致。

## 影响 (Impact)
- 受影响的能力：
  - Pine 指标状态定义
  - 双核方向结算与中性过滤
  - 多周期统一表达
  - 指标可视化与解释文档
- 受影响的代码与文档：
  - `Finance/02_核心代码/源代码/khaos/Pine Script代码/KHAOS.pine`
  - `Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_TO_INDICATOR_MAPPING.md`
  - `Finance/04_项目文档/02_开发文档/01_开发日志/2026-03-27.md`

## 新增需求 (ADDED Requirements)
### 需求：双核状态主表达
系统必须把 Pine 指标的主表达切换为 breakout / reversion 双核状态，并保证任一时刻都能明确区分当前是否处于 breakout、reversion 或 neutral/filter 区。

#### 场景：状态识别成功
- **当** 指标接收到新的 KHAOS 物理代理输入
- **则** 脚本应先完成双核状态判断，再输出当前所属状态，而不是仅输出一个需要二次解释的抽象主线值

### 需求：方向与中性区显式输出
系统必须在双核状态之外，进一步显式表达 bull / bear 方向与 neutral/filter 区，使用户能直接读取“状态类型 + 方向 + 是否应过滤”。

#### 场景：中性过滤
- **当** breakout 与 reversion 都未达到有效触发条件，或方向确认不足
- **则** 指标应进入 neutral/filter 区，而不是勉强归类为某个方向信号

### 需求：全周期统一下的 breakout 诚实表达
系统必须保持全周期统一逻辑，但应针对 breakout 的高周期稀疏现实提供有限的周期感知输出映射或视觉提示；系统不得通过人为夸大高周期 breakout 强度来制造一致性假象。

#### 场景：高周期 breakout 稀疏
- **当** 指标运行在 1h、4h、1d 等高周期，且 breakout 条件天然更难满足
- **则** 指标可以通过输出映射、视觉弱化或标签提示表达“breakout 更偏短中周期确认核”，但不得直接伪造与低周期同等密度的 breakout 触发

### 需求：可视化围绕双核语义重组
系统必须让可视化首先服务于双核状态识别，优先帮助用户判断“当前更值得关注哪一类状态”，而不是继续以能量波、主线值或旧版 Force 语义作为核心阅读入口。

#### 场景：用户查看图表
- **当** 用户打开指标
- **则** 图上主要信息应围绕 breakout / reversion、bull / bear、neutral/filter 区展开，并能直观看出当前周期更偏向哪一类状态

## 修改的需求 (MODIFIED Requirements)
### 需求：Pine 主脚本状态机
现有 Pine 状态机必须继续保留其骨架与统一物理代理输入，但其命名、结算逻辑对外表达、视觉优先级与说明文本必须整体向 iter9 双核语义靠拢。

### 需求：Pine 技术文档
Pine 技术文档必须不再使用 Force、红蓝粗线或“神秘力场值”作为主叙事，而应完整说明双核状态、方向判断、中性过滤以及 breakout 的周期适用差异。

## 移除的需求 (REMOVED Requirements)
### 需求：单一 Force 叙事作为 Pine 主表达
**原因**：该叙事已经与 iter9 的 breakout / reversion 双核训练主线脱节，且会误导后续实现继续围绕抽象能量值组织指标可视化。
**迁移**：迁移到“以 breakout / reversion 双核状态为主表达，并补充方向、中性过滤与周期适用说明”的新语义体系。
