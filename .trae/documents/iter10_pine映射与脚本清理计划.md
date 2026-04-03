# iter10 Pine 映射与脚本清理计划

## 一、Summary

- 目标一：把 iter9 的最新训练成果落地到 [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine)，使 Pine 指标与当前 breakout / reversion 双核训练语义一致。
- 目标二：按“全周期统一”原则设计 Pine 侧逻辑，不先按时间周期拆分成多套版本，但会在统一逻辑中明确记录当前高周期 breakout 偏弱这一现实约束。
- 目标三：更新当日开发日志，使用 **2026-03-27** 新建开发日志文件，记录 iter9 最终成果向 Pine 映射的原则、约束与下一步实施判断。
- 目标四：清理明确过时、失配、不会再使用的脚本文件，范围包含 Pine 相关旧脚本和旧训练/分析临时脚本。

## 二、Current State Analysis

### 1. 当前 Pine 实现状态

- 当前实际 Pine 文件只有一个：[KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219)。
- 该脚本已不属于旧版 “Force 红蓝粗线” 体系，而是：
  - 使用 Z-Score 对齐；
  - 拥有 `raw_breakout` / `raw_reversion`；
  - 引入 `anchor_len` 锚点机制；
  - 使用 `current_phase` 状态机做相位结算。
- 但脚本里的阈值、可视化语义和训练侧最新结论仍未完全同步到 iter9 阶段。

### 2. 当前训练结论与 Pine 的关键落差

- iter9 已确认：
  - 最优点在 Epoch 15；
  - breakout / reversion 双核更平衡；
  - breakout 伪信号率下降；
  - reversion 跨周期稳定；
  - breakout 在高周期被统一阈值显著压制。
- 当前 Pine 还没有把这些最新结论转成一套面向 “全周期统一” 的输出语义，例如：
  - breakout 需要更强调“短中周期主用、全周期统一逻辑但高周期更稀疏”；
  - reversion 需要更明确作为跨周期稳定核；
  - 可视化需要从“旧阈值提示器”转向“最新状态核解释器”。

### 3. 当前仓库中的明显过时脚本

- Pine 相关失配脚本：
  - [run_pine_local_checks.py](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/03_实验与验证/脚本/pipeline/run_pine_local_checks.py#L1-L60)
  - 问题：引用不存在的 `src.khaos.analysis.verify_pine_export`，且 Pine 路径仍是旧的 `src/khaos/pine/KHAOS.pine`
- 训练/分析临时脚本目录中已有多批阶段性脚本与杂项文件：
  - [测试与临时脚本 目录](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/03_实验与验证/脚本/测试与临时脚本)
  - 包含旧训练脚本、旧分析脚本、临时检查脚本、一次性测试文件、`__pycache__`
- 已知旧版归档 Pine 副本：
  - [旧代码归档/KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/旧代码归档/khaos_kan/KHAOS.pine#L1-L80)
  - 该文件属于历史实现，与当前源代码目录中的 Pine 实现体系不同。

## 三、Assumptions & Decisions

- 决策 1：Pine 目标采用 **全周期统一**，不先为 5m/15m 与 1h/4h/1d 分裂成多套脚本。
- 决策 2：开发日志不继续追加到 [2026-03-26.md](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/04_项目文档/02_开发文档/01_开发日志/2026-03-26.md)，而是按项目规则新增 **2026-03-27.md**。
- 决策 3：删除范围包含 Pine 相关旧脚本与旧训练脚本，但优先删除“当前目录中明确失配/过时/一次性”的脚本，不把“开发日志、训练日志、最终权重”纳入清理范围。
- 决策 4：Pine 改造以当前 [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219) 为基础重构，不再回退到旧的 Force/红蓝粗线方案。

## 四、Proposed Changes

### A. 开发日志更新

#### 文件

- `Finance/04_项目文档/02_开发文档/01_开发日志/2026-03-27.md`

#### 变更内容

- 新建 2026-03-27 开发日志，记录：
  - iter9 到 Pine 映射的最新原则；
  - “全周期统一”这一策略选择；
  - breakout / reversion 在 Pine 侧各自承担的角色；
  - 高周期 breakout 当前偏弱这一现实约束；
  - 为什么下一步不再延长 epoch，而转向 Pine 映射与阈值表达层。

#### 原因

- 现有 [2026-03-26.md](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/04_项目文档/02_开发文档/01_开发日志/2026-03-26.md#L127-L197) 已完整记录 iter9 训练结果，下一步应分离为新的实施日志。

### B. Pine 主脚本改造

#### 文件

- [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219)

#### 变更内容

- 以 iter9 的双核结论重新整理 Pine 结构，重点包括：
  - 保留 `//@version=6` 与现有 feature engineering 主骨架；
  - 更新参数命名、分组和视觉语义，使输入含义与 breakout / reversion 双核对齐；
  - 重写或整理 breakout / reversion 的统一判定逻辑，让其更贴合 iter9 的状态核含义；
  - 调整可视化输出，让使用者能明确区分：
    - breakout 主状态；
    - reversion 主状态；
    - 相位方向；
    - 中性/过滤区；
  - 去除仍带有旧时代映射痕迹、但不再服务当前训练语义的表达。

#### 实施原则

- 不直接把 Python 训练代码逐行硬翻到 Pine。
- 不引入 Pine 中无法稳定表达的训练态变量。
- 在“全周期统一”前提下，显式保留 breakout 与 reversion 的差异性，而不是强行把两者揉成一个单一 Force 指标。
- 尽量避免换行复杂度，保持 Pine v6 语法稳定，符合用户已有 Pine 规则偏好。

### C. 旧脚本与冗余脚本清理

#### 计划清理的高优先级候选

- `Finance/03_实验与验证/脚本/pipeline/run_pine_local_checks.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/check_files2.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/check_log.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/check_log2.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/p.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/reorg.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/test_bulk.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/test_bulk2.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/test_ds.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/test_es.csv`
- `Finance/03_实验与验证/脚本/测试与临时脚本/__pycache__/`

#### 计划纳入清理评估的旧训练脚本

- `Finance/03_实验与验证/脚本/测试与临时脚本/run_iter4_full_train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/run_iter5_full_train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/run_iter6_full_train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/run_iter7_full_train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/run_iter8_full_train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iter4_results.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iter5_results.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iter6_results.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iter7_results.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iter8_results.py`

#### 保留项

- `run_iter9_full_train.py`
- `analyze_iter9_results.py`
- iter9 最终日志与权重
- 当前源代码目录中的 [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219)

#### 说明

- 旧代码归档目录中的历史文件不作为第一优先级删除对象；若执行时确认“归档本身也要瘦身”，再做单独处理。
- 对旧 iter 脚本的删除，执行前会先核查它们是否仍被当前文档或自动流程显式引用；若仍被引用，则改为归档或保留。

### D. 文档一致性修补

#### 文件

- [PIKAN_TO_INDICATOR_MAPPING.md](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/04_项目文档/02_开发文档/02_技术文档/PIKAN_TO_INDICATOR_MAPPING.md#L1-L45)

#### 变更内容

- 将其更新为与当前 Pine 实现一致的说明，至少修正：
  - 旧的 Force/红蓝粗线叙事；
  - 已失效的阈值说明；
  - 与当前真实 Pine 路径不一致的表述。

#### 原因

- 如果只改 [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219) 而不修正文档，仓库会继续保留两套互相冲突的 Pine 叙事。

## 五、Implementation Steps

1. 新建 2026-03-27 开发日志，记录 iter9 → Pine 的映射原则与下一步实施目标。
2. 复读当前 [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219) 并按 Pine v6 约束梳理结构。
3. 根据 iter9 结果重写 Pine 的状态映射、阈值表达与可视化输出。
4. 更新 Pine 映射技术文档，使其与当前实现一致。
5. 删除高优先级冗余脚本。
6. 核查旧训练脚本与旧分析脚本是否仍被引用，删除明确不会再使用的部分。
7. 做 Pine 语法级检查与静态复读，确保不会引入换行/缩进类错误。
8. 输出最终结果：说明 Pine 改动、删除清单、验证方式与剩余限制。

## 六、Verification

- 代码核查：
  - 复读 [KHAOS.pine](file:///d:/《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》/Finance/02_核心代码/源代码/khaos/Pine%20Script代码/KHAOS.pine#L1-L219) 全文，确认 Pine v6 语法、换行、缩进、参数形式稳定。
- 文档核查：
  - 确认 2026-03-27 开发日志已落入正确目录并使用当天日期命名。
  - 确认 Pine 映射文档不再引用旧 Force 红蓝线逻辑。
- 清理核查：
  - 重新列目录确认冗余脚本已清除。
  - 确认保留文件仅包含最新必要训练脚本、分析脚本与当前 Pine 主脚本。
- 结果核查：
  - 确认最终汇报能明确回答：
    - Pine 如何承接 iter9 最新训练成果；
    - 哪些旧脚本被删；
    - 哪些限制仍存在，尤其是高周期 breakout 偏弱的问题。
