# Trae Agent Protocols: Skills & Memory (Project: KHAOS/PI-KAN)

本文档定义了 Trae Agent 在本项目中的协作流程、核心技能（Skills）定义以及记忆（Memory）管理规范。

## 1. 核心记忆 (Core Memory) 管理规范

为了保持长期协作的上下文连贯性，我们将记忆分为以下三类进行管理。Agent 需主动调用 `manage_core_memory` 工具维护这些信息。

### 1.1 Knowledge (项目知识库)
*定义：项目中的恒定事实、核心架构、数学模型定义。*
*   **核心架构**: EKF (Extended Kalman Filter) 分离 Fair Value 与 Kinetic Energy; PI-KAN (Physics-Informed KAN) 计算 Gravity Force。
*   **文件结构**: 
    *   `src/khaos/pine/KHAOS.pine`: 生产环境指标代码。
    *   `scripts/pipeline/optimize_pikan_evolution.py`: 进化算法优化器。
    *   `src/khaos/optimization/pine_replica.py`: Python 版的 Pine 逻辑复刻（用于训练）。
*   **关键概念**:
    *   **Trend Purity**: Q > R (Flow State).
    *   **Phase Transition**: Volatility Compression + Chaos Dominance (R > Q).
    *   **Slope Brake**: 高 Slope 会抑制做空信号（防止逆势）。

### 1.2 Rule (用户规则与偏好)
*定义：用户明确指定的约束条件。*
*   **编程规范**: Pine Script v6 语法；严格的缩进规则。
*   **数据原则**: 禁止使用模拟数据；必须使用真实历史数据训练。
*   **交互原则**: 遇到编译错误需核查全文；日志需关注观点革新而非流水账。
*   **开发哲学**: "Physics-Informed" —— 参数优化必须符合物理意义（如 Bias > 0 时 Gravity 必须 < 0）。

### 1.3 Experience (经验与教训)
*定义：在迭代过程中习得的、可复用的解决问题的方法论。*
*   **Grid Search 优化**: 大数据量下需使用 Loop Inversion 减少内存占用。
*   **信号连续性**: 使用 Latch (锁存器) 逻辑替代单点触发，以形成连续的信号区间。
*   **参数一致性**: Python 训练端的逻辑（如 `pine_replica.py`）必须与 Pine Script 100% 逐行对齐，包括 `tanh` 的近似实现。
*   **适应度函数设计**: 单纯追求 Profit 会导致过拟合；应使用 Pivot Recall (反转点召回率) + Precision (准确率) 的组合指标。

---

## 2. 核心技能 (Skills) 定义

Agent 在本项目中需扮演四个特定角色（Skills），根据任务需求切换。

### Skill 1: The Analyst (诊断与分析)
*   **触发场景**: 用户反馈“信号不对”、“漏掉高点”、“绘线错误”。
*   **SOP (标准作业程序)**:
    1.  **现象确认**: 确认用户描述的图表位置（时间、价格形态）。
    2.  **逻辑回溯**: 在 `pine_replica.py` 或 `KHAOS.pine` 中脑补运行逻辑。
    3.  **特征归因**: 分析是哪个特征导致了问题（例如：Bias 够大但 Slope 也大 -> 导致 Gravity 被抵消）。
    4.  **输出**: 明确的物理/数学解释（如“Slope Brake 机制生效”）。

### Skill 2: The Architect (架构设计)
*   **触发场景**: 需要引入新参数、修改核心公式或添加新功能。
*   **SOP**:
    1.  **物理意义定义**: 新增项代表什么物理量？（如 Entropy 代表市场混乱度）。
    2.  **公式推导**: 将物理量转化为数学表达式（归一化 -> 激活函数 -> 权重组合）。
    3.  **一致性检查**: 确保新逻辑不破坏原有的 EKF 稳定性。

### Skill 3: The Optimizer (参数进化)
*   **触发场景**: 逻辑没问题，但信号位置不准；或需要适应新市场。
*   **SOP**:
    1.  **边界设定**: 在 `optimize_pikan_evolution.py` 中设定合理的参数搜索范围（Bounds）。
    2.  **目标函数调整**: 根据需求调整 Fitness Function（如增加“早发信号”的奖励，降低“逆势”的惩罚）。
    3.  **执行训练**: 运行 DE 算法，获取收敛结果。
    4.  **结果验证**: 检查生成的权重是否出现了极端的数值（过拟合迹象）。

### Skill 4: The Implementer (工程落地)
*   **触发场景**: 将优化结果部署到 Pine Script。
*   **SOP**:
    1.  **代码同步**: 将 Python 优化的权重 (`w_q_...`, `w_m_...`) 填入 Pine Script。
    2.  **语法合规**: 确保符合 Pine v6 规范（缩进、类型转换）。
    3.  **可视化**: 更新 Plot 逻辑，确保视觉上直观（如颜色变化、区间绘制）。

---

## 3. 工作流 (Workflow) 示例

**场景：用户报告“最高点没有空单信号”**

1.  **Analyst**: 分析发现是 Slope 权重过大，导致在急涨时 Gravity 无法转负。
2.  **Architect**: 决定不修改公式结构，而是通过调整参数权重来解决。
3.  **Optimizer**: 
    *   修改 Fitness Function，加大对“Pivot 点”的召回奖励。
    *   运行进化算法，让模型学会“降低 Slope 权重”或“增加 Bias 权重”。
4.  **Implementer**: 将新生成的权重更新到 `KHAOS.pine`，并提交给用户验证。
