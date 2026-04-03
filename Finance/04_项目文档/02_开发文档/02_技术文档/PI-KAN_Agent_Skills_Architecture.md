# PI-KAN Agent Skills Architecture
**Version**: 1.0
**Date**: 2026-01-09

## 1. 架构设计理念 (Design Philosophy)
受 Claude "Agent Skills" 架构的启发，我们将 **PI-KAN (Physics-Informed KAN)** 系统重构为一个基于**能力 (Skills)** 的模块化智能体系统。

在传统量化系统中，策略往往是一堆耦合的代码。而在 **Skills Architecture** 中，我们将交易系统的核心能力解耦为独立的、可描述的、可执行的单元。这不仅有助于人类开发者维护，更重要的是为未来的 AI 自主交易代理 (Autonomous Trading Agent) 提供了标准化的“操作手册”。

### 核心原则
1.  **能力即文档 (Skill as Documentation)**: 每个能力由一个 `SKILL.md` 定义，包含该能力的元数据、使用场景、以及具体的执行逻辑（代码或指令）。
2.  **渐进式披露 (Progressive Disclosure)**: 系统（或 AI）仅在需要特定能力时加载相关上下文，避免信息过载。
3.  **物理约束内嵌 (Embedded Physics)**: 所有的决策类 Skill 都必须包含物理约束（如动能守恒）作为前置检查条件。

---

## 2. PI-KAN 能力图谱 (Skill Graph)

我们将 PI-KAN 系统的能力划分为三个核心层级：

### Level 1: 感知层 (Sensing Skills)
*负责从混沌的市场数据中提取物理特征*
*   **`market_sensing/ekf_estimation`**: 使用扩展卡尔曼滤波 (EKF) 分离价格的势能与动能。
*   **`market_sensing/hurst_regime`**: 计算 Hurst 指数以识别当前的市场体制（均值回归 vs 趋势跟随）。
*   **`market_sensing/entropy_analysis`**: 使用排列熵 (Permutation Entropy) 量化市场复杂度。

### Level 2: 认知层 (Cognitive Skills)
*负责结合物理规律进行非线性推理*
*   **`phase_detection/kan_inference`**: 调用 Kolmogorov-Arnold Network 预测相变概率。
*   **`phase_detection/physics_validation`**: 验证预测结果是否违反物理守恒定律（如动能衰竭定律）。

### Level 3: 决策层 (Decision Skills)
*负责最终的交易执行与风控*
*   **`execution/signal_generation`**: 综合感知与认知结果，生成买卖信号。
*   **`execution/dynamic_risk_control`**: 基于当前熵值动态调整止损与仓位。

---

## 3. Skill 结构标准 (Standard Structure)

每个 Skill 目录 (如 `skills/market_sensing`) 包含一个 `SKILL.md`，结构如下：

```yaml
---
name: [Skill Name]
description: [One-line description of what this skill does]
version: 1.0.0
dependencies: [List of required tools/libraries]
physics_constraints: [Applicable physics laws]
---

# Instructions
[Detailed instructions on how to perform this skill]

# Context
[Background knowledge required]

# Interface
[Input/Output specification]
```

---

## 4. 实施路线图 (Implementation Roadmap)
1.  **Phase 1**: 建立 `skills/` 目录结构，将现有的 Pine Script 和 Python 逻辑映射为 Skill 文档。
2.  **Phase 2**: 开发 "Skill Executor" (基于 MCP)，允许 AI 直接调用这些 Skill 进行回测或分析。
3.  **Phase 3**: 实现全自动的 PI-KAN Agent，使其能根据市场状态自主选择合适的 Skill 组合。
