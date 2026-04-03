# OpenViking 集成架构：上下文数据库与 Agent 开发规范

## 1. 简介
OpenViking 是由字节跳动火山引擎 Viking 团队开源的 AI Agent 上下文数据库 (Context Database)。它采用“文件系统范式 (File System Paradigm)”来统一管理 Agent 所需的各类上下文信息（如记忆、技能、工具配置、资源等），通过层级化的上下文传递和自进化机制，实现高效的 Agent 开发与运行。

在本项目 (PI-KAN 金融相变探测) 中引入 OpenViking 架构，旨在解决以下问题：
1.  **Token 消耗优化**: 通过结构化的上下文管理和智能检索，避免将无关信息（如整个代码库或历史记录）一次性塞入 Context Window。
2.  **开发流程规范化**: 统一 Agent (如 KHAOS 交易员、策略分析师) 的数据访问接口，避免硬编码和散乱的状态管理。
3.  **长期记忆与自进化**: 支持将交易历史、相变模式识别结果持久化为“知识”，供未来决策使用。

## 2. 核心概念

### 2.1 上下文即文件 (Context as File System)
OpenViking 将所有上下文抽象为文件和目录树。
*   `/memory/`: 长期记忆，如历史交易记录、成功的相变预警案例。
*   `/skills/`: 技能库，如 `calculate_hurst`, `execute_trade`, `fetch_data`。
*   `/resources/`: 静态资源，如配置文件、Prompt 模板。
*   `/session/`: 当前会话的短期上下文。

### 2.2 核心组件
1.  **ContextDatabase (CDB)**: 核心存储引擎，负责上下文的读写、索引和检索。
2.  **ContextFileSystem (CFS)**: 文件系统接口，允许 Agent 像操作文件一样操作上下文（`read`, `write`, `mount`）。
3.  **Retriever**: 混合检索器，支持基于语义（Vector）和关键词（BM25）的上下文查找。

## 3. 架构设计 (PI-KAN Project Integration)

我们将构建一个轻量级的 OpenViking 适配层 (`src/infrastructure/open_viking/`)，模拟其核心功能。

### 3.1 目录结构映射
```
Finance/
├── .context_db/             # 本地上下文数据库存储 (模拟 OpenViking 后端)
│   ├── memory/              # 长期记忆
│   │   ├── trading_logs/    # 交易日志 (JSON/Parquet)
│   │   └── patterns/        # 识别到的 KHAOS 相变模式
│   ├── skills/              # 技能描述与元数据
│   └── config/              # 系统配置
└── src/
    └── infrastructure/
        └── open_viking/
            ├── core.py      # ContextDatabase, FileSystem 抽象
            ├── memory.py    # 记忆管理 (Vector Store 封装)
            └── retriever.py # 检索逻辑
```

### 3.2 工作流集成
1.  **初始化**: Agent 启动时，挂载 (Mount) 必要的上下文目录（如 `/memory/patterns`）。
2.  **任务执行**:
    *   Agent 接收指令（如“分析当前市场状态”）。
    *   Agent 通过 `CFS.search("/memory/patterns", query="high volatility crash")` 检索历史相似案例。
    *   仅将检索到的 TOP-K 案例放入 Prompt，显著减少 Token。
3.  **结果回写**: 任务完成后，将新的发现（如“Hurst < 0.3 且波动率飙升”）写入 `/memory/patterns/new_case.json`，实现自进化。

## 4. 实施路线图

1.  **Phase 1: 基础架构 (当前)**
    *   定义 `ContextDatabase` 和 `ContextFileSystem` 接口。
    *   实现基于本地文件系统和简单的向量检索 (FAISS/Chroma) 的 MVP 版本。
2.  **Phase 2: Agent 改造**
    *   重构 `KHAOS` 策略代码，使其通过 `ContextDB` 读取配置和历史数据，而非直接读取 CSV。
3.  **Phase 3: 知识库构建**
    *   将开题报告、物理公式文档结构化存入 `/knowledge/` 目录，供 Agent 实时查询物理约束公式。

## 5. Token 优化策略
*   **按需加载**: 不再在 System Prompt 中包含所有规则，而是根据用户 Query 动态加载相关规则。
*   **语义压缩**: 将长文本总结为向量摘要存入 DB，检索时先匹配摘要。
