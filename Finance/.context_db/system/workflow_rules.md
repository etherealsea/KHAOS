# OpenViking 开发工作流规范 (Finance Project)

## 核心原则 (Core Principles)
1.  **上下文优先 (Context First)**: 
    *   在开始任何开发任务前，首先通过 `Finance/.context_db/system` 或 `Finance/.context_db/memory` 检索相关上下文。
    *   **禁止** 凭空假设项目结构或依赖关系。
    *   **禁止** 将整个代码库或无关文件一次性塞入 Context Window (节省 Token)。

2.  **文件系统范式 (File System Paradigm)**:
    *   将所有非代码的上下文信息（如规则、模式、配置）视为“文件”。
    *   Agent 必须通过 `ContextFileSystem` (模拟) 接口读取这些信息。
    *   例如：在修改 PI-KAN 模型前，先读取 `Finance/.context_db/memory/patterns/pi_kan_architecture.md`。

3.  **自进化与回写 (Self-Evolution)**:
    *   任务完成后，如果发现了新的模式（如“某种参数组合导致模型发散”），必须将其记录回 `Finance/.context_db/memory/patterns/`。
    *   如果是成功的代码变更，更新 `Finance/.context_db/skills/` 下的相关技能描述。

## 标准开发流程 (Standard Workflow)

### Step 1: 检索上下文 (Retrieve)
Agent 接收任务后，首先执行：
*   **Action**: Search `.context_db` for relevant files.
*   **Example**: "Task: Fix Hurst calculation bug." -> Search `Finance/.context_db/skills/hurst_calculation.md` and `Finance/src/khaos/utils/physics_metrics.py`.

### Step 2: 制定计划 (Plan)
基于检索到的上下文，制定修改计划。
*   **Requirement**: 明确列出受影响的文件和预期的行为变更。
*   **Check**: 确认计划是否符合 `Finance/.context_db/system/architecture_rules.md` (如“必须保持 EKF 可微分”)。

### Step 3: 执行与验证 (Execute & Verify)
*   **Action**: 修改代码。
*   **Validation**: 运行单元测试或回测脚本。
*   **Constraint**: 任何新引入的依赖必须记录在 `requirements.txt`。

### Step 4: 更新上下文 (Update Context)
*   **Action**: 将本次任务的关键发现写入 `.context_db`。
*   **Example**: "Found that `min_window=10` is too small for Hurst, updated default to 30." -> Write to `Finance/.context_db/memory/tuning_logs.md`.

## MCP 工具使用规范 (MCP Usage)
为了减少 Token 消耗和避免工具冗余：
1.  **首选内置工具**: 文件读写 (`Read`, `Write`)、搜索 (`SearchCodebase`)、终端 (`RunCommand`) 是最高效的。
2.  **按需使用高级工具**:
    *   `Chrome DevTools`: 仅在需要调试前端页面或爬取动态网页时启用。日常后端开发 **禁用**。
    *   `Office MCP`: 仅在需要操作 Excel/Word 复杂格式时启用。生成简单 CSV/Markdown **禁用** (直接写文件即可)。
    *   `Sequential Thinking`: 仅在处理极其复杂的逻辑推理时启用。简单任务 **禁用**。
3.  **保持工具列表精简**: 任何时候，如果一个任务可以通过写 Python 脚本完成，就不要调用专门的 MCP 工具（如 Excel 操作）。

## 目录结构映射 (Directory Mapping)
*   `/system/`: 系统级规则 (本文件所在位置)。
*   `/memory/`: 长期记忆 (模式、日志)。
*   `/skills/`: 可复用的技能片段 (代码片段、Prompt 模板)。

## Global Naming Convention
All new files and directories (except code packages) must use Chinese names or include Chinese descriptions. For code directories, a mapping document must be maintained.
