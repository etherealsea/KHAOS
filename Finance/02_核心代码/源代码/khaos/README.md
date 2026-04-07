# KHAOS 核心包说明

`khaos/` 是当前研究工程的主代码包，负责承接从物理特征、数据处理到 PI-KAN 训练与终端映射的核心逻辑。

## 模块划分

- `核心引擎/`
  物理特征与指标逻辑，包括 `Hurst`、`EKF`、`Entropy`、`MLE` 等相变相关计算。
- `数据处理/`
  数据加载、标准化、A 股支持、样本权重与训练集 profile。
- `模型定义/`
  PI-KAN / KAN 主体、注意力残差块、RevIN 等网络结构。
- `模型训练/`
  trainer、loss、选模逻辑、规则提取与模型分析。
- `回测模块/`
  更偏解释与回看，例如 `symbolic_extractor.py`。
- `同花顺公式/`
  同花顺表达层文件、代理映射与验证说明。
- `Pine Script代码/`
  TradingView 侧脚本实现。
- `工具箱/`
  报告生成、可视化与辅助工具。

## 代码链路

从当前实现看，典型链路是：

1. `数据处理/` 组织输入数据与训练样本。
2. `核心引擎/physics.py` 生成统一的物理特征。
3. `模型定义/kan.py` 定义双核网络结构。
4. `模型训练/train.py` 组织训练、验证与 checkpoint 选择。
5. `回测模块/`、`同花顺公式/`、`Pine Script代码/` 负责解释、验证与终端表达。

## 关键入口

- 物理特征核心：
  [`核心引擎/physics.py`](核心引擎/physics.py)
- A 股数据支持与时间切分：
  [`数据处理/ashare_support.py`](数据处理/ashare_support.py)
- 数据集权重与 profile：
  [`数据处理/ashare_dataset.py`](数据处理/ashare_dataset.py)
- 主模型定义：
  [`模型定义/kan.py`](模型定义/kan.py)
- 主训练器：
  [`模型训练/train.py`](模型训练/train.py)
- 约束与损失：
  [`模型训练/loss.py`](模型训练/loss.py)
- 同花顺代理验证：
  [`同花顺公式/ths_core_proxy.py`](同花顺公式/ths_core_proxy.py)
- 时序解释与导出：
  [`回测模块/symbolic_extractor.py`](回测模块/symbolic_extractor.py)

## 与仓库级 runner 的关系

`模型训练/train.py` 是通用训练底座，但当前实践中更常通过仓库级 runner 组装：

- 数据路径
- dataset / loss / score profile
- smoke 与 formal 配置
- 本地输出目录

当前 A 股主线常用入口：

- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](../../../03_实验与验证/脚本/测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
- [`Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py`](../../../03_实验与验证/脚本/测试与临时脚本/run_iterA5_ashare_train.py)

## 边界说明

- `旧代码归档/` 仅作为历史参考，不是默认编辑入口。
- 终端公式层负责消费和表达，不应反向定义训练真值。
- 模型权重、日志与大部分实验输出默认保留在本地，不作为源码包的一部分。
