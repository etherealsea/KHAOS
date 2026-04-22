# Iter12 去硬门槛化：Soft Gate 全贯通设计文档 (方案A)

## 1. 背景与目标
在 Iter12 的去硬门槛化交接节点中，`kan.py` 的物理门控已从不可学习的 detached hard gates 调整为可参与反向传播的 **soft_annealed** 机制。然而，这一调整尚未完全贯通到训练入口与 runner 层。

为了能在本地机器上对门控参数进行系统性的扫参调优（观察 `gate_floor_hit_rate` 等指标变化对胜率的提升作用），我们需要将 soft gate 的关键超参数通过 CLI 参数贯通至模型构造并写入实验记录（run_manifest），不再硬编码在网络中。

## 2. 参数贯通范围
以下参数将从 `run_iter12_multiasset_closed_loop.py` → `train.py` → `kan.py` 实现全链路贯通：

| 参数名 | 说明 | 默认值 (Formal) |
| --- | --- | --- |
| `gate_mode` | 门控模式，可选 `soft_annealed`, `legacy_hard`, `disabled` 等 | `soft_annealed` |
| `gate_floor_breakout` | 突破信号门控的下限保护值 | `0.25` |
| `gate_floor_reversion` | 反转信号门控的下限保护值 | `0.35` |
| `gate_anneal_fraction` | 门控斜率退火占总 epoch 的比例 | `0.40` |
| `horizon_search_spec` | 多视野搜索候选集合 (逗号分隔，方案 B 补充，避免退化为单视野) | `6,10,14,20` |

## 3. 代码修改计划

### 3.1 Runner 层 (`scripts/run_iter12_multiasset_closed_loop.py`)
- 在 `SMOKE_PRESET` 和 `FORMAL_PRESET` 中增加上述门控参数的默认值。
- 在 `argparse` 定义中新增上述 CLI 参数。
- 在 `resolve_effective_config` 中正确合并配置。
- 在拼装 `cmd` 时，将这些参数传入 `train.py`。

### 3.2 Train 层 (`Finance/02_核心代码/源代码/khaos/模型训练/train.py`)
- 在 `argparse` 中增加上述参数。
- 将这些参数作为字典存入 `run_manifest`。
- 在实例化 `KHAOS_KAN` 时，将 `args` 中的对应参数传入模型构造函数。

### 3.3 测试与验证层
- 修改/补齐相应的单元测试（如 `test_iter12_multiasset_runner.py`），确保默认预设和参数贯通逻辑的正确性。
- 通过一个 smoke 训练的 mock-up 跑通单步以验证参数流向没有问题。
