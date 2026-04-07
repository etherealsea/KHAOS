# 实验脚本索引

这个目录是仓库的“操作层入口”。虽然其中有个子目录叫 `测试与临时脚本`，但它目前实际上包含了多条仍在使用的主线 runner。

## 先看哪些脚本

如果你是新协作者或新的 AI，建议按这个顺序理解：

1. [`setup/fetch_ashare_data.py`](setup/fetch_ashare_data.py)
2. [`测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
3. [`测试与临时脚本/run_iterA5_ashare_train.py`](测试与临时脚本/run_iterA5_ashare_train.py)
4. [`测试与临时脚本/analyze_iterA5_ashare_results.py`](测试与临时脚本/analyze_iterA5_ashare_results.py)
5. [`测试与临时脚本/validate_iterA5_ths_proxy.py`](测试与临时脚本/validate_iterA5_ths_proxy.py)

## 目录说明

### `setup/`

用于准备运行环境或补齐外部依赖数据。

重点文件：

- [`setup/fetch_ashare_data.py`](setup/fetch_ashare_data.py)
  A 股数据抓取、补齐和 coverage 检查入口
- [`setup/download_github_data.py`](setup/download_github_data.py)
  额外数据下载脚本

### `测试与临时脚本/`

名字偏历史，但这里目前是 A 股主线最重要的 runner 集中地。

重点文件：

- [`测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
  当前建议优先阅读和继续推进的实验入口，包含 `iterA4_refresh`、`shortT_balanced_v1`、`shortT_balanced_v2` 等定义
- [`测试与临时脚本/run_iterA5_ashare_train.py`](测试与临时脚本/run_iterA5_ashare_train.py)
  `iterA5` 正式训练 runner
- [`测试与临时脚本/analyze_iterA5_ashare_results.py`](测试与临时脚本/analyze_iterA5_ashare_results.py)
  `iterA5` 分析与基线对比
- [`测试与临时脚本/validate_iterA5_ths_proxy.py`](测试与临时脚本/validate_iterA5_ths_proxy.py)
  同花顺代理一致性验证

其他 `iterA1 ~ iterA4` 的 runner / analyzer 主要是阶段性历史记录，可作为对照模板。

### `pipeline/`

这里是更通用的研究和旧管线脚本。

用途包括：

- 通用训练 / 回测
- 研究型分析
- 优化器与验证脚本

如果目标是继续当前 A 股主线，不建议从 `pipeline/` 开始改动，除非你已经确认需要复用其中的公共逻辑。

## 当前推荐工作流

1. 准备或补齐 A 股训练数据
2. 先跑 teacher-first 系列 smoke 实验
3. 查看 per-timeframe / constraint / gate 指标
4. 只有在 smoke 通过后，才进入更重的正式训练
5. 保留 THS 代理验证，不用单一 teacher 分数做最终结论

## 当前最重要的判断

- `iterA5` 是一个重要的工程迭代，但不是当前默认升级候选
- `shortT_breakout_v1` 是探索分支，不应直接当作默认主线
- 当前更值得继续验证的是 `shortT_balanced_v1` / `shortT_balanced_v2`
