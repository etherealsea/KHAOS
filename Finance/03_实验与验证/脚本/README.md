# 实验与验证脚本导览

这个目录是项目的操作层入口，负责把核心代码、数据目录和阶段性实验组织起来。目录命名里保留了一些历史痕迹，但当前活跃主线也在这里运行。

## 推荐先读的脚本

如果目标是理解当前 A 股工程主线，建议按下面顺序阅读：

1. [`setup/fetch_ashare_data.py`](setup/fetch_ashare_data.py)
2. [`测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
3. [`测试与临时脚本/run_iterA5_ashare_train.py`](测试与临时脚本/run_iterA5_ashare_train.py)
4. [`测试与临时脚本/analyze_iterA5_ashare_results.py`](测试与临时脚本/analyze_iterA5_ashare_results.py)
5. [`测试与临时脚本/validate_iterA5_ths_proxy.py`](测试与临时脚本/validate_iterA5_ths_proxy.py)

## 目录说明

### `setup/`

用于准备本地环境、抓取或补齐数据、定位外部依赖目录。

重点文件：

- [`setup/fetch_ashare_data.py`](setup/fetch_ashare_data.py)
  A 股数据抓取、补齐与 coverage 检查入口。
- [`setup/download_github_data.py`](setup/download_github_data.py)
  额外数据下载脚本。

### `测试与临时脚本/`

这个名字带有历史阶段痕迹，但目前它仍是 A 股主线最重要的 runner 区。

重点文件：

- [`测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](测试与临时脚本/run_teacher_first_ashare_ablation_train.py)
  当前最值得继续推进的入口，包含 `shortT_balanced_v1`、`shortT_balanced_v2` 等实验定义。
- [`测试与临时脚本/run_iterA5_ashare_train.py`](测试与临时脚本/run_iterA5_ashare_train.py)
  `iterA5` 正式训练 runner。
- [`测试与临时脚本/analyze_iterA5_ashare_results.py`](测试与临时脚本/analyze_iterA5_ashare_results.py)
  `iterA5` 结果分析与基线对照。
- [`测试与临时脚本/validate_iterA5_ths_proxy.py`](测试与临时脚本/validate_iterA5_ths_proxy.py)
  同花顺代理一致性验证。

`iterA1 ~ iterA4` 相关 runner 与 analyzer 主要用于阶段性回看、对照和复盘。

### `pipeline/`

这里保留更通用的研究脚本、旧管线与历史实验入口。

用途包括：

- 通用训练 / 回测
- 研究型分析
- 优化器与验证脚本

如果目标是继续当前 A 股主线，通常不建议从 `pipeline/` 直接开始改动，除非已经明确需要复用其中的公共逻辑。

### 根目录历史脚本

目录中还保留了一些较早阶段的验证脚本，例如：

- `02_math_model_test.py`
- `03_feature_engineering_verify.py`
- `04_comprehensive_verification.py`

它们更多反映项目早期或通用研究阶段的验证路径。

## 按任务找入口

### 准备 A 股数据

- [`setup/fetch_ashare_data.py`](setup/fetch_ashare_data.py)

### 跑当前推荐的 smoke / ablation

- [`测试与临时脚本/run_teacher_first_ashare_ablation_train.py`](测试与临时脚本/run_teacher_first_ashare_ablation_train.py)

### 跑 `iterA5` 正式训练参考

- [`测试与临时脚本/run_iterA5_ashare_train.py`](测试与临时脚本/run_iterA5_ashare_train.py)

### 看结果分析

- [`测试与临时脚本/analyze_iterA5_ashare_results.py`](测试与临时脚本/analyze_iterA5_ashare_results.py)
- [`测试与临时脚本/analyze_iterA4_ashare_results.py`](测试与临时脚本/analyze_iterA4_ashare_results.py)
- [`测试与临时脚本/analyze_iterA3_ashare_results.py`](测试与临时脚本/analyze_iterA3_ashare_results.py)

### 做终端代理验证

- [`测试与临时脚本/validate_iterA5_ths_proxy.py`](测试与临时脚本/validate_iterA5_ths_proxy.py)
- [`测试与临时脚本/validate_iterA4_ths_proxy.py`](测试与临时脚本/validate_iterA4_ths_proxy.py)

## 当前推荐工作流

1. 准备或补齐本地 A 股训练数据。
2. 先运行 teacher-first 系列 smoke 实验。
3. 结合分周期、gate、constraint 指标定位问题。
4. 只有在 smoke 通过后，再进入更重的 formal 训练。
5. 训练结论要和 THS 代理验证一起判断，不能只看单一 teacher 分数。

## 当前需要记住的判断

截至 `2026-04-07`：

- `iterA5` 是重要工程迭代，但不是当前默认升级候选。
- `shortT_breakout_v1` 主要是探索分支。
- 当前更值得继续验证的是 `shortT_balanced_v1` 与 `shortT_balanced_v2`。
