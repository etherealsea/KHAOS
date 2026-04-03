# Tasks
- [x] Task 1: 优化 KAN 正则化的数值稳定性
  - [x] SubTask 1.1: 修改 `Finance/02_核心代码/源代码/khaos/模型定义/kan.py` 的 `regularization_loss` 方法，将除法及对数运算中的平滑项从 `1e-8` 增大到 `1e-4`。
- [x] Task 2: 提升 RevIN 模块的混合精度兼容性
  - [x] SubTask 2.1: 修改 `Finance/02_核心代码/源代码/khaos/模型定义/revin.py` 的 `_get_statistics` 方法，确保 `torch.mean` 和 `torch.var` 使用 FP32 (`dtype=torch.float32`) 计算。
  - [x] SubTask 2.2: 在 `_normalize` 和 `_denormalize` 中，确保计算过程对 `float16` 的输入具备类型安全性，避免因极值产生 `NaN`。
- [x] Task 3: 强化训练脚本的梯度裁剪与 AMP 调度
  - [x] SubTask 3.1: 修改 `Finance/02_核心代码/源代码/khaos/模型训练/train.py`，配置 `autocast` 以优先尝试使用 `torch.bfloat16`（当硬件支持时）。
  - [x] SubTask 3.2: 优化 `scaler.unscale_(optimizer)` 后的梯度检查，确保即使偶然产生无效梯度，也能被安全跳过而不是引发程序崩溃。