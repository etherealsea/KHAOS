# KHAOS-KAN 模型训练流程优化方案 (Training Optimization Plan)

## 问题总结 (Problem Summary)

在先前的训练尝试中，暴露出以下问题导致训练过程过久且容易中断/挂起：

1. **数据量庞大与冗余采样**：5m 和 15m 的高频数据包含数十万行，如果按顺序逐个 batch 训练，不仅耗时极长，且大部分高频震荡期（死水期）对物理相变的学习贡献度极低。此前的“快速测试模式”虽然粗暴截断了最后 2000 行，但牺牲了历史相变样本的多样性，不能作为正式训练方案。
2. **多资产串行训练效率低**：当前脚本采用外层循环串行遍历 15 个资产文件（`for data_path in final_files:`），每遇到一个文件就重新初始化 `DataLoader` 和走完 `epochs`。这种方式无法让模型在一个 epoch 内同时见识到跨资产、跨周期的特征分布。
3. **物理特征计算冗余**：在 `train_loader` 迭代内部，`physics(batch_x)` 被调用。这意味着对于重叠的滚动窗口，同一根 K 线的物理特征（如 EKF, MLE 等）被反复计算了多次，极大拖慢了前向传播速度。

## 优化方案 (Optimization Strategy)

在\*\*不牺牲任何物理学要求（对齐代理指标、临界慢化损失）和最终提取效果（符号公式提取），且强制保留近十年完整数据（无信息遗漏）\*\*的前提下，对训练流程进行系统性提速重构：

### 1. 物理特征的“离线全量预计算”与 GPU 加速 (GPU-Accelerated Offline Pre-computation)

* **改动逻辑**：将 `PhysicsLayer` 的计算从训练循环的 Forward 阶段彻底剥离。

* **具体做法**：

  * 在 `FinancialDataset.__init__` 阶段，将近十年的全量数据一次性转移到 **GPU (CUDA)** 上。

  * 利用 PyTorch 的纯张量向量化操作，一次性对整段全量序列计算物理特征（Hurst, MLE, Entropy, EKF 等）。

  * 计算完成后，将生成的完整 `[N, 8]` 特征矩阵保存在 GPU 显存或 CPU 内存中（视显存容量而定）。

* **收益**：彻底消除了滚动窗口造成的冗余重复计算，同时利用 GPU 的并行计算能力瞬间完成十年数据的特征构建，预计训练循环的计算耗时将下降 **90%** 以上。

### 2. 多资产多周期全局混合批处理 (Global Mixed Batching & Pinned Memory)

* **改动逻辑**：打破单一资产循环训练的壁垒，实现真正的全局多态训练，并优化数据流转。

* **具体做法**：

  * 遍历所有选定的资产文件，生成各自的 `Dataset`，保留全部近十年的时间序列，**绝不进行任何优先采样或人为截断过滤**。

  * 使用 `torch.utils.data.ConcatDataset` 将所有资产的连续样本池无缝合并。

  * 构建一个全局的 `DataLoader`。为了支撑近十年的海量数据吞吐，开启 `num_workers > 0` 和 `pin_memory=True`，加速 CPU 到 GPU 的数据传输。

* **收益**：模型在一个 Batch 内就能同时看到 Gold 的 1h 数据和 SPX 的 5m 数据的完整上下文。这不仅大幅提升了模型的泛化能力，还通过多进程和固定内存解决了 IO 瓶颈。

### 3. KAN 网络与注意力计算的并行优化 (Model-Level Acceleration)

* **改动逻辑**：优化模型前向传播的底层效率。

* **具体做法**：

  * 审查 `kan.py` 和 `attention.py`，确保所有的矩阵乘法和张量展开（Unfold）操作都在 GPU 上以最高效的方式运行。

  * 引入 `torch.amp.autocast` (自动混合精度 AMP)，在不损失精度的前提下，将计算密集型的 Forward 和 Backward 过程降至 FP16/BF16，进一步压榨 GPU 算力，节省显存并提速。

## 实施步骤 (Execution Steps)

1. **重构** **`physics.py`** **与** **`data_loader.py`**：

   * 编写 `compute_physics_features_bulk(data_tensor)` 函数，支持在 GPU 上对整条序列（如百万级 K 线）进行向量化计算。

   * 移除所有临时截断代码，确保加载近十年的完整数据，并在 `__init__` 中调用批量特征提取。
2. **重构** **`train.py`**：

   * 废弃串行文件循环，建立全局 `ConcatDataset` 和支持多进程及 `pin_memory` 的大 `DataLoader`。

   * 引入 `torch.cuda.amp.autocast` 和 `GradScaler` 实现混合精度训练。

   * 训练循环内直接提取 `[batch, window_size, 8]` 的预计算特征送入网络。
3. **验证与保存**：运行修改后的脚本，观察终端输出的 Batch 耗时是否达到毫秒级，确保十年全量数据的训练流程能够流畅跑通。

