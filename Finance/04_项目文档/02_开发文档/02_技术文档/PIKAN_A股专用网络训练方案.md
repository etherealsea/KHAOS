# PIKAN A 股专用网络训练方案

更新日期：2026-03-31

## 1. 任务目标

本方案的目标不是继续沿用当前多资产版本做局部修补，而是：

- 单独训练一个 **A 股专用** 的 PI-KAN / KHAOS 网络版本；
- 最终训练集 **只来自 A 股**；
- 保留 iter9 的最新双核架构与物理语义；
- 让模型先在 A 股内部学会“什么是真 breakout、什么是真 reversion”，再进入同花顺指标蒸馏阶段。

这意味着：

- 最终网络不得混入 BTC / ETH / XAU / WTI / SPX / ES 等非 A 股样本；
- 即便允许前期做小规模实验，**最终 checkpoint 必须是 A 股纯数据训练得到**；
- 下游同花顺指标的蒸馏母体，也应切换为这一版 A 股专用模型，而不是继续使用通用 iter9。

---

## 2. 为什么必须单独训练 A 股版本

当前 iter9 的训练主线覆盖：

- BTC / ETH
- ES / SPX
- XAU / WTI

这套数据分布和 A 股存在显著差异：

- A 股不是 24/7 连续交易；
- 存在午休断档；
- 存在涨跌停限制；
- 存在更高频的跳空、竞价和公告驱动；
- 部分个股存在长时间停牌、重组、复牌冲击；
- A 股短中周期上“假突破”和“超跌反抽”结构更常见。

因此，若继续用现有多资产 teacher 直接蒸馏到同花顺，风险是：

- breakout 规则学到的是海外连续市场的释放模式；
- reversion 规则学到的是非 A 股的失衡修复模式；
- 同花顺指标会被迫依赖大量校准和人为修补，难以保留 iter9 的原始规律。

结论：

**先训练一个 A 股专用 teacher，再从该 teacher 蒸馏到同花顺，才是正确顺序。**

---

## 3. 必须继承的 iter9 最新成果

本方案不回退到旧版 Force / Bias 单核框架，而是直接继承 iter9 的最新设计。

### 3.1 iter9 模型结构

iter9 当前核心结构位于：

`Finance/02_核心代码/源代码/khaos/模型定义/kan.py`

关键组件包括：

- `RevIN` 输入归一化
- `AttentionResidualBlock` 时序上下文建模
- `temporal_pool` 全局时序池化
- `breakout_local_pool` 最近局部 breakout 触发池
- `reversion_local_pool` 最近局部 reversion 触发池
- `state_gate` 全局共享状态门控
- `breakout_head` 与 `reversion_head` 双头 KAN 输出

这说明 iter9 学到的不是某个单一指标，而是：

- 全局上下文
- 局部触发锚点
- 双核状态竞争

### 3.2 iter9 输入特征

当前 physics 特征在：

`Finance/02_核心代码/源代码/khaos/核心引擎/physics.py`

共 14 维：

1. `Hurst`
2. `Volatility`
3. `EKF_Vel`
4. `EKF_Res`
5. `Entropy`
6. `MLE`
7. `Price_Mom`
8. `EMA_Div`
9. `Entropy_Delta`
10. `Entropy_Curv`
11. `MLE_Delta`
12. `EKF_Res_Delta`
13. `EMA_Div_Delta`
14. `Compression`

### 3.3 iter9 标签定义

当前标签构造在：

`Finance/02_核心代码/源代码/khaos/数据处理/data_loader.py`

iter9 的双核标签不是传统技术指标式标签。

#### breakout 标签关注：

- `path_efficiency`
- `directional_efficiency`
- `continuation_release`
- `adverse_penalty`
- `compression`
- `future volatility / entropy release`

也就是说，breakout 核学习的是：

**压缩后释放是否真实、是否延续、是否不是假启动。**

#### reversion 标签关注：

- `imbalance_strength`
- `imbalance_alignment`
- `terminal_confirmation`
- `reversal_quality`
- `continuation_excursion`
- `entropy_rise`

也就是说，reversion 核学习的是：

**当前是否已经形成极端失衡，并且开始进入真正有效的回归修复。**

### 3.4 iter9 损失约束

损失定义位于：

`Finance/02_核心代码/源代码/khaos/模型训练/loss.py`

iter9 明确在优化：

- `breakout_event_gap`
- `reversion_event_gap`
- `breakout_hard_negative`
- `reversion_hard_negative`
- `p4_reversion_setup`
- `p6_mle_chaos`
- `p7_csd`
- `p7_false_reversion`

这意味着：

- 网络在学习真 breakout 与伪 breakout 的边界；
- 网络在学习真 reversion 与伪 reversion 的边界；
- 网络在吸收混沌释放、临界慢化和结构转折这些物理规律。

### 3.5 iter9 当前汇报结果口径

基于当前 iter9 best checkpoint 的分析脚本：

`Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iter9_results.py`

可以得到两点重要结论：

- breakout 更偏 **中期高质量爆发过滤器**
- reversion 更偏 **广覆盖但确认型回归核**

这与开发日志一致：

- breakout 偏短中周期
- reversion 跨周期更稳

因此，A 股版本不应强行把两核训成同一种行为风格。

---

## 4. A 股专用版本的总体策略

### 4.1 总原则

本次训练采取：

- **架构继承 iter9**
- **数据切换为纯 A 股**
- **标签与 sample weighting 保持双核思路**
- **仅在 A 股微结构层做必要改造**

### 4.2 不做的事

- 不回退到旧版单核 Force 模型
- 不把目标改成传统涨跌预测器
- 不把标签改成 MACD / RSI 式简单规则
- 不把可视化主线定义反向灌进 teacher 网络训练

### 4.3 要做的事

- 让网络继续学习“相变规律”
- 让 breakout 继续学习“压缩后真实释放”
- 让 reversion 继续学习“极端失衡后的有效修复”
- 让标签与损失在 A 股市场里重新成立

---

## 5. 标的池设计

## 5.1 选股原则

首批 A 股训练池必须满足：

- 流动性足够高
- 历史长度尽量长
- 行业结构足够分散
- 避免 ST、壳股、长期停牌和极端题材股主导训练
- 优先选择沪深两市核心大盘股和行业龙头

## 5.2 首批正式训练标的（24 只）

### 金融与非银

- `600036` 招商银行
- `601166` 兴业银行
- `600030` 中信证券
- `601318` 中国平安

### 消费与家电

- `600519` 贵州茅台
- `000858` 五粮液
- `600887` 伊利股份
- `000333` 美的集团
- `600690` 海尔智家

### 周期、资源与基建

- `600309` 万华化学
- `601899` 紫金矿业
- `600031` 三一重工
- `600900` 长江电力
- `600028` 中国石化

### 科技与成长

- `300750` 宁德时代
- `002594` 比亚迪
- `002475` 立讯精密
- `002415` 海康威视
- `300059` 东方财富

### 医药与高端制造

- `600276` 恒瑞医药
- `300760` 迈瑞医疗
- `300124` 汇川技术
- `601012` 隆基绿能
- `603288` 海天味业

## 5.3 备用标的（如主标的存在长停牌或数据缺失）

- `000651` 格力电器
- `000725` 京东方A
- `601668` 中国建筑
- `601088` 中国神华
- `603986` 兆易创新
- `002142` 宁波银行

## 5.4 为什么这样选

这套标的池兼顾了：

- 银行、券商、保险
- 白酒、食品、家电
- 化工、资源、基建、电力
- 新能源、电子、软件/互联网金融
- 医药与工业自动化

它的目标不是押注某个行业，而是让模型在 A 股内部见到尽可能多样的：

- 趋势推进
- 假突破
- 极端失衡
- 回归修复
- 公告跳空
- 宽幅震荡

---

## 6. 数据集构建方案

## 6.1 数据来源优先级

当前仓库没有现成 A 股数据管线，因此本任务建议按以下优先级取数：

1. 用户已有本地 A 股 CSV / 同花顺导出数据
2. 可用的 iFinD / 同花顺数据接口
3. Tushare / AkShare 等公开数据源

原则是：

- 最终落地到 `training_ready` 的文件格式必须与现有 pipeline 兼容
- 中间取数工具可以更换，但训练输入格式必须统一

## 6.2 数据字段

强制要求字段：

- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`

可选保留：

- `amount`
- `turnover`
- `adj_factor`

## 6.3 复权口径

推荐使用：

- **前复权价格**

原因：

- 有利于长期连续训练
- 能减少分红送股造成的伪相变

体量类字段处理：

- 成交量保留原始口径
- 成交额可选保留但不作为首版核心输入

## 6.4 周期体系

### 第一版主训练周期

- `15m`
- `60m`
- `1d`

### 第二版补充周期

- `5m`

### 暂不作为第一版核心训练周期

- `240m`

原因：

- A 股单日交易时长与午休结构使 `240m` 在第一版中容易和日线语义重叠；
- 先把 `15m / 60m / 1d` 跑稳，更有利于让网络学到有效相变规律；
- `5m` 噪声更强，建议在主结构稳定后再加。

## 6.5 时间切分

建议按时间做严格顺序切分：

- Train：`2018-01-01` 至 `2023-12-31`
- Val：`2024-01-01` 至 `2024-12-31`
- Test：`2025-01-01` 至数据最新日期

如果分钟级数据无法覆盖到 2018：

- 允许分钟级从可获得的最早稳定年份开始
- 但必须确保 Train / Val / Test 的时间顺序不被打破

## 6.6 清洗规则

必须处理：

- 停牌导致的长缺口
- 复牌后的异常跳空
- 一字涨停 / 一字跌停
- 极端低成交量死盘阶段
- 新股上市前 250 个交易日样本不足问题

建议规则：

- 剔除 ST 与退市整理个股
- 上市不足 250 个交易日的样本暂不纳入第一版
- 一字板样本保留为状态参考，但默认不作为 breakout 正样本主来源
- 长停牌前后样本降低权重或屏蔽

---

## 7. 网络训练设计

## 7.1 总体要求

训练一个 **A 股专用版本**，命名建议：

- `iterA1`
- 或 `khaos_kan_ashare_iterA1`

## 7.2 架构策略

第一版采取：

- **完全继承 iter9 主架构**

即：

- 不改双头结构
- 不改单独 breakout / reversion heads
- 不改 attention + local pool + gate 的主骨架
- 不在第一版就新增 A 股特供结构层

原因：

- 先验证“纯 A 股数据 + iter9 双核架构”本身能否重新学出 A 股相变规律；
- 如果第一版就同时改数据、改架构、改标签，后续无法判断哪一项真正有效。

## 7.3 训练参数建议

第一版可直接以 iter9 为基准启动：

- `window_size = 20`
- `horizon = 10`
- `hidden_dim = 64`
- `layers = 3`
- `grid_size = 10`
- `epochs = 16`
- `batch_size = 256`
- `lr = 1e-3`
- `early_stop_patience = 4`

但建议额外加入：

- **按周期/标的分层采样**

避免：

- `15m` 样本量过大完全淹没 `60m / 1d`
- 个别超长历史标的主导训练

## 7.4 训练方式

最终正式版要求：

- **从头训练（scratch）**

可允许的中间实验：

- warm-start 仅用于调试 pipeline 是否可跑通

但正式交付 checkpoint 必须满足：

- 权重训练数据仅来自 A 股样本

---

## 8. A 股标签与损失的移植策略

## 8.1 breakout 标签如何移植

保持 iter9 的 breakout 思路不变：

- 关注压缩后释放
- 关注路径效率
- 关注方向效率
- 关注延续性
- 惩罚假启动和大幅回吐

但在 A 股中增加以下修正：

1. **涨跌停可交易性约束**
   - 一字板不作为 breakout 正样本核心来源
   - 非可成交极端条形不应主导 breakout 标签

2. **竞价/跳空约束**
   - 若未来位移主要来自隔夜跳空，而非连续释放，降低 breakout 目标权重

3. **量能死区过滤**
   - 低活跃样本不应轻易标成 breakout

## 8.2 reversion 标签如何移植

保持 iter9 的 reversion 思路不变：

- 关注极端失衡
- 关注失衡方向一致性
- 关注终端确认
- 关注修复质量优于继续延伸

但在 A 股中增加以下修正：

1. **跌停/涨停附近的不可交易阶段处理**
   - 极端状态可作为失衡参考
   - 但不能让无法成交的 bar 成为回归正样本唯一依据

2. **午休与日内断档处理**
   - 避免把交易制度导致的时间跳点误识别为回归确认

3. **公告驱动极端跳空处理**
   - 保留其状态信息
   - 但降低其对“可执行回归”标签的主导权

## 8.3 损失函数策略

第一版建议：

- 保持 `PhysicsLoss` 主体不变

只做两类必要修正：

1. A 股特定样本的 `sample_weights` 调整
2. 可交易性 mask 接入 event / hard negative 计算

不建议第一版就大改：

- `p3/p4/p6/p7`
- `event_gap_loss`
- `hard_negative_penalty`

因为这些正是 iter9 最新规律探寻成果的一部分，应该先保住。

---

## 9. 训练结果的目标口径

这次训练的目标不是“涨跌预测准确率最高”，而是：

- breakout 核在 A 股上重新学出真实释放
- reversion 核在 A 股上重新学出有效修复
- 保持双核语义清晰

## 9.1 结果评价不看单一指标

必须同时看：

- kernel-wise precision
- recall
- F1
- hard negative rate
- signal frequency
- label frequency
- OOS 分周期表现
- 按标的分组的稳定性

## 9.2 与当前汇报成果对齐的验收区间

当前内部汇报口径中，你已经提到：

- 爆发核在某些口径下约 `56%`
- 回归核在某些口径下约 `60%+`

本方案将其视为：

- **对齐区间**
- 而不是唯一单点指标

建议验收表述为：

- breakout 核在 A 股 OOS 上达到或逼近内部汇报的 `56%` 量级区间
- reversion 核在 A 股 OOS 上达到或逼近内部汇报的 `60%+` 量级区间
- 同时不能以牺牲大量 signal purity 或 kernel 语义混乱为代价

换句话说：

- 不接受“靠信号极少换高胜率”的假优化
- 不接受“回归和爆发学成同一种信号”的伪成功

---

## 10. 实施步骤

## 阶段 A：建立 A 股数据管线

输出：

- A 股标的清单
- 原始 CSV 数据
- 标准化后的 `training_ready` 文件

## 阶段 B：接通 A 股版 rolling dataset

输出：

- A 股专用 `create_rolling_datasets` 工作流
- A 股样本质量检查
- 时间切分检查

## 阶段 C：训练 `iterA1`

输出：

- A 股 scratch 训练日志
- best / final checkpoint

## 阶段 D：分析结果

输出：

- breakout_eval
- reversion_eval
- probe correlations
- 决策树规则摘要
- 按周期 / 按标的表现汇总

## 阶段 E：形成蒸馏准备集

输出：

- A 股 teacher 输出样本集
- 后续同花顺 student 蒸馏母体

---

## 11. 预期新增文件

建议在新对话里优先新增或生成以下文件：

### 代码侧

- `Finance/03_实验与验证/脚本/测试与临时脚本/run_iterA1_ashare_train.py`
- `Finance/03_实验与验证/脚本/测试与临时脚本/analyze_iterA1_ashare_results.py`
- `Finance/03_实验与验证/脚本/setup/fetch_ashare_data.py`

### 数据与配置侧

- `Finance/01_数据中心/03_研究数据/research_processed/training_ready/` 下的 A 股样本
- `Finance/04_项目文档/02_开发文档/02_技术文档/A股训练标的清单.md`

### 权重与报告侧

- `Finance/02_核心代码/模型权重备份/iterA1_ashare/`
- `Finance/04_项目文档/04_实验报告/KHAOS_A股_iterA1_Training_Report.md`

---

## 12. 这份方案给下一次 AI 对话的直接指令

新对话启动后，应按以下优先顺序推进：

1. 建立 A 股标的池与数据拉取/导入脚本
2. 生成标准化 `training_ready` 文件
3. 确认 A 股数据在现有 14 维 physics 特征流程下可正常运行
4. 训练 `iterA1` A 股 scratch 版本
5. 跑 `analyze_iterA1_ashare_results.py`
6. 输出双核评估与规则摘要
7. 再决定是否进入同花顺蒸馏

---

## 13. 最终判断

如果目标是把“latest iter9 的规律探寻成果”真正移植到同花顺，那么最重要的一步不是继续猜主线，而是先做：

**A 股专用 teacher 网络**

只有这一步完成后：

- breakout 才会学到 A 股自己的释放结构
- reversion 才会学到 A 股自己的极端修复结构
- 同花顺蒸馏才有可靠母体

一句话总结：

**先训练一个纯 A 股、iter9 同源的独立神经网络版本，再从它蒸馏同花顺指标，而不是继续拿通用 iter9 勉强适配 A 股。**
