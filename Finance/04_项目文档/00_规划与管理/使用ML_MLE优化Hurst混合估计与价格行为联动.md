## 目标
- 以最大似然估计和机器学习优化当前“BM混合Hurst + 价格行为评分”框架的参数与结构，使其更好地区分趋势/均值回归/震荡。
- 输出可在 Pine v6 中直接使用的固定参数表或分时间框架的参数映射函数。

## 数据与标注
- 数据来源：真实交易所K线（优先：加密主流币、外汇、股指）。使用Python抓取（如CCXT）或读取本地CSV，覆盖多时间框架（1m/5m/15m/1h/4h/1d）。
- 训练/验证划分：滚动窗口切分（按时间），确保严格时序。
- 价格行为标注（弱监督）：
  - 趋势：线性回归R2高、延续率高、零穿率低且突破后延续≥k根。
  - 均值回归：零穿率高、R2低、在极端带附近回跳成交。
  - 震荡：|H_centered|接近0、穿越频繁但非极端集中、延续一般。
- 可选：无监督HMM得到三类隐含状态，再用价格行为规则对齐为TR/MR/Chop。

## 特征工程
- Hurst四路径估计：GHE(去均值)、DFA、Haar/二次变差、改良R/S；参数：窗口组(H_len1/2/3)、更新周期(H_update_every)、收缩γ映射常数。
- 行为特征：
  - 趋势拟合：回归R2（窗口Lr）、延续率cont、零穿率zcr。
  - 形态/结构：突破成功率、极端带外的回收率、signature slope（不同采样步长的波动-步长斜率）。
  - 波动因子：归一化偏差`dev_smooth`、动能`energy_sym`、尾部指标（如过度长影线比率）。

## 模型与似然
- 判别层：三分类softmax模型`p = softmax(W·f + b)`；f含H混合与行为特征。
- 似然目标：最大化观测类别的对数似然（加权NLL，对类不均衡加权）；
  - 增强项：未来k步的延续一致性似然（趋势段持续、MR段回跳成功），权重λ平衡。
- 备选：贝叶斯优化直接最大化策略相关目标（如趋势段命中率、误报率、k步收益的Sharpe proxy）。

## 超参空间
- H组：H_len1/2/3（每TF取离散集合）、H_update_every、γ收缩映射系数（spread→γ的斜率范围）。
- 行为窗口：Lr、zcr平滑窗口、cont平滑窗口。
- 判别权重：W与b（softmax线性层）；可加L2正则与温度标定（calibration）。

## 训练流程（Python）
- 数据加载：`data_loader.py`（CCXT或CSV），统一OHLCV与TF转换。
- 估计器：`estimators.py` 实现四类H路径与混合函数（与Pine同式）。
- 特征：`features.py` 生成R2/cont/zcr等；`labelers.py` 规则标注或HMM对齐。
- 目标：`objective.py` 实现加权NLL与延续似然；
- 优化：
  - 小维度：Nelder–Mead或Powell对窗口与γ映射进行MLE；
  - 大维度：Optuna(TPE)或Bayesian Optimization，交叉验证滚动评估；早停与参数冻结。
- 评估：OOS AUC/F1、趋势段召回、误报率、k步延续率、零穿率对比。

## 结果落地（Pine可用）
- 固定参数表：按TF（1m/5m/15m/1h/4h/1d）输出`{H_len1/2/3, H_update_every, γ系数, Lr, 行为窗口, softmax系数(W,b)}` JSON。
- Pine集成：
  - 映射函数：根据`timeframe.multiplier`选择参数集；
  - 行为评分：在现有`p_trend/p_mr/p_chop`内部替换为`softmax(W·f+b)`；保持显示接口不变。
  - 若不引入矩阵常量：可将W,b简化为少量系数（如`α_H, α_R2, α_cont, α_zcr`）并硬编码为输入默认值。

## 校验与鲁棒化
- 资产泛化：跨资产交叉验证；对新资产使用已训参数评估并记录漂移。
- 厚尾与微结构防护：最小窗约束、winsorize极端值、签名斜率剔除微结构主导段。
- 稳健收缩：当分歧升高，γ自动收敛至0.5，防止假趋势。

## 交付物
- 代码：`train.py`、`estimators.py`、`features.py`、`labelers.py`、`objective.py`、`optimize.py`。
- 产物：`params.json`（TF参数包）与评估报告（指标表格与图）。
- Pine补丁：参数映射与判别层替换的最小变更patch（不改变UI）。

## 时间线
- 第1阶段：数据与标注（1–2天）
- 第2阶段：特征与估计器实现、初步MLE（2–4天）
- 第3阶段：Bayes优化与OOS评估（3–5天）
- 第4阶段：Pine集成与可视验证（1–2天）

## 可选增强
- 无监督HMM对齐提高标签质量；
- 在线微调：定期滚动再训练轻调`W,b`（保持窗口常量）；
- 结构分数：HH/HL、LH/LL形态一致性分数并入f特征。