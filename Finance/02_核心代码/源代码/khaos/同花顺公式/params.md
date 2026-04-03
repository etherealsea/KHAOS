# KHAOS 同花顺 iterA1(A股) 参数说明

## 主线定义

当前主线是：

**不做额外平滑的一阶 EKF 价格预测线**

计算方式：

- `PRICE_VEL = CLOSE - REF(CLOSE,1)`
- `RHO_RAW` 由 `HURST` 推导
- `EKF_PRED = REF(CLOSE,1) + REF(RHO_RAW,1) * REF(PRICE_VEL,1)`

这表示：

- 以前一根真实价格为起点
- 用前一根真实速度做一步前推
- 用 `HURST` 决定速度延续强度

## 三种信号的实现细节

### 橙色：爆发

- `BK_RULE = VOL<=SIGMA_REF AND EKF_RES>RES_NODE AND EKF_RES<=0`
- `BK_SCORE` 综合压缩、MLE、趋势、压力、熵拐点、静波动六类证据
- `BK_SCORE >= BK_EVT_TH` 且分数强于两类反转分支，才显示橙色

### 蓝色：多头趋势的反转

- 蓝色语义：多头趋势被反转
- 数学上使用的是高位向下回归分支：`DNREV_RULE`
- 其核心特征是：`EMA_DIV > 0`，且高位回归证据成立
- 再叠加方向确认、动量确认和趋势确认
- 同时减去 breakout 压力，防止橙色环境被误吃掉

### 紫色：空头趋势的反转

- 紫色语义：空头趋势被反转
- 数学上使用的是低位向上回归分支：`UPREV_RULE`
- 其核心特征是：`EMA_DIV <= 0`、`EKF_RES <= RES_NODE`、`ENT <= ENT_BULL`
- 再叠加方向确认、熵确认和趋势确认
- 同样减去 breakout 压力，避免与爆发段混淆

## 显示方式

- `MAIN_BASE`
  - 灰色连续底线，始终显示
- `MAIN_BK`
  - 橙色覆写，表示爆发
- `MAIN_BLUE`
  - 蓝色覆写，表示多头趋势的反转
- `MAIN_PURPLE`
  - 紫色覆写，表示空头趋势的反转
