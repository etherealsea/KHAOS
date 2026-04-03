# KHAOS 同花顺公式 iterA1(A股) 单主线版

本目录当前只交付一个同花顺副图主指标：[KHAOS_THS_CORE.txt](./KHAOS_THS_CORE.txt)。

## 当前设计口径

这版当前使用：

- 一根不做额外平滑的一阶 `EKF` 价格预测主线
- 橙色：爆发
- 蓝色：多头趋势的反转
- 紫色：空头趋势的反转
- 灰色：非主导状态下的连续底线

## 这根预测线如何计算

显示主线采用价格域的一阶预测代理：

1. `PRICE_VEL = CLOSE - REF(CLOSE,1)`
   - 直接使用上一根到当前根的原始价格速度

2. `RHO_RAW`
   - 由 `HURST` 推导，决定“上一根速度”向下一根推进多少

3. `EKF_PRED = REF(CLOSE,1) + REF(RHO_RAW,1) * REF(PRICE_VEL,1)`
   - 用上一根真实价格，加上上一根真实速度乘上推进系数，得到当前预测价格

这里没有再对显示主线做 `EMA(CLOSE,19)` 之类的额外平滑。

## 这三种信号如何定义与表示

### 1. 爆发

定义：

- 先看压缩释放条件 `BK_RULE`
- 再叠加 `Compression / MLE / Trend / Pressure / Entropy Turn / Vol Quiet` 六类证据
- 得到 `BK_SCORE`
- 只有当 `BK_SCORE >= BK_EVT_TH`，并且 `BK_SCORE` 同时强于两类反转分支时，才判定为爆发

显示：

- 预测线继续画同一根 `EKF_PRED`
- 颜色覆写成橙色

### 2. 蓝色信号

定义：

- 蓝色表示“多头趋势的反转”
- 在实现上对应高位向下回归的分支，也就是：
  - `EMA_DIV > 0`
  - 且 `ENT <= ENT_BEAR` 或 `RET > 0`
- 然后叠加 `REV_SETUP / DNREV_ENT / DNREV_MOM / DNREV_CONFIRM / TREND`
- 再减去 `BK_SCORE` 和 `CONT_PRESS` 的惩罚，防止爆发阶段被误判成反转
- 最后只有当这一路分数胜过紫色分支，且超过 `RV_EVT_TH`，才显示蓝色

显示：

- 仍然画同一根 `EKF_PRED`
- 颜色覆写成蓝色

### 3. 紫色信号

定义：

- 紫色表示“空头趋势的反转”
- 在实现上对应低位向上回归的分支，也就是：
  - `EMA_DIV <= 0`
  - `EKF_RES <= RES_NODE`
  - `ENT <= ENT_BULL`
- 然后叠加 `REV_SETUP / UPREV_ENT / UPREV_CONFIRM / ENT_RISE / TREND`
- 同样减去 `BK_SCORE` 和 `CONT_PRESS` 惩罚
- 最后只有当这一路分数胜过蓝色分支，且超过 `RV_EVT_TH`，才显示紫色

显示：

- 仍然画同一根 `EKF_PRED`
- 颜色覆写成紫色

## 为什么现在主线不会断

- `MAIN_BASE` 灰色底线始终存在
- 橙 / 蓝 / 紫都只是对同一根线做状态覆写
- 所以不会因为 `DRAWNULL` 切换而整段消失
