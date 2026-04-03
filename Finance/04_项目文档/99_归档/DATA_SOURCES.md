# 数据源配置说明文档 (Data Sources Configuration)

> **版本**: 1.0  
> **日期**: 2025-11-28  
> **状态**: 已验证 (Verified)

## 1. 概述
本项目采用**零密钥 (No-Key)** 数据获取策略，旨在降低部署门槛并提高系统的独立性。所有金融数据均直接从公开的CSV下载接口获取，不依赖OpenBB、FMP或AlphaVantage等需要API Key的服务。

---

## 2. 数据源详细配置

### 2.1 股票 (Stocks) & 外汇 (Forex) & 指数 (Indices)
- **提供方**: Stooq (stooq.com)
- **获取方式**: HTTP GET 请求 (CSV)
- **URL 模式**: `https://stooq.com/q/d/l/?s={SYMBOL}&i=d`
- **频率限制**: 建议请求间隔 > 3秒
- **覆盖范围**:
  - **美股**: AAPL.US, MSFT.US, AMZN.US, GOOGL.US, TSLA.US
  - **外汇**: EURUSD
  - **指数**: ^SPX (S&P 500), ^NDX (Nasdaq 100), ^DJI (Dow Jones)
  - **ETF**: GLD.US (黄金), USO.US (原油), VIXY.US (波动率)

### 2.2 加密货币 (Cryptocurrency)
- **提供方**: CryptoDataDownload (cryptodatadownload.com)
- **获取方式**: HTTP GET 请求 (CSV)
- **URL 模式**: `https://www.cryptodatadownload.com/cdd/{Exchange}_{Pair}_{Timeframe}.csv`
- **覆盖范围**:
  - **交易对**: BTCUSDT
  - **交易所**: Binance
  - **时间周期**: Daily (d), Hourly (1h)
  - **分钟级**: 正在接入 (minute)

---

## 3. 代理策略 (Proxy Strategy)
由于部分衍生品数据（期货、期权）难以获得免费且高质量的历史数据，本项目采用以下代理策略：

| 原始需求 | 代理标的 (Proxy) | 逻辑依据 |
| :--- | :--- | :--- |
| **股指期货 (ES, NQ)** | 现货指数 (^SPX, ^NDX) | 长期趋势高度一致，仅基差不同，不影响分形特征分析 |
| **商品期货 (GC, CL)** | 实物ETF (GLD, USO) | 追踪商品价格的高流动性ETF，能够准确反映价格波动 |
| **期权数据 (Options)** | 波动率ETF (VIXY) | 使用VIX短期期货ETF作为市场隐含波动率(IV)的观测窗口 |

---

## 4. 数据预处理标准
所有下载的数据在进入KHAOS模型前，均经过以下标准化处理：
1. **日期索引**: 统一转换为 `datetime` 对象，并设为 DataFrame 索引。
2. **格式兼容**: 使用 `pd.to_datetime(..., format='mixed')` 兼容不同源的日期格式（如 `YYYY-MM-DD` vs `YYYY-MM-DD HH:MM:SS`）。
3. **时间过滤**: 默认截取最近 10 年的数据。
4. **空值处理**: 剔除空行与元数据行。

---

## 5. 维护与扩展
- **添加新资产**: 在 `fetch_data_direct.py` 的列表中添加对应的 Stooq 代码或 CDD 交易对。
- **Stooq 代码查询**: 访问 [stooq.com](https://stooq.com) 搜索准确的代码（如 `.US` 后缀）。
- **故障排查**: 如遇 Stooq 返回 HTML 错误，通常是反爬虫触发，请检查 `User-Agent` 头或增加请求间隔。
