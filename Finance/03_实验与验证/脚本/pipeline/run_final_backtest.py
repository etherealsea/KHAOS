import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from khaos_framework.backtesting.engine import BacktestingEngine, BarData, Interval, Exchange
from khaos_framework.strategies.khaos_ekf_kan import KhaosEkfKanStrategy
from khaos_quant_engine.khaos_ekf_core import KhaosAlgo

def run_backtest():
    # 1. 创建回测引擎
    engine = BacktestingEngine()
    
    # 2. 设置回测参数
    engine.set_parameters(
        vt_symbol="BTCUSDT.BINANCE",
        interval="1m",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        rate=0.0004, # 万4手续费
        slippage=0.0,
        size=1,
        pricetick=0.01,
        capital=100_000
    )
    
    # 3. 添加策略
    engine.add_strategy(KhaosEkfKanStrategy, {})
    
    # 4. 生成模拟历史数据 (BarData)
    print("生成回测数据...")
    # 使用之前的合成数据生成器，但转换为 BarData 格式
    prices = KhaosAlgo().prices # 这里需要稍微改一下，直接重新生成
    
    # 简单生成正弦波加趋势数据
    t = np.linspace(0, 100, 1000)
    raw_prices = 100 + t * 0.1 + np.sin(t) * 2 + np.random.normal(0, 0.5, 1000)
    
    history_data = []
    start_dt = datetime(2024, 1, 1)
    for i, p in enumerate(raw_prices):
        bar = BarData(
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE,
            datetime=start_dt + timedelta(minutes=i),
            interval=Interval.MINUTE,
            open_price=p,
            high_price=p + 0.5,
            low_price=p - 0.5,
            close_price=p,
            volume=100
        )
        history_data.append(bar)
        
    engine.set_data(history_data)
    
    # 5. 运行回测
    engine.run_backtesting()
    
    # 6. 计算结果
    df = engine.calculate_result()
    if not df.empty:
        print("\n--- 回测成交记录 ---")
        print(df[['datetime', 'direction', 'offset', 'price', 'volume']].head())
        
        stats = engine.calculate_statistics(df)
        print("\n--- 统计指标 ---")
        print(stats)
        
        df.to_csv("backtest_trades.csv")
        print("成交记录已保存至 backtest_trades.csv")
    else:
        print("无成交")

if __name__ == "__main__":
    run_backtest()
