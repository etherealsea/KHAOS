import os
import pandas as pd
import numpy as np
from datetime import timedelta

def generate_mock_ashare_data(output_dir, assets, start_date="2023-01-01", end_date="2025-06-01"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 模拟真实A股交易时间
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    timeframes = {
        '1d': {'freq': '1D', 'points_per_day': 1},
        '4h': {'freq': '4H', 'points_per_day': 1}, # 模拟每天一个240m聚合
        '1h': {'freq': '1H', 'points_per_day': 4},
        '15m': {'freq': '15min', 'points_per_day': 16},
        '5m': {'freq': '5min', 'points_per_day': 48}
    }
    
    for asset in assets:
        print(f"Generating mock data for {asset}...")
        
        # 生成日线级别的基准随机游走
        days = len(dates)
        daily_returns = np.random.normal(loc=0.0001, scale=0.02, size=days)
        daily_prices = 10.0 * np.exp(np.cumsum(daily_returns))
        
        for tf, tf_info in timeframes.items():
            points = tf_info['points_per_day']
            total_points = days * points
            
            # 插值生成高频价格
            if points == 1:
                prices = daily_prices
            else:
                # 简单线性插值 + 随机微小噪声
                x = np.arange(days)
                x_interp = np.linspace(0, days - 1, total_points)
                prices = np.interp(x_interp, x, daily_prices)
                noise = np.random.normal(0, 0.002, total_points)
                prices = prices * (1 + noise)
                
            # 构造 OHLCV
            df = pd.DataFrame()
            
            # 生成时间列 (只在交易日内生成时间点)
            if tf == '1d' or tf == '4h':
                df['time'] = dates
            else:
                times = []
                for d in dates:
                    # 模拟 9:30 - 11:30, 13:00 - 15:00
                    if tf == '1h':
                        times.extend([d + timedelta(hours=h, minutes=30) for h in [9, 10, 13, 14]])
                    elif tf == '15m':
                        for h, m in [(9,30), (10,0), (10,30), (11,0), (13,0), (13,30), (14,0), (14,30)]:
                            times.extend([d + timedelta(hours=h, minutes=m + i*15) for i in range(2)])
                    elif tf == '5m':
                        for h, m in [(9,30), (10,0), (10,30), (11,0), (13,0), (13,30), (14,0), (14,30)]:
                            times.extend([d + timedelta(hours=h, minutes=m + i*5) for i in range(6)])
                # 截断以匹配长度
                df['time'] = times[:total_points]
                
            # OHLC
            df['close'] = prices
            df['open'] = prices * (1 + np.random.normal(0, 0.001, total_points))
            df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.002, total_points)))
            df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.002, total_points)))
            df['volume'] = np.random.randint(10000, 1000000, total_points)
            
            # 确保 time 格式正确
            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存
            file_path = os.path.join(output_dir, f"{asset}_{tf}.csv")
            df.to_csv(file_path, index=False)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 将核心代码库加入环境变量以引入配置
    sys.path.append(str(Path(__file__).parents[4] / 'Finance' / '02_核心代码' / '源代码'))
    from khaos.数据处理.ashare_support import ASHARE_PRIMARY_ASSETS, ASHARE_FALLBACK_ASSETS
    
    target_dir = "Finance/01_数据中心/03_研究数据/research_raw/ashare/imports"
    
    all_assets = [f"{code}.SZ" for code in (ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS)]
    
    print(f"Generating mock data for {len(all_assets)} assets across 5 timeframes...")
    generate_mock_ashare_data(target_dir, all_assets)
    print(f"Done! Mock data saved to {target_dir}")
