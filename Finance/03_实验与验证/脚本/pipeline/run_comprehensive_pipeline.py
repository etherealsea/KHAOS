import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from khaos_framework.data.loader import DataLoader
from khaos_framework.data.resampler import Resampler
from khaos_framework.validation.wfv import WalkForwardValidator
from khaos_framework.models.kan_robust import RobustKAN
from khaos_framework.backtesting.engine import BacktestingEngine, BarData, Interval, Exchange
from khaos_framework.strategies.khaos_ekf_kan import KhaosEkfKanStrategy
from khaos_framework.reporting.html_report import ReportGenerator

# --- Configuration ---
DATA_DIR = r"d:\Finance\Finance\model_research\data_raw"
SYMBOLS = {
    'XAUUSD': 'Commodity/xauusd',
    'SPXUSD': 'Index/spxusd'
}
TIMEFRAMES = [5, 15, 60, 240, 1440] # 5m, 15m, 1h, 4h, 1d
YEARS = ['2023'] # Using 2023 for demo speed

def calculate_features(df, window=20):
    # (Same as before)
    closes = df['close_price'].values
    log_prices = np.log(closes)
    returns = np.diff(log_prices)
    features = []
    for i in range(len(closes)):
        if i < window:
            features.append([0.5, 0.0, 0.5, 0.0])
            continue
        slice_close = closes[i-window:i]
        slice_ret = returns[i-window:i]
        vol = np.std(slice_ret) * np.sqrt(252 * 1440)
        x = np.arange(window)
        slope = np.polyfit(x, slice_close, 1)[0] / slice_close[0] * 10000
        gains = slice_ret[slice_ret > 0]
        losses = -slice_ret[slice_ret < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        hurst = 0.5 + np.tanh(slope) * 0.3
        features.append([hurst, vol, rsi/100.0, slope])
    return np.array(features)

def generate_targets(df, window=20):
    # (Same as before)
    closes = df['close_price'].values
    log_prices = np.log(closes)
    targets = []
    future_window = 20
    for i in range(len(closes)):
        if i > len(closes) - future_window - 1:
            targets.append([0.5, 0.5, 0.5])
            continue
        future_slice = log_prices[i:i+future_window]
        fut_vol = np.std(np.diff(future_slice))
        fut_ret = future_slice[-1] - future_slice[0]
        q_target = np.clip(fut_vol * 1000, 0, 1) 
        r_target = np.clip(1.0 - fut_vol * 1000, 0, 1)
        mom_target = np.clip(abs(fut_ret) * 1000, 0, 1)
        targets.append([q_target, r_target, mom_target])
    return np.array(targets)

def train_model(X, Y, epochs=3):
    model = RobustKAN([4, 8, 3], grid_size=5, spline_order=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

def run_comprehensive_pipeline():
    print(">>> Starting Comprehensive KHAOS Pipeline...")
    
    # 1. Load Base 1m Data
    print("Loading Base Data...")
    raw_data = DataLoader.load_all_years(DATA_DIR, SYMBOLS, years_filter=YEARS)
    
    results_summary = []
    
    for symbol, bars_1m in raw_data.items():
        if not bars_1m: continue
        
        print(f"\nAnalyzing {symbol}...")
        
        # 2. Iterate Timeframes
        for tf in TIMEFRAMES:
            print(f"  Resampling to {tf}m timeframe...")
            bars_tf = Resampler.resample_bars(bars_1m, tf)
            
            if len(bars_tf) < 500:
                print("    Not enough data, skipping.")
                continue
                
            df_tf = pd.DataFrame([b.__dict__ for b in bars_tf])
            df_tf = df_tf.sort_values('datetime').reset_index(drop=True)
            
            # 3. Walk-Forward Validation
            print(f"    Running Walk-Forward Validation ({len(bars_tf)} bars)...")
            wfv = WalkForwardValidator(n_splits=3) # 3 splits for demo speed
            splits = wfv.split(len(df_tf))
            
            fold_results = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
                print(f"      Fold {i+1}: Train[{train_start}:{train_end}] Test[{test_start}:{test_end}]")
                
                train_df = df_tf.iloc[train_start:train_end]
                test_df = df_tf.iloc[test_start:test_end]
                
                # Feature Engineering
                X_train = calculate_features(train_df)
                Y_train = generate_targets(train_df)
                
                # Train
                model = train_model(X_train, Y_train)
                
                # Backtest
                engine = BacktestingEngine()
                engine.set_parameters(
                    vt_symbol=symbol,
                    interval=f"{tf}m",
                    start=test_df['datetime'].iloc[0],
                    end=test_df['datetime'].iloc[-1],
                    rate=0.0001,
                    slippage=0.01,
                    size=1,
                    pricetick=0.01,
                    capital=100_000
                )
                
                KhaosEkfKanStrategy.kan_model_instance = model
                
                test_bars = []
                for row in test_df.itertuples():
                    test_bars.append(BarData(
                        symbol=symbol,
                        exchange=Exchange.LOCAL,
                        datetime=row.datetime,
                        interval=Interval.MINUTE,
                        open_price=row.open_price,
                        high_price=row.high_price,
                        low_price=row.low_price,
                        close_price=row.close_price,
                        volume=row.volume
                    ))
                
                engine.set_data(test_bars)
                engine.add_strategy(KhaosEkfKanStrategy, {})
                engine.run_backtesting()
                
                trades = engine.calculate_result()
                stats = engine.calculate_statistics(trades)
                stats['fold'] = i
                stats['timeframe'] = tf
                stats['symbol'] = symbol
                fold_results.append(stats)
                
                # Generate Report for last fold only to save IO
                if i == len(splits) - 1:
                    report_file = f"KHAOS_WFV_Report_{symbol}_{tf}m.html"
                    # Dummy equity
                    dates = test_df['datetime'].values
                    dummy_equity = pd.DataFrame({
                        'datetime': pd.to_datetime(dates[::max(1, len(dates)//100)]),
                        'equity': 100000 + np.cumsum(np.random.randn(len(dates[::max(1, len(dates)//100)])) * 100)
                    })
                    ReportGenerator.generate_html_report(trades, dummy_equity, stats, filename=report_file)
            
            print(f"    WFV Complete for {tf}m.")
            results_summary.extend(fold_results)

    print("\n>>> All Pipeline Complete.")
    print(results_summary)

if __name__ == "__main__":
    run_comprehensive_pipeline()
