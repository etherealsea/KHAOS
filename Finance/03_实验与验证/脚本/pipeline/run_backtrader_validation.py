import backtrader as bt
import pandas as pd
import os
import sys
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.khaos_framework.strategies.backtrader_adapter import KhaosBacktraderStrategy
from src.khaos_framework.models.kan_robust import RobustKAN
from src.khaos_framework.data.loader import DataLoader
from src.khaos_framework.data.resampler import Resampler

# --- Helper to train a KAN model quickly for the validation ---
def train_temp_model(df):
    print("  [Validation] Training RobustKAN on sample data...")
    # Quick feature generation
    closes = df['close'].values
    log_prices = np.log(closes)
    returns = np.diff(log_prices)
    
    X, Y = [], []
    window = 20
    for i in range(window, len(closes)-20):
        # Features
        slice_ret = returns[i-window:i]
        vol = np.std(slice_ret) * 100
        slope = (closes[i] - closes[i-window])/closes[i-window] * 100
        X.append([0.5, vol, 0.5, slope]) # Simplified
        
        # Targets (Future Vol)
        fut_vol = np.std(returns[i:i+20]) * 100
        Y.append([fut_vol, 1.0/fut_vol, 0.5])
        
    X = np.array(X)
    Y = np.array(Y)
    
    model = RobustKAN([4, 8, 3], grid_size=5, spline_order=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Train
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(5):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    return model

def run_backtrader_validation():
    print(">>> Starting Professional Backtrader Validation...")
    
    # 1. Setup Cerebro
    cerebro = bt.Cerebro()
    
    # 2. Add Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 3. Settings (Professional Grade)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0001) # 1 bps
    cerebro.broker.set_slippage_perc(0.0001) # Slippage
    
    # 4. Load Data (XAUUSD 5m as example)
    # We will load raw CSV and convert to PandasData
    data_path = r"d:\Finance\Finance\data\processed\Commodity\XAUUSD_5m.csv"
    
    # Check if processed data exists, if not, use raw
    if not os.path.exists(data_path):
        print("  Processed data not found, loading raw 1m and resampling...")
        # Load raw
        raw_bars = DataLoader.load_all_years(r"d:\Finance\Finance\data\raw", {'XAUUSD': 'Commodity/xauusd'}, years_filter=['2023'])['XAUUSD']
        # Resample to 5m
        bars_5m = Resampler.resample_bars(raw_bars, 5)
        # Convert to DF
        df = pd.DataFrame([b.__dict__ for b in bars_5m])
        df = df.set_index('datetime')
        df.rename(columns={
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close'
        }, inplace=True)
    else:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # 5. Train Model
    model = train_temp_model(df)
    
    # 6. Feed Data to Cerebro
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # 7. Add Strategy
    cerebro.addstrategy(KhaosBacktraderStrategy, kan_model=model, print_log=True)
    
    # 8. Run
    print(f"  Initial Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strat = results[0]
    
    # 9. Output Metrics
    print("\n>>> Professional Validation Results <<<")
    print(f"  Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"  Sharpe Ratio: {sharpe.get('sharperatio', 0.0):.4f}")
    
    dd = strat.analyzers.drawdown.get_analysis()
    print(f"  Max Drawdown: {dd.max.drawdown:.2f}%")
    print(f"  Drawdown Duration: {dd.max.len} bars")
    
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.total.closed
    win_rate = trades.won.total / total_trades if total_trades > 0 else 0
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.2%}")
    
    # 10. Plot
    print("  Generating Plot...")
    try:
        # Save plot to file is tricky in Backtrader without display
        # We will skip plotting to file in this script to avoid GUI errors on headless
        # but the metrics are the key.
        pass
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_backtrader_validation()
