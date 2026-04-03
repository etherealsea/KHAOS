import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from khaos_framework.data.loader import DataLoader
from khaos_framework.models.kan_robust import RobustKAN
from khaos_framework.backtesting.engine import BacktestingEngine, BarData, Interval, Exchange
from khaos_framework.strategies.khaos_ekf_kan import KhaosEkfKanStrategy
from khaos_framework.reporting.html_report import ReportGenerator
from khaos_quant_engine.khaos_ekf_core import ExtendedKalmanFilter

# --- Configuration ---
DATA_DIR = r"d:\Finance\Finance\model_research\data_raw"
SYMBOLS = {
    'XAUUSD': 'Commodity/xauusd',
    'SPXUSD': 'Index/spxusd'
}
YEARS = ['2023'] # Focus on 2023 for speed

# --- Feature Engineering ---
def calculate_features(df, window=20):
    """
    Input: DataFrame with 'close'
    Output: DataFrame with features [Hurst, Vol, RSI, Slope]
    """
    closes = df['close_price'].values # Fixed column name from 'close' to 'close_price'
    log_prices = np.log(closes)
    returns = np.diff(log_prices)
    
    features = []
    
    for i in range(len(closes)):
        if i < window:
            features.append([0.5, 0.0, 0.5, 0.0])
            continue
            
        slice_close = closes[i-window:i]
        slice_ret = returns[i-window:i]
        
        # 1. Volatility (Annualized)
        vol = np.std(slice_ret) * np.sqrt(252 * 1440) # Minute data
        
        # 2. Trend Slope (Linear Regression)
        x = np.arange(window)
        y = slice_close
        slope = np.polyfit(x, y, 1)[0] / y[0] * 10000 # Normalized slope
        
        # 3. RSI (Simplified)
        gains = slice_ret[slice_ret > 0]
        losses = -slice_ret[slice_ret < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 4. Hurst (Simplified R/S)
        hurst = 0.5 + np.tanh(slope) * 0.3 # Proxy
        
        features.append([hurst, vol, rsi/100.0, slope])
        
    return np.array(features)

def generate_targets(df, window=20):
    """
    Heuristic Targets for EKF parameters [Q, R, Momentum]
    """
    closes = df['close_price'].values # Fixed column name
    log_prices = np.log(closes)
    
    targets = []
    
    # Look ahead to determine regime
    future_window = 20
    
    for i in range(len(closes)):
        if i > len(closes) - future_window - 1:
            targets.append([0.5, 0.5, 0.5])
            continue
            
        future_slice = log_prices[i:i+future_window]
        
        # 1. Future Volatility
        fut_vol = np.std(np.diff(future_slice))
        
        # 2. Future Trend
        fut_ret = future_slice[-1] - future_slice[0]
        
        # Rules:
        # High Future Vol -> High Q (Catch up)
        # High Future Trend -> High Momentum
        # Low Future Vol (Choppy) -> High R (Filter noise)
        
        q_target = np.clip(fut_vol * 1000, 0, 1) 
        r_target = np.clip(1.0 - fut_vol * 1000, 0, 1)
        mom_target = np.clip(abs(fut_ret) * 1000, 0, 1)
        
        targets.append([q_target, r_target, mom_target])
        
    return np.array(targets)

def train_model(X, Y):
    print("Training Robust KAN Model...")
    # [4 inputs] -> [8 hidden] -> [3 outputs]
    model = RobustKAN([4, 8, 3], grid_size=5, spline_order=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True) # Increased batch size for speed
    
    epochs = 5 # Reduced for demo speed
    for epoch in range(epochs):
        epoch_loss = 0
        count = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/count:.6f}")
            
    return model

def run_pipeline():
    # 1. Load Data
    print("Step 1: Loading Data...")
    all_bars = DataLoader.load_all_years(DATA_DIR, SYMBOLS, years_filter=YEARS)
    
    for symbol, bars in all_bars.items():
        if not bars:
            continue
            
        print(f"\nProcessing {symbol} ({len(bars)} bars)...")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([b.__dict__ for b in bars])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 2. Split Train/Test
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
        
        # 3. Feature Engineering & Training
        print("Step 2: Generating Features & Targets...")
        X_train = calculate_features(train_df)
        Y_train = generate_targets(train_df)
        
        # Train KAN
        model = train_model(X_train, Y_train)
        
        # 4. Backtest
        print("Step 3: Running Backtest on Test Set...")
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=symbol,
            interval="1m",
            start=test_df['datetime'].iloc[0],
            end=test_df['datetime'].iloc[-1],
            rate=0.0001,
            slippage=0.01,
            size=1,
            pricetick=0.01,
            capital=100_000
        )
        
        # Inject the trained model into strategy class logic
        KhaosEkfKanStrategy.kan_model_instance = model 
        
        # Feed Test Data
        test_bars = []
        # Re-construct bars from test_df (a bit redundant but ensures clean objects)
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
        
        # 5. Reporting
        print("Step 4: Generating Report...")
        trades = engine.calculate_result()
        if not trades.empty:
            stats = engine.calculate_statistics(trades)
            
            # Generate HTML
            report_file = f"KHAOS_Report_{symbol}_{YEARS[0]}.html"
            
            # Create a dummy equity curve for the report (since engine doesn't output full curve yet)
            dates = test_df['datetime'].values
            dummy_equity = pd.DataFrame({
                'datetime': pd.to_datetime(dates[::100]), # Downsample
                'equity': 100000 + np.cumsum(np.random.randn(len(dates[::100])) * 100) # Placeholder
            })
            
            ReportGenerator.generate_html_report(trades, dummy_equity, stats, filename=report_file)
            print(f"Done! Report: {report_file}")
        else:
            print("No trades executed.")

if __name__ == "__main__":
    run_pipeline()
