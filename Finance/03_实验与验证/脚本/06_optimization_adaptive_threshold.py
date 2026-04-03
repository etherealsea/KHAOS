import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from khaos_model import KHAOS_Model
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df.set_index('Date', inplace=True)
        elif 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            df.set_index('Datetime', inplace=True)
        
        # Ensure numeric columns
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def backtest_strategy(df, model, strategy_type='fixed', param=0.8):
    """
    strategy_type: 'fixed' or 'adaptive'
    param: 
        - for 'fixed': threshold value (e.g., 0.8)
        - for 'adaptive': percentile (e.g., 0.95 for 95th percentile)
    """
    processed = model.process_data(df)
    khaos = processed['KHAOS']
    
    signals = pd.Series(0, index=processed.index)
    
    if strategy_type == 'fixed':
        threshold = param
        signals[khaos > threshold] = -1  # Short
        signals[khaos < -threshold] = 1   # Long
        upper_bound = pd.Series(threshold, index=processed.index)
        lower_bound = pd.Series(-threshold, index=processed.index)
        
    elif strategy_type == 'adaptive':
        # Rolling Quantile Threshold
        # Using a lookback window (e.g., 300 bars ~ 1 year for daily, or ~2 weeks for 1h)
        window = 300
        upper_bound = khaos.rolling(window=window, min_periods=50).quantile(param)
        lower_bound = khaos.rolling(window=window, min_periods=50).quantile(1 - param)
        
        # Signal logic: Cross the adaptive threshold
        signals[khaos > upper_bound] = -1
        signals[khaos < lower_bound] = 1
        
    # Filter consecutive signals (same direction)
    # Only take the first signal of a cluster or when signal changes
    # Shift 1 to compare with previous
    prev_signals = signals.shift(1).fillna(0)
    entries = signals[
        (signals != 0) & 
        (signals != prev_signals)
    ]
    
    return processed, entries, signals, upper_bound, lower_bound

def evaluate_entries(df, entries, holding_period=5):
    if entries.empty:
        return {
            'total_signals': 0,
            'win_rate': 0,
            'avg_return': 0,
            'profit_factor': 0
        }
    
    wins = 0
    total_return = 0
    gross_profit = 0
    gross_loss = 0
    
    for timestamp, signal in entries.items():
        if timestamp not in df.index:
            continue
            
        loc_idx = df.index.get_loc(timestamp)
        if loc_idx + holding_period >= len(df):
            continue
            
        entry_price = df['Close'].iloc[loc_idx]
        exit_price = df['Close'].iloc[loc_idx + holding_period]
        
        # Return calculation (Long: Exit-Entry, Short: Entry-Exit)
        if signal == 1: # Long
            ret = (exit_price - entry_price) / entry_price
        else: # Short
            ret = (entry_price - exit_price) / entry_price
            
        total_return += ret
        
        if ret > 0:
            wins += 1
            gross_profit += ret
        else:
            gross_loss += abs(ret)
            
    total_signals = len(entries)
    win_rate = wins / total_signals if total_signals > 0 else 0
    avg_return = total_return / total_signals if total_signals > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'total_signals': total_signals,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'profit_factor': profit_factor
    }

def run_comparison(file_path, asset_name):
    print(f"\n--- Optimizing {asset_name} ---")
    df = load_data(file_path)
    if df is None:
        return

    model = KHAOS_Model()
    
    # 1. Fixed Threshold (0.8)
    _, entries_fixed, _, _, _ = backtest_strategy(df, model, 'fixed', 0.8)
    stats_fixed = evaluate_entries(df, entries_fixed)
    
    # 2. Adaptive Threshold (95th Percentile)
    _, entries_adaptive, _, _, _ = backtest_strategy(df, model, 'adaptive', 0.95)
    stats_adaptive = evaluate_entries(df, entries_adaptive)
    
    # Print Comparison
    print(f"{'Metric':<20} | {'Fixed (0.8)':<15} | {'Adaptive (95%)':<15}")
    print("-" * 60)
    print(f"{'Signal Count':<20} | {stats_fixed['total_signals']:<15} | {stats_adaptive['total_signals']:<15}")
    print(f"{'Win Rate':<20} | {stats_fixed['win_rate']:.2%}<15 | {stats_adaptive['win_rate']:.2%}<15")
    print(f"{'Profit Factor':<20} | {stats_fixed['profit_factor']:.2f}<15 | {stats_adaptive['profit_factor']:.2f}<15")
    print(f"{'Avg Return':<20} | {stats_fixed['avg_return']:.4f}<15 | {stats_adaptive['avg_return']:.4f}<15")
    
    # Calculate frequency percentage
    total_bars = len(df)
    freq_fixed = (stats_fixed['total_signals'] / total_bars) * 100
    freq_adaptive = (stats_adaptive['total_signals'] / total_bars) * 100
    print(f"{'Frequency %':<20} | {freq_fixed:.2f}%{'':<9} | {freq_adaptive:.2f}%{'':<9}")

def main():
    # Define files
    files = [
        ("d:\\Finance\\Finance\\model_research\\data\\AAPL_1d.csv", "AAPL (Daily)"),
        ("d:\\Finance\\Finance\\model_research\\data\\SPX_Index_4h.csv", "SPX (4H)"),
        ("d:\\Finance\\Finance\\model_research\\data\\BTC_15m.csv", "BTC (15m)")
    ]
    
    print("Starting Optimization Verification...")
    for path, name in files:
        run_comparison(path, name)

if __name__ == "__main__":
    main()
