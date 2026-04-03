import pandas as pd
import numpy as np
import os
from khaos_model import KHAOS_Model

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class StrategyEvaluator:
    def __init__(self, model):
        self.model = model
        
    def generate_signals(self, df, threshold=0.95, strategy='adaptive'):
        """
        Generate entry signals based on KHAOS extremes.
        Long: KHAOS < lower_dynamic_threshold
        Short: KHAOS > upper_dynamic_threshold
        """
        processed = self.model.process_data(df)
        
        # Define Signals
        processed['Signal'] = 0
        khaos = processed['KHAOS']
        
        if strategy == 'fixed':
            # Legacy Fixed Threshold (0.8)
            processed.loc[khaos > 0.8, 'Signal'] = -1  # Short
            processed.loc[khaos < -0.8, 'Signal'] = 1   # Long
        else:
            # Adaptive Threshold (Default)
            upper, lower = self.model.get_dynamic_thresholds(khaos, percentile=threshold)
            processed.loc[khaos > upper, 'Signal'] = -1
            processed.loc[khaos < lower, 'Signal'] = 1
        
        # Filter consecutive signals to avoid spamming (Take first signal of a cluster)
        # Shift(1) to see previous signal. If previous was same, ignore.
        # Note: In real trading, we might pyramid, but for validation, let's check unique entry events.
        processed['Signal_Change'] = processed['Signal'].diff()
        
        # Keep only new entries (Signal != 0 and changed from something else or 0)
        # Simplification: Only take signal if previous bar was not the same signal
        entries = processed[
            (processed['Signal'] != 0) & 
            (processed['Signal'] != processed['Signal'].shift(1))
        ].copy()
        
        return processed, entries

    def evaluate_performance(self, df, entries, holding_period=5):
        """
        Evaluate trade outcomes over a fixed holding period.
        Metrics: Win Rate, Return, Volume Ratio (Energy), Signal Freq
        """
        if entries.empty:
            return None

        results = []
        
        # Pre-calculate average volume for comparison
        # Using a rolling window to determine "normal" volume baseline
        df['Vol_MA_50'] = df['Volume'].rolling(50).mean()
        
        for idx, row in entries.iterrows():
            # Get index location
            try:
                loc = df.index.get_loc(idx)
            except KeyError:
                continue
                
            if loc + holding_period >= len(df):
                continue
                
            entry_price = row['Price']
            direction = row['Signal'] # 1 or -1
            
            # Exit Stats
            exit_row = df.iloc[loc + holding_period]
            exit_price = exit_row['Close'] # Assuming exit at Close of Nth bar
            
            # 1. Return Calculation
            raw_return = (exit_price - entry_price) / entry_price
            trade_return = raw_return * direction
            
            # 2. Volume/Energy Analysis
            # Calculate Volume Ratio during the trade vs Baseline
            trade_window_vol = df['Volume'].iloc[loc:loc+holding_period].mean()
            baseline_vol = row['Attention'] * df['Volume'].iloc[loc-20:loc].mean() # Using Attention as proxy or raw MA
            # Fallback to calculated MA if Attention logic is complex
            if np.isnan(baseline_vol) or baseline_vol == 0:
                 baseline_vol = df['Vol_MA_50'].iloc[loc]

            vol_ratio = trade_window_vol / (baseline_vol + 1e-6)
            
            # 3. Max Excursion (Drawdown/Runup)
            future_window = df.iloc[loc:loc+holding_period]
            if direction == 1: # Long
                max_favorable = (future_window['High'].max() - entry_price) / entry_price
                max_adverse = (future_window['Low'].min() - entry_price) / entry_price
            else: # Short
                max_favorable = (entry_price - future_window['Low'].min()) / entry_price
                max_adverse = (entry_price - future_window['High'].max()) / entry_price
                
            results.append({
                'Entry_Time': idx,
                'Signal': 'Long' if direction == 1 else 'Short',
                'Return': trade_return,
                'Win': trade_return > 0,
                'Vol_Ratio': vol_ratio,
                'Max_Favorable': max_favorable,
                'Max_Adverse': max_adverse
            })
            
        return pd.DataFrame(results)

def main():
    print("========================================================")
    print("   KHAOS Comprehensive Model Verification (Script v3)   ")
    print("========================================================")
    print("Metrics: Win Rate, Expectancy, Volume Energy, Frequency")
    print("Holding Period: 5 Bars (Short-term Reversion)")
    print("Signal Strategy: Adaptive Threshold (95th Percentile)")
    print("--------------------------------------------------------\n")

    model = KHAOS_Model()
    evaluator = StrategyEvaluator(model)
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    target_files = ['AAPL_1d.csv', 'EURUSD_1h.csv', 'BTC_15m.csv', 'SPX_Index_4h.csv', 'VIX_ETF_5m.csv']
    files_to_process = [f for f in target_files if f in files]

    if not files_to_process:
        print(f"No target files found in {DATA_DIR}.")
        return

    summary_stats = []

    for filename in files_to_process:
        print(f"Analyzing {filename}...")
        file_path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(file_path)
        
        # Standardize Date/Index
        if 'date' in df.columns:
            try: df['date'] = pd.to_datetime(df['date'], format='mixed')
            except: df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'Date' in df.columns:
            try: df['Date'] = pd.to_datetime(df['Date'], format='mixed')
            except: df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # 1. Generate Signals (Adaptive)
        processed_df, entry_signals = evaluator.generate_signals(df, threshold=0.95, strategy='adaptive')
        
        # 2. Evaluate
        trades = evaluator.evaluate_performance(df, entry_signals, holding_period=5)
        
        if trades is None or trades.empty:
            print(f"  > No signals generated for {filename}")
            continue
            
        # 3. Aggregate Statistics
        total_trades = len(trades)
        win_rate = trades['Win'].mean()
        avg_return = trades['Return'].mean()
        avg_vol_ratio = trades['Vol_Ratio'].mean()
        
        # Expectancy (Avg Win / Avg Loss)
        winning_trades = trades[trades['Win']]
        losing_trades = trades[~trades['Win']]
        avg_win = winning_trades['Return'].mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades['Return'].mean()) if not losing_trades.empty else 0
        profit_factor = (winning_trades['Return'].sum() / abs(losing_trades['Return'].sum())) if not losing_trades.empty and losing_trades['Return'].sum() != 0 else float('inf')
        
        # Frequency
        days_span = (df.index[-1] - df.index[0]).days
        if days_span == 0: days_span = 1
        freq_per_day = total_trades / days_span

        # Volume Confirmation (Win vs Loss Volume)
        # Do winning trades have higher volume support?
        vol_ratio_win = winning_trades['Vol_Ratio'].mean() if not winning_trades.empty else 0
        vol_ratio_loss = losing_trades['Vol_Ratio'].mean() if not losing_trades.empty else 0

        print(f"  > Trades: {total_trades} | Win Rate: {win_rate:.1%} | Profit Factor: {profit_factor:.2f}")
        print(f"  > Avg Return: {avg_return:.4%} | Vol Ratio (Win/Loss): {vol_ratio_win:.2f} / {vol_ratio_loss:.2f}")
        
        summary_stats.append({
            'Asset': filename.replace('.csv', ''),
            'Trades': total_trades,
            'Win Rate': f"{win_rate:.1%}",
            'Avg Return': f"{avg_return:.2%}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Freq (Daily)': f"{freq_per_day:.2f}",
            'Vol Ratio (All)': f"{avg_vol_ratio:.2f}",
            'Vol Ratio (Win)': f"{vol_ratio_win:.2f}"
        })

    # Display Final Report Table
    print("\n================ COMPREHENSIVE REPORT ================")
    stats_df = pd.DataFrame(summary_stats)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats_df)
    print("========================================================")
    print("Key Metrics Explanation:")
    print("- Win Rate: Percentage of trades with >0 return after 5 bars.")
    print("- Profit Factor: Gross Wins / Gross Losses. > 1.5 is good.")
    print("- Vol Ratio (Win): Average Volume during winning trades vs Baseline.")
    print("  > If Vol Ratio > 1.0, it means moves are supported by volume.")
    print("========================================================")

if __name__ == "__main__":
    main()
