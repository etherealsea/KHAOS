import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. Configuration & Data Loading
# ==============================================================================
DATA_DIR = r"d:\Finance\Finance\data\Model_research\data"
ASSETS = ["BTC_5m.csv", "EURUSD_5m.csv", "Gold_ETF_5m.csv", "SPX_Index_5m.csv", "TSLA_5m.csv"]

# PI-KAN Weights (Current Best from KHAOS.pine)
# Q Weights
W_Q = {
    'hurst': 0.28885, 'vol': 0.00200, 'bias': -0.00222, 
    'slope': -0.00773, 'entropy': -0.00401, 'bias_c': 0.28012
}
# R Weights
W_R = {
    'hurst': -0.27829, 'vol': -0.01926, 'bias': 0.02291, 
    'slope': -0.03829, 'entropy': 0.02793, 'bias_c': 0.26812
}

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        # Rename 'date' to 'datetime' if needed
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ==============================================================================
# 2. Feature Engineering (Replicating KHAOS.pine Logic)
# ==============================================================================
def ema_z_tanh(series, span):
    ema = series.ewm(span=span, adjust=False).mean()
    std = series.ewm(span=span, adjust=False).std()
    z = (series - ema) / (std + 1e-9)
    return np.tanh(np.clip(z, -20, 20))

def calc_features(df):
    df = df.copy()
    df['log_price'] = np.log(df['close'])
    df['log_ret'] = df['log_price'].diff()
    
    # 1. Hurst Proxy
    def get_hurst(x):
        if len(x) < 2: return 0.5
        y = x
        X = np.arange(len(y))
        coeffs = np.polyfit(X, y, 1)
        slope = coeffs[0] * 10000
        return 0.5 + 0.3 * np.tanh(slope/10.0)
    
    df['hurst'] = df['log_price'].rolling(100).apply(get_hurst, raw=True)
    
    # 2. Volatility (StdDev of Returns)
    df['vol'] = df['log_ret'].rolling(20).std()
    
    # 3. Bias (Distance from EMA20)
    df['ema20'] = df['log_price'].ewm(span=20, adjust=False).mean()
    df['bias'] = (df['log_price'] - df['ema20']) * 100
    
    # 4. Slope (LinReg Slope of Price)
    def get_slope(x):
        if len(x) < 2: return 0
        coeffs = np.polyfit(np.arange(len(x)), x, 1)
        return coeffs[0]
    df['slope'] = df['log_price'].rolling(14).apply(get_slope, raw=True)
    
    # 5. Entropy (StdDev of Slope - simple proxy used in Pine)
    df['entropy'] = df['slope'].rolling(14).std()
    
    # Drop NaNs
    df = df.dropna()
    
    # Universal Normalization (Window = 20, as per current Pine)
    norm_win = 20
    df['n_hurst'] = ema_z_tanh(df['hurst'], norm_win)
    df['n_vol'] = ema_z_tanh(df['vol'], norm_win)
    df['n_bias'] = ema_z_tanh(df['bias'], norm_win)
    df['n_slope'] = ema_z_tanh(df['slope'], norm_win)
    df['n_entropy'] = ema_z_tanh(df['entropy'], norm_win)
    
    return df.dropna()

# ==============================================================================
# 3. KHAOS State Reconstruction
# ==============================================================================
def apply_pikan(df):
    # Calculate Q and R using known weights
    df['Q'] = (W_Q['hurst'] * df['n_hurst'] + W_Q['vol'] * df['n_vol'] + 
               W_Q['bias'] * df['n_bias'] + W_Q['slope'] * df['n_slope'] + 
               W_Q['entropy'] * df['n_entropy'] + W_Q['bias_c'])
    
    df['R'] = (W_R['hurst'] * df['n_hurst'] + W_R['vol'] * df['n_vol'] + 
               W_R['bias'] * df['n_bias'] + W_R['slope'] * df['n_slope'] + 
               W_R['entropy'] * df['n_entropy'] + W_R['bias_c'])
    
    # Physical scaling (ensure positive)
    df['Q'] = np.maximum(df['Q'], 0.01)
    df['R'] = np.maximum(df['R'], 0.01)
    
    # Regime Ratios
    df['QR_Ratio'] = df['Q'] / df['R']
    df['Diff_QR'] = df['Q'] - df['R']
    
    return df

# ==============================================================================
# 4. Target Definition: Volatility Expansion
# ==============================================================================
def define_targets(df):
    # Future Volatility (next 12 bars = 1 hour for 5m data)
    df['future_vol'] = df['vol'].shift(-12)
    
    # Expansion Ratio: Future Vol / Current Vol
    # We want to catch moments where volatility is low NOW but explodes LATER
    df['vol_expansion'] = df['future_vol'] / (df['vol'] + 1e-9)
    
    # Label: 1 if Volatility doubles (2.0x) within 1 hour, else 0
    # Also require that current volatility is not already extremely high (avoid catching peak-to-peak)
    # Filter: Current Vol < 75th percentile (we want to catch expansions from calm)
    vol_thresh = df['vol'].quantile(0.75)
    
    df['is_breakout'] = ((df['vol_expansion'] > 2.0) & (df['vol'] < vol_thresh)).astype(int)
    
    return df.dropna()

# ==============================================================================
# 5. Analysis
# ==============================================================================
def analyze_precursors(combined_df):
    print(f"Total Data Points: {len(combined_df)}")
    breakouts = combined_df[combined_df['is_breakout'] == 1]
    print(f"Total Breakouts Identified: {len(breakouts)}")
    
    if len(breakouts) == 0:
        print("No breakouts found with current criteria.")
        return

    # Compare States
    normal_state = combined_df[combined_df['is_breakout'] == 0].mean(numeric_only=True)
    breakout_state = breakouts.mean(numeric_only=True)
    
    print("\n--- State Comparison (Average) ---")
    print(f"{'Feature':<15} | {'Normal':<10} | {'Pre-Breakout':<10} | {'Diff %':<10}")
    print("-" * 55)
    
    features = ['n_hurst', 'n_vol', 'n_bias', 'n_slope', 'n_entropy', 'Q', 'R', 'QR_Ratio', 'Diff_QR']
    
    for f in features:
        norm_val = normal_state[f]
        break_val = breakout_state[f]
        diff = ((break_val - norm_val) / (abs(norm_val) + 1e-9)) * 100
        print(f"{f:<15} | {norm_val:10.4f} | {break_val:10.4f} | {diff:10.1f}%")

    # Save distribution plots
    # We want to see if there's a specific threshold for Q, R, or Entropy that separates them
    results_txt = "analysis_results.txt"
    with open(results_txt, "w") as f:
        f.write("Analysis Results\n")
        f.write("-" * 20 + "\n")
        for f_name in features:
            f.write(f"{f_name}: Normal={normal_state[f_name]:.4f}, Breakout={breakout_state[f_name]:.4f}\n")

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    all_data = []
    
    for asset_file in ASSETS:
        path = os.path.join(DATA_DIR, asset_file)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"Processing {asset_file}...")
        df = load_data(path)
        if df is not None:
            df = calc_features(df)
            df = apply_pikan(df)
            df = define_targets(df)
            df['asset'] = asset_file
            all_data.append(df)
            
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        analyze_precursors(combined_df)
    else:
        print("No data processed.")
