
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ==============================================================================
# Configuration
# ==============================================================================
INPUT_DIR = r"d:\Finance\Finance\data\model_research\data_processed\training_ready"
OUTPUT_DIR = r"d:\Finance\Finance\data\model_research\data_with_pikan"
NORM_WIN = 20

# Weights (from KHAOS.pine / khaos_reversal_weights.txt)
# Q Weights (Trend Structure)
W_Q = {
    'hurst': 0.27964, 'vol': 0.04498, 'bias': -0.01884, 
    'slope': 0.00464, 'pe': -0.03062, 'b': 0.27959
}
# R Weights (Chaos)
W_R = {
    'hurst': -0.27180, 'vol': 0.01335, 'bias': -0.03009, 
    'slope': 0.01662, 'pe': -0.00309, 'b': 0.26624
}
# M Weights (Gravity - Trained with Sign Change Entropy)
W_M = {
    'hurst': -0.01323, 'vol': 0.02305, 'bias': -0.41245, 
    'slope': -0.13026, 'pe': -0.00603, 'b': -0.00799
}

# ==============================================================================
# Helpers
# ==============================================================================
def calc_hurst_proxy(price_log, window=100):
    change = price_log.diff(window).abs()
    path = price_log.diff(1).abs().rolling(window).sum()
    er = change / (path + 1e-9)
    return er

def calc_entropy_sign(series, window=30):
    diff = series.diff(1)
    sign_change = (diff * diff.shift(1) < 0).astype(int)
    entropy = sign_change.rolling(window).mean()
    return entropy

def calc_entropy_stdev(series, window=14):
    return series.rolling(window).std()

def normalize_universal(series, window):
    ema = series.ewm(span=window, adjust=False).mean()
    # EWM StdDev approximation
    # std = sqrt( ewm(x^2) - ewm(x)^2 ) is not exact for EWM but close
    # Using pandas rolling std for simplicity or standard ewm std
    std = series.ewm(span=window, adjust=False).std()
    z = (series - ema) / (std + 1e-9)
    return np.tanh(z)

def process_file(file_path):
    df = pd.read_csv(file_path)
    
    # Ensure standard columns
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    
    if 'close' not in df.columns:
        print(f"Skipping {file_path}: No close column")
        return None
        
    # Pre-calc basics
    df['log_price'] = np.log(df['close'])
    df['log_ret'] = df['log_price'].diff()
    
    # Features
    # 1. Hurst
    df['hurst_raw'] = calc_hurst_proxy(df['log_price'], 100)
    
    # 2. Volatility (Stdev of log ret)
    df['vol_raw'] = df['log_ret'].rolling(20).std()
    
    # 3. Bias (Log Price - EMA20)
    ema20 = df['log_price'].ewm(span=20, adjust=False).mean()
    df['bias_raw'] = (df['log_price'] - ema20) * 100.0
    
    # 4. Slope (Log Price change 5 bars)
    df['slope_raw'] = (df['log_price'] - df['log_price'].shift(5)) * 100.0
    
    # 5. Entropy (Two versions)
    # PE Sign (for M) - window 30 (from training script default)
    df['pe_sign_raw'] = calc_entropy_sign(df['log_price'], 30)
    # PE Stdev (for Q/R) - window 14 on slope (from Pine)
    df['pe_stdev_raw'] = calc_entropy_stdev(df['slope_raw'], 14)
    
    # Normalize Features (Universal Normalization window=20)
    df['n_hurst'] = normalize_universal(df['hurst_raw'].fillna(0.5), NORM_WIN).fillna(0)
    df['n_vol'] = normalize_universal(df['vol_raw'].fillna(0), NORM_WIN).fillna(0)
    df['n_bias'] = normalize_universal(df['bias_raw'].fillna(0), NORM_WIN).fillna(0)
    df['n_slope'] = normalize_universal(df['slope_raw'].fillna(0), NORM_WIN).fillna(0)
    
    df['n_pe_sign'] = normalize_universal(df['pe_sign_raw'].fillna(0.5), NORM_WIN).fillna(0)
    df['n_pe_stdev'] = normalize_universal(df['pe_stdev_raw'].fillna(0), NORM_WIN).fillna(0)
    
    # Calculate Outputs
    # Q (Process Noise) - uses pe_stdev
    df['q_val'] = (
        W_Q['hurst'] * df['n_hurst'] + 
        W_Q['vol'] * df['n_vol'] + 
        W_Q['bias'] * df['n_bias'] + 
        W_Q['slope'] * df['n_slope'] + 
        W_Q['pe'] * df['n_pe_stdev'] + 
        W_Q['b']
    )
    # R (Measurement Noise) - uses pe_stdev
    df['r_val'] = (
        W_R['hurst'] * df['n_hurst'] + 
        W_R['vol'] * df['n_vol'] + 
        W_R['bias'] * df['n_bias'] + 
        W_R['slope'] * df['n_slope'] + 
        W_R['pe'] * df['n_pe_stdev'] + 
        W_R['b']
    )
    
    # M (Gravity) - uses pe_sign
    df['m_weight'] = (
        W_M['hurst'] * df['n_hurst'] + 
        W_M['vol'] * df['n_vol'] + 
        W_M['bias'] * df['n_bias'] + 
        W_M['slope'] * df['n_slope'] + 
        W_M['pe'] * df['n_pe_sign'] + 
        W_M['b']
    )
    
    # Save valid columns
    out_cols = [c for c in df.columns if 'n_' not in c and 'raw' not in c and 'log_' not in c]
    # Keep m_weight, q_val, r_val, and original columns
    keep_cols = list(df.columns[:6]) + ['m_weight', 'q_val', 'r_val', 'hurst_raw', 'vol_raw'] 
    # Also keep x_vol/x_hurst for dynamic optimization (normalize them?)
    # optimize_afc expects 'x_vol' and 'x_hurst' as normalized features for dynamic logic.
    # In optimize_afc: 
    # df['x_vol'] = normalize_custom(df['vol_raw'], 300)
    # df['x_hurst'] = normalize_custom(df['hurst_raw'], 300)
    # So I can just pass 'vol_raw' and 'hurst_raw' and let optimize_afc normalize them?
    # Yes, optimize_afc calculates x_vol/x_hurst itself.
    # But I should provide 'vol_raw' and 'hurst_raw' clearly.
    # Actually optimize_afc calculates vol_raw itself if not present?
    # optimize_afc: df['vol_raw'] = df['close'].pct_change().rolling(20).std()
    # optimize_afc: df['hurst_raw'] = calc_hurst_proxy(...)
    
    # So I just need to save the dataframe with 'm_weight', 'q_val', 'r_val'.
    
    return df

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files in {INPUT_DIR}")
    
    for f in tqdm(files):
        try:
            df_out = process_file(f)
            if df_out is not None:
                basename = os.path.basename(f)
                out_path = os.path.join(OUTPUT_DIR, basename)
                df_out.to_csv(out_path, index=False)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()
