import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr

# ==============================================================================
# 1. Configuration & Data Loading
# ==============================================================================
DATA_DIR = r"d:\Finance\Finance\data\Model_research\data"
ASSETS = ["BTC_5m.csv", "EURUSD_5m.csv", "Gold_ETF_5m.csv", "SPX_Index_5m.csv", "TSLA_5m.csv"]

# PI-KAN Weights (Extracted from training_results.txt)
W_Q = {'hurst': 0.27964, 'vol': 0.04498, 'bias': -0.01884, 'slope': 0.00464, 'entropy': -0.03062, 'bias_c': 0.27959}
W_R = {'hurst': -0.27180, 'vol': 0.01335, 'bias': -0.03009, 'slope': 0.01662, 'entropy': -0.00309, 'bias_c': 0.26624}
W_M = {'hurst': -0.01652, 'vol': 0.02474, 'bias': -0.09982, 'slope': 0.03680, 'entropy': 0.01228, 'bias_c': 0.00889}

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns: df = df.rename(columns={'date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ==============================================================================
# 2. Feature Engineering
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
    
    # Hurst
    def get_hurst(x):
        if len(x) < 2: return 0.5
        y = x
        X = np.arange(len(y))
        coeffs = np.polyfit(X, y, 1)
        slope = coeffs[0] * 10000
        return 0.5 + 0.3 * np.tanh(slope/10.0)
    df['hurst'] = df['log_price'].rolling(100).apply(get_hurst, raw=True)
    
    # Volatility
    df['vol'] = df['log_ret'].rolling(20).std()
    
    # Bias (EMA20)
    df['ema20'] = df['log_price'].ewm(span=20, adjust=False).mean()
    df['bias'] = (df['log_price'] - df['ema20']) * 100
    
    # Slope
    def get_slope(x):
        if len(x) < 2: return 0
        coeffs = np.polyfit(np.arange(len(x)), x, 1)
        return coeffs[0]
    df['slope'] = df['log_price'].rolling(14).apply(get_slope, raw=True)
    
    # Entropy (Proxy)
    df['entropy'] = df['slope'].rolling(14).std()
    
    df = df.dropna()
    
    # Normalization (Window 100)
    norm_win = 100
    df['n_hurst'] = ema_z_tanh(df['hurst'], norm_win)
    df['n_vol'] = ema_z_tanh(df['vol'], norm_win)
    df['n_bias'] = ema_z_tanh(df['bias'], norm_win)
    df['n_slope'] = ema_z_tanh(df['slope'], norm_win)
    df['n_entropy'] = ema_z_tanh(df['entropy'], norm_win)
    
    return df.dropna()

# ==============================================================================
# 3. KHAOS Logic
# ==============================================================================
def apply_khaos(df):
    # Calculate Q, R, M
    for col, weights in zip(['Q', 'R', 'M'], [W_Q, W_R, W_M]):
        df[col] = (weights['hurst'] * df['n_hurst'] + weights['vol'] * df['n_vol'] + 
                   weights['bias'] * df['n_bias'] + weights['slope'] * df['n_slope'] + 
                   weights['entropy'] * df['n_entropy'] + weights['bias_c'])
    
    df['Q'] = np.maximum(df['Q'], 0.01)
    df['R'] = np.maximum(df['R'], 0.01)
    
    # Regime Definition
    df['is_flow'] = (df['Q'] > df['R'])
    df['is_chaos'] = (df['R'] > df['Q'])
    
    return df

# ==============================================================================
# 4. Strategy Simulation
# ==============================================================================
def run_simulation(df):
    # Simple Trend Strategy: Buy if Price > EMA20, Sell if Price < EMA20
    # Filter: Only trade if Regime is FLOW (Q > R)
    
    df['signal_base'] = np.where(df['log_price'] > df['ema20'], 1, -1)
    df['signal_khaos'] = np.where(df['is_flow'], df['signal_base'], 0) # Flat in Chaos
    
    # Returns
    df['ret_base'] = df['signal_base'].shift(1) * df['log_ret']
    df['ret_khaos'] = df['signal_khaos'].shift(1) * df['log_ret']
    
    # Metrics
    sharpe_base = df['ret_base'].mean() / (df['ret_base'].std() + 1e-9) * np.sqrt(288*365) # Annualized roughly
    sharpe_khaos = df['ret_khaos'].mean() / (df['ret_khaos'].std() + 1e-9) * np.sqrt(288*365)
    
    # Win Rate
    win_base = len(df[df['ret_base'] > 0]) / len(df[df['ret_base'] != 0])
    win_khaos = len(df[df['ret_khaos'] > 0]) / len(df[df['ret_khaos'] != 0]) if len(df[df['ret_khaos'] != 0]) > 0 else 0
    
    # Gravity Analysis
    # Does Extreme M predict Reversal?
    # Define "Extreme M" as top/bottom 5%
    m_high = df['M'].quantile(0.95)
    m_low = df['M'].quantile(0.05)
    
    # Signal: Contrarian
    df['sig_grav'] = 0
    df.loc[df['M'] > m_high, 'sig_grav'] = 1 # High M means Price is LOW (Restoring Force UP) -> BUY
    df.loc[df['M'] < m_low, 'sig_grav'] = -1 # Low M means Price is HIGH (Restoring Force DOWN) -> SELL
    
    df['ret_grav'] = df['sig_grav'].shift(1) * df['log_ret']
    sharpe_grav = df['ret_grav'].mean() / (df['ret_grav'].std() + 1e-9) * np.sqrt(288*365)
    
    print(f"Asset: {df['asset'].iloc[0]}")
    print(f"  Base Trend Sharpe: {sharpe_base:.2f} (Win: {win_base:.2f})")
    print(f"  KHAOS Filter Sharpe: {sharpe_khaos:.2f} (Win: {win_khaos:.2f}) -> Improvement: {sharpe_khaos - sharpe_base:.2f}")
    print(f"  Gravity Reversal Sharpe: {sharpe_grav:.2f}")
    
    return sharpe_base, sharpe_khaos, sharpe_grav

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    results = []
    for asset_file in ASSETS:
        path = os.path.join(DATA_DIR, asset_file)
        if not os.path.exists(path): continue
        
        df = load_data(path)
        if df is None: continue
        
        df = calc_features(df)
        df = apply_khaos(df)
        df['asset'] = asset_file
        
        results.append(run_simulation(df))

    if results:
        avg_base = np.mean([r[0] for r in results])
        avg_khaos = np.mean([r[1] for r in results])
        avg_grav = np.mean([r[2] for r in results])
        
        print("\n=== Global Findings ===")
        print(f"Avg Base Sharpe:  {avg_base:.2f}")
        print(f"Avg KHAOS Sharpe: {avg_khaos:.2f}")
        print(f"Avg Gravity Sharpe: {avg_grav:.2f}")
        
        with open("khaos_power_results.txt", "w") as f:
            f.write(f"Base_Sharpe: {avg_base:.2f}\n")
            f.write(f"KHAOS_Sharpe: {avg_khaos:.2f}\n")
            f.write(f"Gravity_Sharpe: {avg_grav:.2f}\n")
