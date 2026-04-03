
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import os
import glob

# ==============================================================================
# 1. Configuration
# ==============================================================================
DATA_DIR = r"d:\Finance\Finance\data\Model_research\data"
ASSETS = ["BTC", "EURUSD", "Gold_ETF", "SPX_Index", "TSLA", "AAPL", "AMZN", "MSFT", "NDX_Index", "Oil_ETF"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Physics Parameters
HURST_LEN = 100
EMA_LEN = 20
ENTROPY_LEN = 30
ZIGZAG_PERCENT = 0.02 # 2% move for reversal identification (adaptive per asset?)

# ==============================================================================
# 2. Helper Functions (Physics Engine)
# ==============================================================================
def tanh(x):
    return np.tanh(x)

def calc_hurst_proxy(price_log, window=100):
    """Calculates a proxy for Hurst using adaptive slope logic."""
    # Simplified Hurst proxy for speed: based on Volatility vs Range
    # Real Hurst is slow. We use the logic: High Trend = High Hurst.
    # We use Slope of Linear Regression / Volatility as a proxy.
    
    # Rolling Slope
    slope = price_log.diff(1).rolling(window).mean() * window
    # Rolling Volatility
    vol = price_log.diff(1).rolling(window).std() * np.sqrt(window)
    
    # Hurst Proxy: 0.5 + 0.5 * tanh(Slope/Vol) is direction, but Hurst is scalar [0,1].
    # Standard Hurst: Rescaled Range.
    # Fast Proxy: Efficiency Ratio (ER) = Net Change / Sum of Changes
    change = price_log.diff(window).abs()
    path = price_log.diff(1).abs().rolling(window).sum()
    er = change / (path + 1e-9)
    return er # ER is a good proxy for Hurst (1=Trend, 0=Noise)

def calc_entropy(series, window=30):
    """Calculates Permutation Entropy (approximate)."""
    # Simplified: Count sign changes in diff (Zero Crossing Rate)
    # Low ZCR = Low Entropy (Trend), High ZCR = High Entropy (Chaos)
    diff = series.diff(1)
    # 1 if sign changed, 0 otherwise
    sign_change = (diff * diff.shift(1) < 0).astype(int)
    entropy = sign_change.rolling(window).mean()
    # Normalize to [0, 1] (0.5 is random walk)
    return entropy

def zigzag_labels(price, threshold_pct=0.02):
    """
    Identifies peaks and valleys based on percentage threshold.
    Returns a series: 1 (Top), -1 (Bottom), 0 (None).
    """
    # Simple Pivot Logic
    labels = np.zeros(len(price))
    
    # We need a more robust ZigZag implementation or just use Forward Looking Window
    # "Is this point the highest in the next N bars and last N bars?"
    # Let's use a dynamic window based on volatility
    
    # Forward/Backward Window approach (more robust for ML labels)
    # A point is a Top if it's max of [t-N, t+N]
    # N should be dynamic, but let's fix it for simplicity or use ATR
    
    # Using fixed window for now, e.g., 20 bars (matches our EMA20 physics)
    window = 20
    is_top = (price == price.rolling(window*2 + 1, center=True).max())
    is_bottom = (price == price.rolling(window*2 + 1, center=True).min())
    
    labels[is_top] = 1
    labels[is_bottom] = -1
    
    # Filter: Only significant moves?
    # For now, raw pivots are fine, the model will learn which are predictable.
    return labels

def normalize_custom(series, window=100):
    """
    Applies the KHAOS Universal Normalization:
    Z-Score (EMA based) -> Tanh -> [-1, 1]
    """
    ema = series.ewm(span=window).mean()
    std = series.ewm(span=window).std()
    z = (series - ema) / (std + 1e-9)
    return np.tanh(z) # Soft clip to [-1, 1]

# ==============================================================================
# 3. Data Loading & Feature Engineering
# ==============================================================================
def load_and_prep_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)
        if 'time' in df.columns and 'datetime' not in df.columns:
            df.rename(columns={'time': 'datetime'}, inplace=True)
            
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Log Price for Physics
        df['log_price'] = np.log(df['close'])
        
        # 1. Features (The 5 Dimensions of KHAOS)
        # ---------------------------------------
        # A. Bias (Potential Energy): Distance from EMA20
        df['ema20'] = df['log_price'].ewm(span=EMA_LEN).mean()
        df['bias_raw'] = (df['log_price'] - df['ema20']) * 100
        
        # B. Volatility (Temperature): StdDev of Returns
        df['log_ret'] = df['log_price'].diff()
        df['vol_raw'] = df['log_ret'].rolling(20).std()
        
        # C. Hurst (Trend/MeanReversion State)
        df['hurst_raw'] = calc_hurst_proxy(df['log_price'], HURST_LEN)
        
        # D. Slope (Kinetic Direction)
        df['slope_raw'] = df['log_price'].diff(5) * 100
        
        # E. Entropy (Chaos/Disorder)
        df['entropy_raw'] = calc_entropy(df['log_price'], ENTROPY_LEN)
        
        # 2. Universal Normalization (The Lens)
        # -------------------------------------
        # We use a 100-bar window for normalization context
        norm_win = 100
        df['n_bias'] = normalize_custom(df['bias_raw'], norm_win)
        df['n_vol'] = normalize_custom(df['vol_raw'], norm_win)
        df['n_hurst'] = normalize_custom(df['hurst_raw'], norm_win)
        df['n_slope'] = normalize_custom(df['slope_raw'], norm_win)
        df['n_entropy'] = normalize_custom(df['entropy_raw'], norm_win)
        
        # 3. Labels (The Truth)
        # ---------------------
        # We want to predict Reversals. 
        # A Reversal is a point where price changes direction significantly.
        # Let's try to predict the "Distance to next Reversal" or "Reversal Probability"
        # Label: 1 if Top, -1 if Bottom (within lookahead window)
        
        # Use Forward Window Peak Detection
        fwd_window = 20
        # Is this the max price in [t-20, t+20]?
        # We only care about predicting it *before* it happens, but for training labels, 
        # we mark the actual peak. The features at t should predict if t is a peak.
        # Actually, if we want to predict "Reversal Imminent", we should label points 
        # leading up to the peak as 1.
        
        # Strict Definition: Actual Pivot Point
        df['is_top'] = (df['log_price'] == df['log_price'].rolling(window=fwd_window*2+1, center=True).max()).astype(int)
        df['is_bottom'] = (df['log_price'] == df['log_price'].rolling(window=fwd_window*2+1, center=True).min()).astype(int)
        
        # Combined Label: 1 (Top), -1 (Bottom), 0 (Noise)
        df['target'] = df['is_top'] - df['is_bottom']
        
        # Drop NaNs
        df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ==============================================================================
# 4. Training Pipeline
# ==============================================================================
def train_reversal_model():
    all_data = []
    
    print("Loading Data...")
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            # Construct path (assuming structure)
            # We use glob to find files matching asset and tf
            pattern = os.path.join(DATA_DIR, f"{asset}_*{tf}.csv")
            files = glob.glob(pattern)
            for f in files:
                print(f"  Processing {os.path.basename(f)}...")
                df = load_and_prep_data(f)
                if df is not None and len(df) > 500:
                    all_data.append(df)
    
    if not all_data:
        print("No data loaded!")
        return

    full_df = pd.concat(all_data)
    print(f"Total Data Points: {len(full_df)}")
    
    # Features & Target
    X = full_df[['n_hurst', 'n_vol', 'n_bias', 'n_slope', 'n_entropy']]
    y = full_df['target'] # -1, 0, 1
    
    # We want to learn the "Reversal Force" (M).
    # M should be positive for Tops (pushing down) and negative for Bottoms (pushing up)?
    # Wait, Gravity pulls BACK to center.
    # If Price is HIGH (Top), Gravity pulls DOWN (Negative Force).
    # If Price is LOW (Bottom), Gravity pulls UP (Positive Force).
    # So:
    # Top (High Bias) -> Target Force = -1 (Sell)
    # Bottom (Low Bias) -> Target Force = 1 (Buy)
    
    # Let's refine the target for Gravity M:
    # We want M to represent the "Restoring Force".
    y_force = -y # Tops (1) need Down force (-1). Bottoms (-1) need Up force (1).
    
    # However, our PI-KAN usually outputs a magnitude or a coefficient.
    # In KHAOS.pine, M is calculated as w*features.
    # If we want M to be the "Reversal Signal", we should train it to predict this Force.
    
    # Problem: Most points are 0 (Noise).
    # We can use Weighted Regression or Classification.
    # Let's use Ridge Regression to find linear weights that best approximate the Force.
    # This effectively creates a "Reversal Index".
    
    # Filter only Reversal points for training? 
    # No, we want it to output 0 when there is no reversal.
    
    # Weighting: Give high weight to Reversal points (rare class).
    sample_weight = np.abs(y_force) * 10 + 1 # Weight 11 for reversals, 1 for noise
    
    print("Training Ridge Regression (Linear KAN Layer)...")
    model = Ridge(alpha=1.0)
    model.fit(X, y_force, sample_weight=sample_weight)
    
    print("\n=== Learned Weights (The Gravity Equation) ===")
    coefs = model.coef_
    intercept = model.intercept_
    features = ['hurst', 'vol', 'bias', 'slope', 'entropy']
    
    for f, c in zip(features, coefs):
        print(f"  w_{f}: {c:.5f}")
    print(f"  bias_c: {intercept:.5f}")
    
    # Evaluation
    y_pred = model.predict(X)
    full_df['pred_force'] = y_pred
    
    # AUC for Top Detection (Pred Force < Threshold)
    # Top needs Negative Force. So predict_force should be low.
    # Let's invert for AUC calculation: Lower score = Higher Top Probability
    try:
        score_top = roc_auc_score(full_df['is_top'], -full_df['pred_force'])
        score_bottom = roc_auc_score(full_df['is_bottom'], full_df['pred_force'])
        print(f"\nModel Performance (AUC):")
        print(f"  Top Detection: {score_top:.4f}")
        print(f"  Bottom Detection: {score_bottom:.4f}")
    except:
        print("Could not calc AUC (maybe only one class)")

    # Save weights to file
    with open("khaos_reversal_weights.txt", "w") as f:
        f.write("M_Weights (Gravity/Reversal Force):\n")
        for feat, w in zip(features, coefs):
            f.write(f"w_m_{feat} = {w:.5f}\n")
        f.write(f"b_m = {intercept:.5f}\n")

if __name__ == "__main__":
    train_reversal_model()
