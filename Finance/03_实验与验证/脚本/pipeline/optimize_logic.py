
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import gc

# ==============================================================================
# 1. Configuration
# ==============================================================================
DATA_DIR = r"d:\Finance\Finance\data\model_research\data_with_pikan"
OUTPUT_FILE = "logic_optimization_results.csv"

# Fixed Parameters for Logic Comparison (Based on previous best)
CALIB_WINDOW = 1000
SIGMA_TRIG = 1.8
SIGMA_EXIT = 0.8

# Logic Variants to Test
LOGIC_TYPES = ['peak_detect']
THRESHOLDS = [1.5, 1.6, 1.65, 1.7, 1.8]

# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def normalize_custom_safe(series, window=300):
    """Z-Score Normalization to [-1, 1] using Tanh"""
    ema = series.ewm(span=window).mean()
    std = series.ewm(span=window).std()
    z = (series - ema) / (std + 1e-9)
    return np.tanh(z).fillna(0)

def zigzag_labels(price, window=20):
    labels = np.zeros(len(price))
    is_top = (price == price.rolling(window*2 + 1, center=True).max())
    is_bottom = (price == price.rolling(window*2 + 1, center=True).min())
    labels[is_top] = -1
    labels[is_bottom] = 1
    return labels

def load_file_and_prep(file_path):
    try:
        df = pd.read_csv(file_path)
        req_cols = ['m_weight', 'close']
        if not all(col in df.columns for col in req_cols):
            return None
        
        df['m_weight'] = df['m_weight'].fillna(0)
        df['y_target'] = zigzag_labels(df['close'], window=20)
        
        # Calculate Base Gravity (Static Window for fairness)
        m_raw = df['m_weight']
        mean = m_raw.ewm(span=CALIB_WINDOW, adjust=False).mean()
        std = m_raw.ewm(span=CALIB_WINDOW, adjust=False).std().fillna(1e-6)
        
        z = (m_raw - mean) / (std + 1e-9)
        df['gravity'] = np.tanh(z)
        
        # Calculate Price Momentum for Logic 4
        df['log_price'] = np.log(df['close'])
        df['mom'] = df['log_price'].diff(5) # 5-bar momentum
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def evaluate_signals(signals, targets, price):
    """
    Evaluate signals based on:
    1. Precision (True Positives / Total Signals)
    2. Timing Error (Distance to nearest ZigZag pivot)
    """
    # Identify Signal Events (Rising Edges)
    sig_events = (signals != 0) & (np.roll(signals, 1) == 0)
    sig_indices = np.where(sig_events)[0]
    
    if len(sig_indices) == 0:
        return 0, 0, 0
    
    target_indices = np.where(targets != 0)[0]
    if len(target_indices) == 0:
        return 0, len(sig_indices), 0
        
    tp_count = 0
    total_dist = 0
    
    # Match each signal to NEAREST future target
    for idx in sig_indices:
        sig_type = signals[idx] # 1 or -1
        
        # Find future targets of SAME type
        future_targets = target_indices[target_indices > idx]
        
        # Filter by type
        valid_targets = []
        for t_idx in future_targets:
            if targets[t_idx] == sig_type:
                valid_targets.append(t_idx)
                break # Only need the first one
        
        if valid_targets:
            t_idx = valid_targets[0]
            dist = t_idx - idx
            
            # If distance is reasonable (e.g. within 50 bars)
            if dist < 50:
                tp_count += 1
                total_dist += dist
    
    precision = tp_count / len(sig_indices)
    avg_lead = total_dist / tp_count if tp_count > 0 else 0
    
    return precision, avg_lead, len(sig_indices)

# ==============================================================================
# 3. Logic Implementations
# ==============================================================================
def run_logic_entry(df):
    """Logic 1: Trigger immediately when crossing Threshold"""
    thresh = np.tanh(SIGMA_TRIG)
    g = df['gravity'].values
    
    signals = np.zeros(len(g))
    signals[g > thresh] = 1 # Bot Signal (High Upward Force)
    signals[g < -thresh] = -1 # Top Signal (High Downward Force)
    
    # Edge detection to make them "Events" not "Zones"
    events = np.zeros(len(g))
    events[(signals == 1) & (np.roll(signals, 1) != 1)] = 1
    events[(signals == -1) & (np.roll(signals, 1) != -1)] = -1
    return events

def run_logic_exit(df):
    """Logic 2: Trigger when EXITING the zone (Hysteresis)"""
    thresh = np.tanh(SIGMA_TRIG)
    exit_thresh = np.tanh(SIGMA_EXIT)
    g = df['gravity'].values
    
    # State Machine
    state = np.zeros(len(g)) # 0: Neutral, 1: In Bot Zone, -1: In Top Zone
    signals = np.zeros(len(g))
    
    current_state = 0
    for i in range(1, len(g)):
        val = g[i]
        
        if current_state == 0:
            if val > thresh: current_state = 1
            elif val < -thresh: current_state = -1
            
        elif current_state == 1: # In Bot Zone
            if val < exit_thresh:
                current_state = 0
                signals[i] = 1 # Trigger ON EXIT
                
        elif current_state == -1: # In Top Zone
            if val > -exit_thresh:
                current_state = 0
                signals[i] = -1 # Trigger ON EXIT
                
    return signals

def run_logic_peak(df):
    """Logic 3: Trigger at Local Extremum inside Zone"""
    thresh = np.tanh(SIGMA_TRIG)
    g = df['gravity'].values
    
    signals = np.zeros(len(g))
    
    # Vectorized Peak Detection
    # Peak Top: val < -thresh AND val > prev AND prev < prev2
    # This detects a "V" shape at the bottom of the graph (Top Reversal Force)
    
    # Top Reversal (Gravity is Negative)
    is_deep = g < -thresh
    is_turning_up = (g > np.roll(g, 1)) & (np.roll(g, 1) < np.roll(g, 2))
    signals[is_deep & is_turning_up] = -1
    
    # Bottom Reversal (Gravity is Positive)
    is_high = g > thresh
    is_turning_down = (g < np.roll(g, 1)) & (np.roll(g, 1) > np.roll(g, 2))
    signals[is_high & is_turning_down] = 1
    
    return signals

def run_logic_momentum(df):
    """Logic 4: Trigger when Gravity Extreme AND Price Momentum Confirms"""
    thresh = np.tanh(SIGMA_TRIG)
    g = df['gravity'].values
    mom = df['mom'].values
    
    signals = np.zeros(len(g))
    
    # Top Reversal: Gravity < -Thresh (Pulling Down) AND Momentum < 0 (Price Falling)
    signals[(g < -thresh) & (mom < 0)] = -1
    
    # Bot Reversal: Gravity > Thresh (Pulling Up) AND Momentum > 0 (Price Rising)
    signals[(g > thresh) & (mom > 0)] = 1
    
    # Edge detection
    events = np.zeros(len(g))
    events[(signals == 1) & (np.roll(signals, 1) != 1)] = 1
    events[(signals == -1) & (np.roll(signals, 1) != -1)] = -1
    return events

def run_logic_divergence(df):
    """Logic 6: Divergence (Price HH, Gravity LH)"""
    thresh = np.tanh(1.0) # Lower threshold for peak detection
    g = df['gravity'].values
    p = df['close'].values
    
    signals = np.zeros(len(g))
    
    # 1. Find Peaks/Valleys in Gravity
    # Top Peaks (Gravity > 0)
    is_peak = (g > np.roll(g, 1)) & (g > np.roll(g, -1)) & (g > thresh)
    peak_indices = np.where(is_peak)[0]
    
    # Bot Valleys (Gravity < 0)
    is_valley = (g < np.roll(g, 1)) & (g < np.roll(g, -1)) & (g < -thresh)
    valley_indices = np.where(is_valley)[0]
    
    # 2. Check Divergence for Tops
    last_peak_idx = -1
    for idx in peak_indices:
        if last_peak_idx != -1:
            # Check if this peak is "connected" (Gravity didn't cross zero?)
            # Actually, divergence usually happens across a zero-cross or a dip.
            # Classic Divergence: Price makes HH, Oscillator makes LH.
            
            # Distance check: Don't compare with ancient peaks (e.g. > 100 bars)
            if idx - last_peak_idx < 100:
                price_hh = p[idx] > p[last_peak_idx]
                gravity_lh = g[idx] < g[last_peak_idx]
                
                if price_hh and gravity_lh:
                    signals[idx] = -1 # Sell Signal
        
        last_peak_idx = idx
        
    # 3. Check Divergence for Bots
    last_valley_idx = -1
    for idx in valley_indices:
        if last_valley_idx != -1:
            if idx - last_valley_idx < 100:
                price_ll = p[idx] < p[last_valley_idx]
                gravity_hl = g[idx] > g[last_valley_idx] # Gravity is negative, so closer to 0 is "Higher" (Less Negative)
                # Wait, gravity is negative. "Higher Low" means -0.8 vs -1.5. 
                # -0.8 > -1.5. Correct.
                
                if price_ll and gravity_hl:
                    signals[idx] = 1 # Buy Signal
        
        last_valley_idx = idx
        
    return signals

# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    results = []
    
    print(f"Testing {len(files)} files with thresholds {THRESHOLDS}...")
    
    for f_path in tqdm(files):
        df = load_file_and_prep(f_path)
        if df is None: continue
        
        targets = df['y_target'].values
        price = df['close'].values
        g = df['gravity'].values
        
        for thresh_val in THRESHOLDS:
            # Manually run peak detect with dynamic threshold
            t_tanh = np.tanh(thresh_val)
            signals = np.zeros(len(g))
            
            # Peak Top
            is_deep = g < -t_tanh
            is_turning_up = (g > np.roll(g, 1)) & (np.roll(g, 1) < np.roll(g, 2))
            signals[is_deep & is_turning_up] = -1
            
            # Peak Bot
            is_high = g > t_tanh
            is_turning_down = (g < np.roll(g, 1)) & (np.roll(g, 1) > np.roll(g, 2))
            signals[is_high & is_turning_down] = 1
            
            prec, lead, count = evaluate_signals(signals, targets, price)
            results.append({
                'file': os.path.basename(f_path), 
                'threshold': thresh_val, 
                'precision': prec, 
                'lead': lead, 
                'count': count
            })

    # Aggregate
    res_df = pd.DataFrame(results)
    summary = res_df.groupby('threshold').agg({
        'precision': 'mean',
        'lead': 'mean',
        'count': 'mean'
    }).sort_values('precision', ascending=False)
    
    print("\nOptimization Results (Threshold Sensitivity):")
    print(summary)
    summary.to_csv(OUTPUT_FILE)

if __name__ == "__main__":
    main()
