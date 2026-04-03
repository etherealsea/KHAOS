
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
# Expanded search space
CALIB_WINDOWS = list(range(50, 1001, 50))
SIGMA_THRESHOLDS = [1.5, 1.65, 1.8, 1.96, 2.0, 2.2, 2.5] 
HYSTERESIS_EXITS = [0.3, 0.5, 0.7, 0.8]
DYNAMIC_MODES = ['static', 'dynamic_vol', 'dynamic_hurst']

# Grid for interpolation
EMA_GRID = sorted(list(set(
    list(range(20, 2001, 20)) + CALIB_WINDOWS
)))

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
        req_cols = ['m_weight', 'close', 'hurst_raw', 'vol_raw']
        if not all(col in df.columns for col in req_cols):
            return None
        
        # Basic cleanup
        df['m_weight'] = df['m_weight'].fillna(0)
        df['y_target'] = zigzag_labels(df['close'], window=20)
        
        # Dynamic Factors
        df['x_vol'] = normalize_custom_safe(df['vol_raw'], 300)
        df['x_hurst'] = normalize_custom_safe(df['hurst_raw'], 300)
        
        # Factors
        df['factor_vol'] = (1.0 / (1.0 + 0.6 * df['x_vol'])).fillna(1.0)
        df['factor_hurst'] = (1.0 + 0.6 * df['x_hurst']).fillna(1.0)
        
        # Pre-calculate EMAs for the Grid
        # We store them in a dictionary: window -> (mean_series, std_series)
        # This uses memory but we only hold one file at a time now.
        ema_cache = {}
        m_raw = df['m_weight']
        
        # Optimization: ewm is fast
        for w in EMA_GRID:
            # Use span=w
            mean = m_raw.ewm(span=w, adjust=False).mean()
            std = m_raw.ewm(span=w, adjust=False).std().fillna(1e-6)
            ema_cache[w] = (mean, std)
            
        df.attrs['ema_cache'] = ema_cache
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_interpolated_stats(df, target_windows):
    """
    Interpolates Mean and Std from ema_cache based on target_windows array.
    target_windows: array of float/int
    """
    cache = df.attrs['ema_cache']
    grid = np.array(EMA_GRID)
    
    # Clip windows to grid range
    targets = np.clip(target_windows, grid[0], grid[-1])
    
    # Find indices in grid
    indices = np.searchsorted(grid, targets)
    indices = np.clip(indices, 1, len(grid)-1)
    
    w_upper = grid[indices]
    w_lower = grid[indices-1]
    
    denom = (w_upper - w_lower).astype(float)
    frac = (targets - w_lower) / denom
    
    N = len(df)
    
    # Lazy matrix construction (only for this file, so memory is safe)
    if 'ema_matrix_mean' not in df.attrs:
        mat_mean = np.zeros((N, len(grid)))
        mat_std = np.zeros((N, len(grid)))
        for i, w in enumerate(grid):
            m, s = cache[w]
            mat_mean[:, i] = m.values
            mat_std[:, i] = s.values
        df.attrs['ema_matrix_mean'] = mat_mean
        df.attrs['ema_matrix_std'] = mat_std
    
    mat_mean = df.attrs['ema_matrix_mean']
    mat_std = df.attrs['ema_matrix_std']
    
    row_idx = np.arange(N)
    
    val_lower_mean = mat_mean[row_idx, indices-1]
    val_upper_mean = mat_mean[row_idx, indices]
    mean_out = val_lower_mean + frac * (val_upper_mean - val_lower_mean)
    
    val_lower_std = mat_std[row_idx, indices-1]
    val_upper_std = mat_std[row_idx, indices]
    std_out = val_lower_std + frac * (val_upper_std - val_lower_std)
    
    return mean_out, std_out

def run_simulation_vectorized(df, calib_len, thresh_val, exit_val, mode):
    m_raw = df['m_weight'].values
    N = len(m_raw)
    
    # 1. Determine Window Sizes
    if mode == 'static':
        if calib_len in df.attrs['ema_cache']:
             mean, std = df.attrs['ema_cache'][calib_len]
             mean = mean.values
             std = std.values
        else:
             windows = np.full(N, calib_len)
             mean, std = get_interpolated_stats(df, windows)
    else:
        if mode == 'dynamic_vol':
            factor = df['factor_vol'].values
        elif mode == 'dynamic_hurst':
            factor = df['factor_hurst'].values
            
        windows = calib_len * factor
        mean, std = get_interpolated_stats(df, windows)
        
    # 2. Calculate Gravity
    z = (m_raw - mean) / (std + 1e-9)
    gravity = np.tanh(z)
    
    # 3. Vectorized Latch (Set-Reset)
    # Bottom Latch
    bot_set = gravity > thresh_val
    bot_reset = gravity < exit_val
    
    # We use a float state array: 1.0 (Set), 0.0 (Reset), NaN (Hold)
    bot_state_raw = np.full(N, np.nan)
    bot_state_raw[bot_set] = 1.0
    bot_state_raw[bot_reset] = 0.0
    
    # Forward fill NaNs to propagate state
    # Pandas ffill is fast
    bot_state = pd.Series(bot_state_raw).ffill().fillna(0.0).values
    
    # Top Latch
    top_set = gravity < -thresh_val
    top_reset = gravity > -exit_val
    
    top_state_raw = np.full(N, np.nan)
    top_state_raw[top_set] = 1.0
    top_state_raw[top_reset] = 0.0
    
    top_state = pd.Series(top_state_raw).ffill().fillna(0.0).values
    
    # Combined Signal: 1 (Bot), -1 (Top)
    signals = bot_state - top_state
    
    return signals

def evaluate_signals(signals, targets):
    # Signals: 1, -1, 0
    # Targets: 1, -1, 0
    
    # Expand targets to window +/- 2 bars for tolerance
    # ZigZag is precise, but signals might lead/lag slightly
    # Using simple point-wise for now, but if results are bad, we should relax.
    
    p_bot = (signals == 1)
    t_bot = (targets == 1)
    p_top = (signals == -1)
    t_top = (targets == -1)
    
    # TP: Signal Active AND Target Present
    # This is point-wise.
    tp = np.sum(p_bot & t_bot) + np.sum(p_top & t_top)
    
    # FP: Signal Active AND NO Target Present
    # This penalizes long signal duration excessively.
    # If signal is 10 bars long and covers 1 target, we get 1 TP and 9 FPs.
    # This explains the low Precision (0.06).
    # We should count "Signal Events" vs "Target Events", not points.
    
    # Simple fix for Grid Search:
    # Use "Signal Start" only for FP counting?
    # Or, if Target is inside Signal Zone, it's a TP. If Signal Zone has NO Target, it's an FP.
    
    # Implementation of Zone-based Metrics
    # 1. Identify Signal Zones (contiguous 1s or -1s)
    # 2. Check if Target exists in Zone
    
    # Since we are inside a tight loop, we need a fast way.
    # Count of Signal Zones = (signals != signals.shift()).sum() / 2 approx.
    
    # Fast approach:
    # Treat Signal as binary mask.
    # TP = Count of Targets covered by Signal Mask.
    # FN = Count of Targets NOT covered by Signal Mask.
    # FP = Count of Signal Zones that contain NO Targets.
    
    # Calculate TP (Covered Targets)
    tp_points = np.sum(p_bot & t_bot) + np.sum(p_top & t_top)
    
    # Calculate FN (Missed Targets)
    fn_points = np.sum(t_bot & ~p_bot) + np.sum(t_top & ~p_top)
    
    # Calculate FP (False Alarms)
    # Identify rising edges of signals
    bot_starts = (signals == 1) & (np.roll(signals, 1) != 1)
    bot_starts[0] = (signals[0] == 1)
    
    top_starts = (signals == -1) & (np.roll(signals, 1) != -1)
    top_starts[0] = (signals[0] == -1)
    
    # Total Signal Events
    total_signals = np.sum(bot_starts) + np.sum(top_starts)
    
    # Ideally: FP = Total Signals - TP_Events
    # But TP above is TP_Points.
    # We assume 1 Target per Signal Zone (ideal).
    # So TP_Events = TP_Points (since Target is single point).
    
    # So FP_Events = Total_Signal_Events - TP_Points
    # But wait, if Signal covers 2 Targets (rare for reversals), we get 2 TPs.
    # If Signal covers 0 Targets, it's FP.
    
    # Adjusted FP:
    fp_events = total_signals - tp_points
    if fp_events < 0: fp_events = 0 # Should not happen if Targets are sparse
    
    return tp_points, fp_events, fn_points

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        print("No files found!")
        return
        
    print(f"Found {len(files)} files.")
    print("Starting Grid Search (Memory Safe Mode)...")
    
    # Initialize Global Results Dictionary
    # Key: (mode, calib, thresh, exit_sigma)
    # Value: {'tp': 0, 'fp': 0, 'fn': 0}
    global_stats = {}
    
    # Pre-populate keys to avoid checking during loop
    for mode in DYNAMIC_MODES:
        for calib in CALIB_WINDOWS:
            for thresh in SIGMA_THRESHOLDS:
                for exit_sigma in HYSTERESIS_EXITS:
                    if exit_sigma < thresh:
                        global_stats[(mode, calib, thresh, exit_sigma)] = {'tp': 0, 'fp': 0, 'fn': 0}

    # Iterate Files (Outer Loop)
    for f_path in tqdm(files, desc="Processing Files"):
        df = load_file_and_prep(f_path)
        if df is None:
            continue
            
        # Debug: Check data content
        if np.random.rand() < 0.1:
            print(f"File: {os.path.basename(f_path)}")
            print(f"  M_Weight Range: {df['m_weight'].min():.2f} to {df['m_weight'].max():.2f}")
            print(f"  Target Count: {np.sum(np.abs(df['y_target']))}")
            
        # Iterate Parameters (Inner Loop)
        # This is CPU intensive but Memory safe
        
        # Optimization: Pre-calculate tanh thresholds
        thresh_map = {t: np.tanh(t) for t in SIGMA_THRESHOLDS}
        exit_map = {e: np.tanh(e) for e in HYSTERESIS_EXITS}
        
        for mode in DYNAMIC_MODES:
            for calib in CALIB_WINDOWS:
                # Run Simulation Once per (Mode, Calib) if thresholds allow?
                # No, simulation depends on thresholds only for Latch, but mean/std depends on Mode/Calib.
                # So we compute Mean/Std once per (Mode, Calib).
                
                # 1. Get Mean/Std
                m_raw = df['m_weight'].values
                N = len(m_raw)
                
                if mode == 'static':
                    if calib in df.attrs['ema_cache']:
                         mean, std = df.attrs['ema_cache'][calib]
                         mean = mean.values
                         std = std.values
                    else:
                         windows = np.full(N, calib)
                         mean, std = get_interpolated_stats(df, windows)
                else:
                    if mode == 'dynamic_vol':
                        factor = df['factor_vol'].values
                    elif mode == 'dynamic_hurst':
                        factor = df['factor_hurst'].values
                    windows = calib * factor
                    mean, std = get_interpolated_stats(df, windows)
                
                # 2. Calculate Gravity (Z-Score Tanh)
                z = (m_raw - mean) / (std + 1e-9)
                gravity = np.tanh(z)
                
                # Debug Gravity
                # if np.random.rand() < 0.0001:
                #    print(f"Gravity Range: {gravity.min():.2f} to {gravity.max():.2f}")
                
                # 3. Iterate Thresholds
                for thresh in SIGMA_THRESHOLDS:
                    t_val = thresh_map[thresh]
                    
                    for exit_sigma in HYSTERESIS_EXITS:
                        if exit_sigma >= thresh:
                            continue
                        e_val = exit_map[exit_sigma]
                        
                        # Latch Logic (Fast Vectorized)
                        bot_set = gravity > t_val
                        bot_reset = gravity < e_val
                        
                        bot_state_raw = np.full(N, np.nan)
                        bot_state_raw[bot_set] = 1.0
                        bot_state_raw[bot_reset] = 0.0
                        
                        # Use numpy ufunc accumulate for ffill if pandas is slow?
                        # Pandas ffill is optimized C.
                        bot_state = pd.Series(bot_state_raw).ffill().fillna(0.0).values
                        
                        top_set = gravity < -t_val
                        top_reset = gravity > -e_val
                        
                        top_state_raw = np.full(N, np.nan)
                        top_state_raw[top_set] = 1.0
                        top_state_raw[top_reset] = 0.0
                        
                        top_state = pd.Series(top_state_raw).ffill().fillna(0.0).values
                        
                        signals = bot_state - top_state
                        
                        # Evaluate
                        tp, fp, fn = evaluate_signals(signals, df['y_target'].values)
                        
                        # Accumulate
                        key = (mode, calib, thresh, exit_sigma)
                        global_stats[key]['tp'] += tp
                        global_stats[key]['fp'] += fp
                        global_stats[key]['fn'] += fn

        # Explicit cleanup
        del df
        gc.collect()

    # Calculate Final Metrics
    print("Calculating Metrics...")
    results = []
    for key, stats in global_stats.items():
        mode, calib, thresh, exit_sigma = key
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        
        results.append({
            'mode': mode,
            'calib': calib,
            'thresh': thresh,
            'exit': exit_sigma,
            'f1': f1,
            'prec': prec,
            'rec': rec
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("afc_optimization_log_safe.csv", index=False)
    
    print("\nTop 5 Configs per Mode:")
    for mode in DYNAMIC_MODES:
        print(f"\nMode: {mode}")
        subset = df_res[df_res['mode'] == mode].sort_values('f1', ascending=False).head(5)
        print(subset[['calib', 'thresh', 'exit', 'f1', 'prec', 'rec']])

if __name__ == "__main__":
    main()
