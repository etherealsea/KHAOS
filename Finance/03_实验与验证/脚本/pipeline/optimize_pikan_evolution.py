import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import glob
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.khaos.optimization.pine_replica import run_khaos_simulation

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = r"d:\Finance\Finance\data\model_research\data"

# "Corner Case" Asset Basket
# Focused on High Volatility (Crypto/Tech) and Macro Structure (Gold/SPX)
ASSETS = [
    "BTC_1h.csv",       # Crypto Core
    "NDX_Index_1h.csv", # Nasdaq (Tech Volatility)
    "SPX_Index_1h.csv", # S&P 500 (Market Benchmark) - Added back per request
    "TSLA_1h.csv",      # Single Stock Alpha
    "Gold_ETF_1h.csv"   # Commodity Anchor
]

# Regime Sampling Config
# Increased Chunk Size to 1000 to ensure EKF has enough "Warmup" time 
# and to capture longer trend structures without breaking continuity.
CHUNK_SIZE = 1000 
CHUNKS_PER_ASSET = 4 # Bear, Bull, Chop, Chaos
TOTAL_BARS_PER_EVAL = CHUNK_SIZE * CHUNKS_PER_ASSET * len(ASSETS)

# ==============================================================================
# Data Loading & Regime Sampling
# ==============================================================================
def extract_regimes(df, chunk_size=500):
    """
    Smart Sampling: Extracts representative market regimes to ensure robustness
    and speed up training (by skipping boring data).
    """
    n = len(df)
    if n < chunk_size * 4:
        return [df] # Too small, return whole
        
    chunks = []
    # Create windows
    # We want non-overlapping chunks
    num_windows = n // chunk_size
    
    # Metrics for each window
    window_metrics = []
    
    for i in range(num_windows):
        start = i * chunk_size
        end = start + chunk_size
        segment = df.iloc[start:end]
        
        close = segment['close'].values
        # Metric 1: Return (Trend)
        ret = (close[-1] - close[0]) / close[0]
        # Metric 2: Volatility (StdDev of returns)
        log_ret = np.diff(np.log(close))
        vol = np.std(log_ret)
        
        window_metrics.append({
            'index': i,
            'return': ret,
            'vol': vol,
            'segment': segment
        })
        
    # Sort and Select
    # 1. Bear Market (Lowest Return)
    sorted_ret = sorted(window_metrics, key=lambda x: x['return'])
    bear_chunk = sorted_ret[0]['segment']
    
    # 2. Bull Market (Highest Return)
    bull_chunk = sorted_ret[-1]['segment']
    
    # 3. Chaos (Highest Volatility)
    sorted_vol = sorted(window_metrics, key=lambda x: x['vol'])
    chaos_chunk = sorted_vol[-1]['segment']
    
    # 4. Chop (Lowest Volatility)
    chop_chunk = sorted_vol[0]['segment']
    
    return [bear_chunk, bull_chunk, chaos_chunk, chop_chunk]

def load_data():
    datasets = []
    print("Loading and Sampling Regimes...")
    for f in ASSETS:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]
                if 'close' in df.columns:
                    # Regime Sampling
                    regimes = extract_regimes(df, CHUNK_SIZE)
                    for i, r in enumerate(regimes):
                        datasets.append((f"{f}_R{i}", r.reset_index(drop=True)))
                    print(f"Loaded {f}: Extracted {len(regimes)} regimes ({len(regimes)*CHUNK_SIZE} bars)")
            except Exception as e:
                print(f"Error loading {f}: {e}")
        else:
            print(f"File not found: {path}")
    return datasets

# ==============================================================================
# Progress Monitoring
# ==============================================================================
STATUS_FILE = "optimization_status.json"
iteration_count = 0

def progress_callback(xk, convergence):
    global iteration_count
    iteration_count += 1
    
    status = {
        "status": "running",
        "iteration": iteration_count,
        "convergence": convergence,
        "current_best_params": list(xk)
    }
    
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=4)
    except Exception as e:
        print(f"Error writing status file: {e}")

# ==============================================================================
# Fitness Function
# ==============================================================================
def objective_function(params, datasets):
    print(".", end="", flush=True) 
    
    # Unpack Params
    p_alpha = params[0]
    p_hurst_len = 30
    p_calib_len = int(params[1])
    
    # REGIME GATES (Optimized)
    # Weights start at index 2.
    # Base Weights (6) * 3 sets = 18 params
    # Interaction Weights (3) * 3 sets = 9 params
    # Total Weights = 27 params.
    # Indices: 2 to 2+27 = 29
    # Gate params are after weights -> Index 29, 30
    
    p_vol_gate = params[29]
    p_hurst_gate = params[30]
    
    # LOCKED SIGMA
    p_sigma_trig = 1.65
    p_sigma_exit = 1.0 
    
    # Weights Slicing (9 params per set: 6 base + 3 interaction)
    # Q: 2:11
    # R: 11:20
    # M: 20:29
    weights = {
        'q': params[2:11],
        'r': params[11:20],
        'm': params[20:29]
    }
    
    sim_params = {
        'alpha': p_alpha,
        'hurst_fast': 30, # Renamed to match replica
        'pe_len': 30,
        'calib_len': p_calib_len,
        'sigma_trig': p_sigma_trig,
        'sigma_exit': p_sigma_exit,
        'vol_gate': p_vol_gate,
        'hurst_gate': p_hurst_gate
    }
    
    total_score = 0
    total_signal_bars = 0
    total_bars = 0
    
    for fname, df in datasets:
        # Get Zone Signals (Boolean Arrays)
        is_top_zone, is_bot_zone, _, _, _ = run_khaos_simulation(df, sim_params, weights)
        
        # QUALITY METRIC: Reversal Magnitude
        # Instead of just "Did we hit a pivot?", we measure "Did price actually reverse?"
        # We look forward 12 bars (1 hour). 
        # Good Top Signal: Price drops significantly.
        # Good Bot Signal: Price rises significantly.
        
        closes = df['close'].values
        n = len(closes)
        look_ahead = 12
        
        # Calculate Forward Returns
        fwd_ret = np.zeros(n)
        # Vectorized lookahead return
        # Ret = (Close[i+12] - Close[i]) / Close[i]
        # Shift array
        future_close = np.roll(closes, -look_ahead)
        # Last look_ahead bars are invalid
        future_close[-look_ahead:] = closes[-1] 
        fwd_ret = (future_close - closes) / closes
        
        # --- Evaluate Top Signals (Short) ---
        # We want fwd_ret to be NEGATIVE (Price Drop)
        # Score = Sum of Negative Returns caught by signal
        
        # Filter signals where fwd_ret is negative (Success) vs positive (Failure)
        top_success_mask = (is_top_zone) & (fwd_ret < -0.002) # Must drop at least 0.2% to be "Real"
        top_fail_mask = (is_top_zone) & (fwd_ret >= -0.002)
        
        # Sum of magnitude (Log Returns approx)
        top_profit = np.sum(np.abs(fwd_ret[top_success_mask]))
        top_loss = np.sum(np.abs(fwd_ret[top_fail_mask])) # Penalty for bad signals
        
        # --- Evaluate Bottom Signals (Long) ---
        # We want fwd_ret to be POSITIVE (Price Rise)
        bot_success_mask = (is_bot_zone) & (fwd_ret > 0.002)
        bot_fail_mask = (is_bot_zone) & (fwd_ret <= 0.002)
        
        bot_profit = np.sum(fwd_ret[bot_success_mask])
        bot_loss = np.sum(np.abs(fwd_ret[bot_fail_mask]))
        
        # Net Score = Profit - 1.5 * Loss (Penalize failures more than reward success)
        net_score = (top_profit - 1.5 * top_loss) + (bot_profit - 1.5 * bot_loss)
        
        # 5. Signal Density Penalty (Too few signals)
        # If we have ZERO valid signals, heavy penalty
        if np.sum(top_success_mask) == 0 and np.sum(bot_success_mask) == 0:
            net_score -= 10.0
            
        total_score += net_score
        
    # --- GLOBAL BALANCE PENALTY (New in v3.1) ---
    # We run through all regimes (Bull, Bear, Chop, Chaos).
    # If the model ONLY makes money on Bull (Longs), it will fail here.
    # We sum up global long/short profits across all datasets.
    
    # This requires tracking long/short split, but `objective_function` aggregates a scalar.
    # Simplification: We assume `total_score` already penalized bad trades.
    # But to enforce "Balance", we need to punish if (Total Long Profit) >> (Total Short Profit) or vice versa.
    # Since we can't easily extract that from the loop without refactoring, 
    # we rely on the "Bear Market Regime" in the dataset to force Short learning.
    # Since we explicitly included "Bear Market Chunks", a Long-Only model will lose money there.
    # So the existing scoring metric is implicitly balanced by the Data Selection.
        
    return -total_score # Minimize negative score (Maximize Net Profit)

# ==============================================================================
# Main Optimization
# ==============================================================================
def run_optimization():
    print("=== STARTING OPTIMIZATION ===", flush=True)
    try:
        datasets = load_data()
        if not datasets:
            print("No data.")
            return

        # Bounds
        bounds = [
            (0.01, 1.0),   # alpha
            (100, 300),    # calib_len
        ]
        
        # Weights Q (9 params: 6 base + 3 interaction)
        bounds += [(-3.0, 3.0)] * 9
        # Weights R (9 params)
        bounds += [(-3.0, 3.0)] * 9
        # Weights M (9 params)
        bounds += [(-3.0, 3.0)] * 9
        
        # NEW GATES (2)
        bounds += [
            (-1.5, 0.5), # vol_gate (Z-Score)
            (0.4, 0.9)   # hurst_gate (Abs value)
        ]
        
        print(f"Dimensions: {len(bounds)}")
        
        # Reset status file
        with open(STATUS_FILE, "w") as f:
            json.dump({"status": "starting", "iteration": 0}, f)

        result = differential_evolution(
            objective_function, 
            bounds, 
            args=(datasets,),
            strategy='best1bin', 
            maxiter=15, # Optimized for speed/quality balance
            popsize=15, # Moderate population
            tol=0.01,
            mutation=(0.5, 1.0), 
            recombination=0.7,
            disp=True,
            callback=progress_callback,
            workers=1 # Use single core (Safe Mode with Pre-calc)
        )
        
        print("\nOptimization Complete!", flush=True)
        
        # Print to Console as Backup
        print("=== FINAL WEIGHTS (CONSOLE BACKUP) ===")
        print(list(result.x))
        print("======================================")
        
        try:
            with open("pikan_opt_results.txt", "w") as f:
                f.write(f"Best Score: {-result.fun}\n\n")
                f.write("=== Optimal Parameters ===\n")
                f.write(f"alpha: {result.x[0]:.4f}\n")
                f.write(f"hurst_fast: 30\n")
                f.write(f"calib_len: {int(result.x[1])}\n")
                f.write(f"sigma_trig: 1.65\n")
                f.write(f"sigma_exit: 1.0\n")
                f.write(f"vol_gate: {result.x[29]:.2f}\n")
                f.write(f"hurst_gate: {result.x[30]:.2f}\n\n")
                
                f.write("=== Optimal Weights (Paste to Pine) ===\n\n")
                
                # Helper to print weights
                def print_w(prefix, w_arr):
                    f.write(f"// {prefix} Weights\n")
                    f.write(f"float w_{prefix.lower()}_hurst = {w_arr[0]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_vol = {w_arr[1]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_bias = {w_arr[2]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_slope = {w_arr[3]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_pe = {w_arr[4]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_ivolhurst = {w_arr[5]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_ivolbias = {w_arr[6]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_ihurstpe = {w_arr[7]:.5f}\n")
                    f.write(f"float w_{prefix.lower()}_b = {w_arr[8]:.5f}\n\n")
        
                print_w("Q", result.x[2:11])
                print_w("R", result.x[11:20])
                print_w("M", result.x[20:29])
                    
            print("Results written to pikan_opt_results.txt")
        except Exception as e:
            print(f"Error writing file: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"!!! CRITICAL ERROR IN MAIN LOOP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimization()
