import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.khaos.optimization.pine_replica import run_khaos_simulation

DATA_DIR = r"d:\Finance\Finance\data\model_research\data"
ASSETS = ["BTC_1h.csv"] # Test with just one

def load_data():
    datasets = []
    print("Loading test data...")
    path = os.path.join(DATA_DIR, ASSETS[0])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        # Take small chunk
        df = df.iloc[-1000:].reset_index(drop=True)
        datasets.append(("TEST_BTC", df))
        print(f"Loaded {len(df)} bars.")
    return datasets

def objective_function(params, datasets):
    try:
        # Unpack Params
        p_alpha = params[0]
        p_calib_len = int(params[1])
        p_vol_gate = params[29]
        p_hurst_gate = params[30]
        
        weights = {
            'q': params[2:11],
            'r': params[11:20],
            'm': params[20:29]
        }
        
        sim_params = {
            'alpha': p_alpha,
            'hurst_fast': 30,
            'pe_len': 30,
            'calib_len': p_calib_len,
            'sigma_trig': 1.65,
            'sigma_exit': 1.0,
            'vol_gate': p_vol_gate,
            'hurst_gate': p_hurst_gate
        }
        
        total_score = 0
        
        for fname, df in datasets:
            is_top, is_bot, gf, ekf, n_vol = run_khaos_simulation(df, sim_params, weights)
            
            # Simple score
            signals = np.sum(is_top) + np.sum(is_bot)
            total_score += signals
            
        return -total_score # Maximize signals for test
        
    except Exception as e:
        print(f"!!! Error in Objective Function: {e}")
        traceback.print_exc()
        return 0.0

def run_test():
    datasets = load_data()
    
    # 31 Params
    bounds = [(-3.0, 3.0)] * 31
    
    print("Starting Differential Evolution Test (1 iter)...")
    result = differential_evolution(
        objective_function,
        bounds,
        args=(datasets,),
        maxiter=1,
        popsize=2, # Tiny population
        disp=True,
        workers=1
    )
    
    print("\nTest Complete.")
    print("Result X:", list(result.x))
    
    with open("debug_result.txt", "w") as f:
        f.write(str(list(result.x)))
    print("Wrote debug_result.txt")

if __name__ == "__main__":
    run_test()
