import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.khaos.optimization.gpu_replica import KhaosSimulatorGPU
from src.khaos.optimization.pine_replica import add_features_to_df

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = r"d:\Finance\Finance\data\model_research\data"
ASSETS = [
    "BTC_1h.csv", "NDX_Index_1h.csv", "SPX_Index_1h.csv", 
    "TSLA_1h.csv", "Gold_ETF_1h.csv"
]
CHUNK_SIZE = 1000
STATUS_FILE = "optimization_status.json"

# ==============================================================================
# Data Loading
# ==============================================================================
def extract_regimes(df, chunk_size=500):
    n = len(df)
    if n < chunk_size * 4: return [df]
    
    num_windows = n // chunk_size
    window_metrics = []
    
    for i in range(num_windows):
        start = i * chunk_size
        end = start + chunk_size
        segment = df.iloc[start:end]
        close = segment['close'].values
        ret = (close[-1] - close[0]) / close[0]
        log_ret = np.diff(np.log(close))
        vol = np.std(log_ret)
        window_metrics.append({'index': i, 'return': ret, 'vol': vol, 'segment': segment})
        
    sorted_ret = sorted(window_metrics, key=lambda x: x['return'])
    sorted_vol = sorted(window_metrics, key=lambda x: x['vol'])
    
    return [
        sorted_ret[0]['segment'],   # Bear
        sorted_ret[-1]['segment'],  # Bull
        sorted_vol[-1]['segment'],  # Chaos
        sorted_vol[0]['segment']    # Chop
    ]

def load_data_gpu(sim):
    print("Loading data to GPU...")
    count = 0
    for f in ASSETS:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]
                if 'close' in df.columns:
                    regimes = extract_regimes(df, CHUNK_SIZE)
                    for r in regimes:
                        r = r.reset_index(drop=True)
                        r = add_features_to_df(r) # Pre-calc features on CPU
                        sim.add_dataset(r)        # Move to GPU
                        count += 1
            except Exception as e:
                print(f"Error loading {f}: {e}")
    print(f"Loaded {count} regime chunks to GPU.")

# ==============================================================================
# Genetic Algorithm
# ==============================================================================
def run_evolution():
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (will be slow).")
        
    sim = KhaosSimulatorGPU()
    load_data_gpu(sim)
    
    # GA Parameters
    # GPU Optimization: Increased Pop Size for better exploration
    if torch.cuda.is_available():
        POP_SIZE = 500
    else:
        POP_SIZE = 100  # Reduced for CPU safety

    GENERATIONS = 50
    
    # CPU Thread Management
    if not torch.cuda.is_available():
        # Limit CPU threads to prevent system freeze
        # Assuming typical 8-16 core machine, using 4-6 threads is safe.
        torch.set_num_threads(6)
        print("Running on CPU with optimized vectorization (Threads=6).")
    ELITISM = 0.1
    MUTATION_RATE = 0.05
    MUTATION_SCALE = 0.2
    
    # Parameter Bounds (31 params)
    # 0: alpha (0.1, 1.0) - Forced minimum alpha to prevent silence
    # 1: calib_len (100, 300)
    # 2-28: Weights (-3, 3)
    # 29: vol_gate (-1.5, 0.5)
    # 30: hurst_gate (0.4, 0.9)
    
    bounds_min = [0.1, 100.0] + [-3.0]*27 + [-1.5, 0.4]
    bounds_max = [1.0, 300.0] + [3.0]*27 + [0.5, 0.9]
    
    bounds_min = torch.tensor(bounds_min, device=sim.device)
    bounds_max = torch.tensor(bounds_max, device=sim.device)
    
    # Initialize Population
    population = torch.rand(POP_SIZE, 31, device=sim.device)
    population = population * (bounds_max - bounds_min) + bounds_min
    
    # FIX: Force Bias weights (indices 10, 19, 28 within the 2-28 block) to 0.0 initially
    # Indices in population:
    # Alpha=0, Calib=1
    # Q Weights: 2-10 (Bias is 10)
    # R Weights: 11-19 (Bias is 19)
    # M Weights: 20-28 (Bias is 28)
    
    population[:, 10] = 0.0 # Q Bias
    population[:, 19] = 0.0 # R Bias
    population[:, 28] = 0.0 # M Bias
    
    best_score = -float('inf')
    best_params = None
    
    print(f"Starting Evolution: Pop={POP_SIZE}, Gens={GENERATIONS}", flush=True)
    
    try:
        for gen in range(GENERATIONS):
            print(f"Debug: Starting Gen {gen}", flush=True)
            start_time = time.time()
            
            # 1. Evaluate
            scores = sim.simulate_batch(population)
            print("Debug: Batch simulated.", flush=True)
            
            # 2. Stats
            gen_best_score = torch.max(scores).item()
            gen_avg_score = torch.mean(scores).item()
            
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_idx = torch.argmax(scores)
                best_params = population[best_idx].clone()
                
            # 3. Selection (Top k)
            # Sort indices
            sorted_indices = torch.argsort(scores, descending=True)
            
            num_elites = int(POP_SIZE * ELITISM)
            elites = population[sorted_indices[:num_elites]]
            
            # Parents for crossover (Top 50%)
            num_parents = int(POP_SIZE * 0.5)
            parents = population[sorted_indices[:num_parents]]
            
            # 4. Offspring Generation
            num_offspring = POP_SIZE - num_elites
            
            # Random parent pairs
            p1_idx = torch.randint(0, num_parents, (num_offspring,), device=sim.device)
            p2_idx = torch.randint(0, num_parents, (num_offspring,), device=sim.device)
            
            parent1 = parents[p1_idx]
            parent2 = parents[p2_idx]
            
            # Crossover (Uniform)
            mask = torch.rand(num_offspring, 31, device=sim.device) < 0.5
            offspring = torch.where(mask, parent1, parent2)
            
            # Mutation
            mutation_mask = torch.rand(num_offspring, 31, device=sim.device) < MUTATION_RATE
            noise = torch.randn(num_offspring, 31, device=sim.device) * MUTATION_SCALE
            
            # Scale noise for specific params?
            # alpha/weights are fine. calib_len needs larger noise?
            # Let's keep it simple.
            
            offspring = offspring + torch.where(mutation_mask, noise, torch.zeros_like(noise))
            
            # Clamp bounds
            offspring = torch.max(torch.min(offspring, bounds_max), bounds_min)
            
            # Next Gen
            population = torch.cat([elites, offspring], dim=0)
            
            dt = time.time() - start_time
            print(f"Gen {gen+1}/{GENERATIONS} | Best: {gen_best_score:.4f} | Avg: {gen_avg_score:.4f} | Time: {dt:.2f}s")
            
            # Status Update
            status = {
                "status": "running",
                "generation": gen + 1,
                "best_score": best_score,
                "avg_score": gen_avg_score,
                "best_params": best_params.tolist()
            }
            try:
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f)
            except: pass
            
    except Exception as e:
        print(f"!!! CRITICAL ERROR IN EVOLUTION LOOP: {e}")
        import traceback
        traceback.print_exc()

    print("\nOptimization Complete!")
    print(f"Best Score: {best_score}")
    
    # Save Results
    bp = best_params.cpu().numpy()
    with open("pikan_opt_results_gpu.txt", "w") as f:
        f.write(f"Best Score: {best_score}\n\n")
        f.write("=== Optimal Parameters ===\n")
        f.write(f"alpha: {bp[0]:.4f}\n")
        f.write(f"calib_len: {int(bp[1])}\n")
        f.write(f"vol_gate: {bp[29]:.2f}\n")
        f.write(f"hurst_gate: {bp[30]:.2f}\n\n")
        
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

        print_w("Q", bp[2:11])
        print_w("R", bp[11:20])
        print_w("M", bp[20:29])

if __name__ == "__main__":
    run_evolution()
