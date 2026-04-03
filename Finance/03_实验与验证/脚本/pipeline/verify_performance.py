import os
import sys
import torch
import numpy as np
import pandas as pd
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.khaos.optimization.gpu_replica import KhaosSimulatorGPU
from src.khaos.optimization.pine_replica import add_features_to_df

# Best Parameters from Optimization
BEST_PARAMS = {
    "alpha": 0.0176,
    "calib_len": 110,
    "vol_gate": -1.10,
    "hurst_gate": 0.88,
    "w_q": [0.30729, -0.20285, -2.06313, -2.26450, 2.61335, -0.44883, -2.52776, 1.09349, -2.82586],
    "w_r": [3.00000, 1.05663, 0.73354, -1.25818, -2.37902, 0.74605, -0.33767, -0.40155, -2.46854],
    "w_m": [2.95861, -0.19115, -1.70915, 0.69234, -1.55955, 1.17570, 2.73786, 0.81978, -2.67305]
}

DATA_DIR = r"d:\Finance\Finance\data\model_research\data"
ASSETS = ["BTC_1h.csv", "NDX_Index_1h.csv", "SPX_Index_1h.csv", "TSLA_1h.csv", "Gold_ETF_1h.csv"]
CHUNK_SIZE = 1000

def run_verification():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verifying on {device}...")
    
    # 1. Prepare Single Particle (1, 31)
    # 0: alpha
    # 1: calib_len
    # 2-10: w_q
    # 11-19: w_r
    # 20-28: w_m
    # 29: vol_gate
    # 30: hurst_gate
    
    p = []
    p.append(BEST_PARAMS["alpha"])
    p.append(BEST_PARAMS["calib_len"])
    p.extend(BEST_PARAMS["w_q"])
    p.extend(BEST_PARAMS["w_r"])
    p.extend(BEST_PARAMS["w_m"])
    p.append(BEST_PARAMS["vol_gate"])
    p.append(BEST_PARAMS["hurst_gate"])
    
    population = torch.tensor([p], dtype=torch.float32, device=device)
    
    # 2. Setup Simulator
    sim = KhaosSimulatorGPU()
    
    # Load Data
    total_chunks = 0
    for f in ASSETS:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]
                # Split into chunks to match training environment
                n = len(df)
                num_windows = n // CHUNK_SIZE
                for i in range(num_windows):
                    start = i * CHUNK_SIZE
                    end = start + CHUNK_SIZE
                    segment = df.iloc[start:end].reset_index(drop=True)
                    segment = add_features_to_df(segment)
                    sim.add_dataset(segment)
                    total_chunks += 1
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
    print(f"Loaded {total_chunks} data chunks for verification.")
    
    # 3. Run Simulation (Modified to capture detailed stats)
    # Since simulate_batch returns score, we need to replicate the scoring logic here to print details.
    
    print("\n=== Detailed Performance Analysis ===")
    
    total_signals = 0
    total_wins = 0
    total_losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    # We iterate manually to capture stats
    for data in sim.static_features:
        # Re-run core logic for this chunk
        # ... (Simplified copy of gpu_replica logic for verification)
        # To avoid duplicating massive code, we can trust the score logic but we want explicit counts.
        # Let's use the simulator's internal logic if possible, or just reimplement the scoring part lightly.
        
        # We will reimplement the scoring part using the Tensors
        # Inputs
        feats = data['feats'] # (T, 9)
        T = feats.shape[0]
        
        # Weights
        w_q_t = population[:, 2:11]
        w_r_t = population[:, 11:20]
        w_m_t = population[:, 20:29]
        
        # 1. Curves
        q_raw = torch.matmul(feats, w_q_t.T)
        r_raw = torch.matmul(feats, w_r_t.T)
        m_raw = torch.matmul(feats, w_m_t.T)
        
        kan_q = torch.clamp(q_raw, min=0.01)
        kan_r = torch.clamp(r_raw, min=0.01)
        
        # 2. Gravity (Simplified Force Calc for verification - assuming fixed calib_len for speed)
        # Note: In training we used complex rolling. Here we use the exact param.
        c_len = int(BEST_PARAMS["calib_len"])
        
        # Use simple rolling mean/std for verification (CPU/GPU compatible)
        # M is (T, 1)
        m_cpu = m_raw.squeeze(1).cpu().numpy()
        m_series = pd.Series(m_cpu)
        m_mean = m_series.ewm(span=c_len).mean().values # Use EMA as approximation or rolling
        # The training used: m_mean = sum_window / eff_len (Simple Moving Average)
        m_mean = m_series.rolling(window=c_len, min_periods=1).mean().values
        m_std = m_series.rolling(window=c_len, min_periods=1).std().fillna(1.0).values
        
        m_z = (m_cpu - m_mean) / (m_std + 1e-9)
        raw_force = np.tanh(m_z)
        
        # WMA 5
        weights = np.arange(1, 6)
        raw_force_s = pd.Series(raw_force).rolling(5).apply(lambda x: np.dot(x, weights)/15, raw=True).fillna(0).values
        
        # Bias Gating
        n_bias = feats[:, 2].cpu().numpy()
        gravity_force = np.where(n_bias > 0, np.minimum(raw_force_s, 0), raw_force_s)
        gravity_force = np.where(n_bias < 0, np.maximum(gravity_force, 0), gravity_force)
        
        # 3. EKF (We need the Z-score)
        # Use Python implementation for verification (simpler than porting tensor loop here)
        log_price = data['log_price'].cpu().numpy()
        hurst = data['hurst_fast'].cpu().numpy()
        raw_vol = data['vol_kan'].cpu().numpy()
        
        alpha = BEST_PARAMS["alpha"]
        x_p = log_price[0]
        x_v = 0.0
        p00, p01, p10, p11 = 0.1, 0.0, 0.0, 0.1
        
        ekf_z = np.zeros(T)
        
        for t in range(1, T):
            h_diff = hurst[t] - 0.5
            rho = 0.5 + 0.5 / (1.0 + np.exp(-10.0 * h_diff))
            
            # Q/R
            scaler = max(1.0, 100.0 * raw_vol[t])
            # q_raw/r_raw are tensors (T, 1)
            q_val = alpha * 0.01 * kan_q[t].item() * scaler
            r_val = alpha * 0.01 * kan_r[t].item() * scaler
            
            xp_pred = x_p + x_v
            xv_pred = x_v * rho
            
            pp00 = (p00 + p01 + p10 + p11) + q_val
            pp01 = (p01 + p11) * rho
            pp10 = (p10 + p11) * rho
            pp11 = p11 * rho*rho + q_val * 0.01
            
            y = log_price[t] - xp_pred
            S = pp00 + r_val
            if S < 1e-9: S = 1e-9
            
            K0 = pp00 / S
            K1 = pp10 / S
            
            ekf_z[t] = max(min(y / np.sqrt(S), 10.0), -10.0)
            
            x_p = xp_pred + K0 * y
            x_v = xv_pred + K1 * y
            
            p00 = (1 - K0) * pp00
            p01 = (1 - K0) * pp01
            p10 = -K1 * pp00 + pp10
            p11 = -K1 * pp01 + pp11
            
        # 4. Signals
        thresh = np.tanh(1.65)
        exit_thresh = np.tanh(1.0)
        vol_gate = BEST_PARAMS["vol_gate"]
        hurst_gate = BEST_PARAMS["hurst_gate"]
        
        n_vol = data['n_vol'].cpu().numpy()
        hurst_fast = data['hurst_fast'].cpu().numpy()
        
        is_vol_valid = n_vol > vol_gate
        is_struct_valid = np.nan_to_num(hurst_fast, 0.5) < hurst_gate
        is_regime_valid = is_vol_valid & is_struct_valid
        
        latch_top = False
        latch_bot = False
        
        # Forward Returns
        close = data['close'].cpu().numpy()
        fwd_ret = (np.roll(close, -12) - close) / close
        fwd_ret[-12:] = 0
        
        for t in range(T):
            gf = gravity_force[t]
            z = ekf_z[t]
            valid = is_regime_valid[t]
            
            # Reset
            if gf > -exit_thresh: latch_top = False
            if gf < exit_thresh: latch_bot = False
            
            # Trigger
            if (gf < -thresh) and valid and (z > 0.5): latch_top = True
            if (gf > thresh) and valid and (z < -0.5): latch_bot = True
            
            # Count Signals (Only count ENTRY points to avoid flooding stats)
            # Actually, the score sums all active points.
            # But for "Win Rate", we should look at "Zones".
            # Let's count every active bar as a "Trade Decision".
            
            ret = fwd_ret[t]
            
            if latch_top:
                total_signals += 1
                if ret < -0.002: # Short Win
                    total_wins += 1
                    gross_profit += abs(ret)
                else:
                    total_losses += 1
                    gross_loss += abs(ret)
                    
            if latch_bot:
                total_signals += 1
                if ret > 0.002: # Long Win
                    total_wins += 1
                    gross_profit += abs(ret)
                else:
                    total_losses += 1
                    gross_loss += abs(ret)

    print(f"Total Validated Signal Points: {total_signals}")
    if total_signals > 0:
        win_rate = (total_wins / total_signals) * 100
        pf = gross_profit / (gross_loss + 1e-9)
        print(f"Win Rate (Success > 0.2%): {win_rate:.2f}%")
        print(f"Profit Factor: {pf:.2f}")
        print(f"Avg Return per Signal: {(gross_profit - gross_loss) / total_signals * 100:.4f}%")
    else:
        print("No signals generated.")

if __name__ == "__main__":
    run_verification()
