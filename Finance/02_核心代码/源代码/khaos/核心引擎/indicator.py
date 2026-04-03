
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from khaos.核心引擎.physics import DifferentiableEKF, calculate_hurst_proxy, permutation_entropy

class KHAOSIndicator:
    def __init__(self, device='cpu'):
        self.device = device
        self.ekf = DifferentiableEKF().to(device)
        
    def process(self, df):
        """
        Process a DataFrame containing 'close', 'high', 'low' columns.
        Returns the DataFrame with added KHAOS indicator columns.
        """
        # Ensure lowercase
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
            
        # Prepare Tensors
        close = torch.tensor(df['close'].values, dtype=torch.float32).to(self.device).unsqueeze(0) # [1, T]
        
        # 1. Log Price (Scale Invariant)
        log_close = torch.log(close + 1e-8)
        
        # 2. Volatility (for EKF)
        # Simple returns diff
        ret = torch.diff(log_close, dim=1, prepend=log_close[:, :1])
        vol = ret.abs()
        
        # 3. Hurst Exponent (Instability Component 1)
        # Using the proxy function from physics.py
        # Window size 16 is default in physics.py
        hurst = calculate_hurst_proxy(log_close, window=32) # Use slightly larger window for stability
        
        # 4. EKF (Anomaly Component)
        # Returns states: [p, v]
        ekf_states = self.ekf(log_close, hurst, vol)
        ekf_p = ekf_states[:, :, 0]
        
        # Residual = |Price - Estimated_Trend|
        # This measures "Deviation from Physics Consensus"
        residual = (log_close - ekf_p).abs()
        
        # 5. Permutation Entropy (Instability Component 2)
        # Measures "Randomness/Complexity"
        # Using higher order (5) and larger window (128) for better resolution
        pe = permutation_entropy(close, order=5, delay=1, window_size=128)
        
        # --- Post-Processing to Indicators ---
        
        # Convert to Numpy
        hurst_np = hurst.cpu().numpy().flatten()
        pe_np = pe.cpu().numpy().flatten()
        res_np = residual.cpu().numpy().flatten()
        
        # A. KHAOS Instability (Adaptive 0-100)
        # Raw PE is usually 0.8-1.0 for markets.
        # We use Adaptive Min-Max Scaling (Window 500) to highlight relative instability.
        pe_series = pd.Series(pe_np)
        pe_min = pe_series.rolling(window=500, min_periods=100).min()
        pe_max = pe_series.rolling(window=500, min_periods=100).max()
        
        instability_norm = (pe_series - pe_min) / (pe_max - pe_min + 1e-8)
        instability = instability_norm.fillna(0.5).values * 100
        
        # B. KHAOS Anomaly (Z-Score of Residuals)
        # Measures sudden shocks.
        res_series = pd.Series(res_np)
        # Rolling Z-Score over long window (e.g. 100)
        res_mean = res_series.rolling(window=100).mean()
        res_std = res_series.rolling(window=100).std()
        anomaly = (res_series - res_mean) / (res_std + 1e-8)
        anomaly = anomaly.fillna(0).values
        
        # C. KHAOS Phase (Regime)
        # 0: Stable Trend (Green) -> Instability < 30
        # 1: Chaos/Transition (Red) -> Instability > 80
        # 2: Noise/Transition (Grey) -> Between
        
        df['khaos_instability'] = instability
        df['khaos_anomaly'] = anomaly
        df['khaos_hurst'] = hurst_np
        
        # Derived Signals
        # 1. Trend Exhaustion: Instability crosses above 80
        df['signal_exhaustion'] = (df['khaos_instability'] > 80).astype(int)
        
        # 2. Volatility Alert: Anomaly > 3.0 (3 Sigma Event)
        df['signal_vol_alert'] = (df['khaos_anomaly'] > 3.0).astype(int)
        
        return df

if __name__ == "__main__":
    # Test run
    path = r'd:\Finance\Finance\data\model_research\data_processed\Index\SPXUSD_1h.csv'
    if os.path.exists(path):
        print(f"Testing on {path}...")
        df = pd.read_csv(path)
        indicator = KHAOSIndicator()
        df_res = indicator.process(df)
        print(df_res[['close', 'khaos_instability', 'khaos_anomaly']].tail(10))
    else:
        print("Test file not found.")
