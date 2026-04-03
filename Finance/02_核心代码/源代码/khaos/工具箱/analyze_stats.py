
import pandas as pd
import numpy as np
import glob
import os

def analyze_vol_stats(data_dir):
    files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    
    stats = []
    
    for f in files:
        if '1h' not in f and '1d' not in f: continue
        
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]
            
            if 'close' not in df.columns: continue
            
            # Log Returns
            close = df['close'].values
            log_ret = np.diff(np.log(close + 1e-8))
            
            # Future Vol (Horizon 4)
            s_log_ret = pd.Series(log_ret)
            future_vol = s_log_ret.rolling(window=4).std().shift(-4).dropna().values
            
            # Log Vol
            log_future_vol = np.log(future_vol + 1e-8)
            
            # Stats
            mean_val = np.mean(log_future_vol)
            std_val = np.std(log_future_vol)
            
            stats.append({
                'Asset': os.path.basename(f),
                'Mean': mean_val,
                'Std': std_val,
                'Min': np.min(log_future_vol),
                'Max': np.max(log_future_vol)
            })
        except Exception as e:
            print(f"Error {f}: {e}")
            
    df_stats = pd.DataFrame(stats)
    print(df_stats.sort_values('Asset'))
    print("\nGlobal Stats:")
    print(f"Mean of Means: {df_stats['Mean'].mean()}")
    print(f"Mean of Stds: {df_stats['Std'].mean()}")

if __name__ == "__main__":
    analyze_vol_stats(r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed')
