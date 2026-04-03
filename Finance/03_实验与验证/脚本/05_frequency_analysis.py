import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from khaos_model import KHAOS_Model

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def main():
    print("========================================================")
    print("   KHAOS Distribution & Frequency Analysis              ")
    print("========================================================")
    
    files = ['AAPL_1d.csv', 'SPX_Index_4h.csv', 'BTC_15m.csv']
    model = KHAOS_Model()
    
    stats_list = []
    
    for filename in files:
        print(f"\nAnalyzing {filename}...")
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print("File not found, skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Date parsing
        if 'date' in df.columns:
            try: df['date'] = pd.to_datetime(df['date'], format='mixed')
            except: df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'Date' in df.columns:
            try: df['Date'] = pd.to_datetime(df['Date'], format='mixed')
            except: df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # Process
        result = model.process_data(df)
        khaos = result['KHAOS'].dropna()
        
        # Distribution Stats
        mean_val = khaos.mean()
        std_val = khaos.std()
        min_val = khaos.min()
        max_val = khaos.max()
        
        # Threshold Crossings
        count_over_08 = (khaos > 0.8).sum()
        count_under_08 = (khaos < -0.8).sum()
        total_bars = len(khaos)
        
        freq_ratio = (count_over_08 + count_under_08) / total_bars
        
        print(f"  > Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        print(f"  > Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  > > 0.8: {count_over_08} bars | < -0.8: {count_under_08} bars")
        print(f"  > Trigger Freq (Bars): {freq_ratio:.2%}")
        
        stats_list.append({
            'Asset': filename,
            'Std_Dev': std_val,
            'Trigger_Freq': freq_ratio
        })

    print("\n--- Diagnosis ---")
    for stat in stats_list:
        print(f"{stat['Asset']}: StdDev = {stat['Std_Dev']:.4f}, Freq = {stat['Trigger_Freq']:.2%}")
        
    print("\nObservation:")
    print("If StdDev is significantly lower than 0.4-0.5, the distribution is too narrow for fixed 0.8 threshold.")
    print("This confirms why AAPL/SPX have low signal frequency.")

if __name__ == "__main__":
    main()
