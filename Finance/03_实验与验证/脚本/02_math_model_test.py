import pandas as pd
import numpy as np
import os
from khaos_model import KHAOS_Model

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def main():
    print("Initializing KHAOS Model Test...")
    
    # Try to load a file
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    
    if not files:
        print("No data found in model_research/data. Please run fetch_data_direct.py first.")
        # Create a small dummy dataset for syntax validation ONLY
        print("Creating dummy data for syntax validation...")
        dates = pd.date_range(start='2023-01-01', periods=200)
        prices = np.cumsum(np.random.randn(200)) + 100
        df = pd.DataFrame({'close': prices}, index=dates)
        model = KHAOS_Model()
        results = model.process_data(df)
        print("\nResults Head (Dummy):")
        print(results.head())
        return

    print(f"Found {len(files)} files. Processing representative subset...")
    
    # Select a subset of files to test different timeframes and asset classes
    target_files = [
        'AAPL_1d.csv',
        'EURUSD_1h.csv',
        'BTC_15m.csv',
        'SPX_Index_4h.csv',
        'VIX_ETF_5m.csv'
    ]
    
    # Filter files that exist
    files_to_process = [f for f in target_files if f in files]
    
    if not files_to_process:
        print("Warning: Target files not found. Processing first 5 files found.")
        files_to_process = files[:5]

    model = KHAOS_Model()
    
    for filename in files_to_process:
        print(f"\nProcessing {filename}...")
        try:
            file_path = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(file_path)
            
            # Check for date column and set index
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], format='mixed')
                except:
                    df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
                except:
                    df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
            results = model.process_data(df)
            
            print("  > Output Shape:", results.shape)
            print("  > Columns:", results.columns.tolist())
            print("  > Sample Values (Last 3 rows):")
            print(results.tail(3)[['Price', 'Oscillation', 'Hurst']])
            
            # Validation Checks
            osc_min = results['Oscillation'].min()
            osc_max = results['Oscillation'].max()
            hurst_mean = results['Hurst'].mean()
            
            print(f"  > Validation: Oscillation Range [{osc_min:.2f}, {osc_max:.2f}]")
            print(f"  > Validation: Hurst Mean {hurst_mean:.2f}")
            
            if -1.1 <= osc_min and osc_max <= 1.1:
                print("  > [PASS] Oscillation is bounded.")
            else:
                print("  > [FAIL] Oscillation out of bounds!")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
