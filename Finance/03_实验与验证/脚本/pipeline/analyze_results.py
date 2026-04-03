
import pandas as pd

def analyze():
    df = None
    try:
        df = pd.read_csv("afc_optimization_log_safe.csv")
        print("Loaded afc_optimization_log_safe.csv")
    except:
        pass
        
    if df is None:
        try:
            df = pd.read_csv("afc_optimization_log.csv")
            print("Loaded afc_optimization_log.csv")
        except Exception as e:
            print(f"Error reading CSVs: {e}")
            
    if df is None:
        print("No data found.")
        return

    print("\n=== Best Config Per Mode (Sorted by F1) ===")
    modes = ['static', 'dynamic_vol', 'dynamic_hurst']
    
    for mode in modes:
        subset = df[df['mode'] == mode]
        if subset.empty:
            print(f"\nMode: {mode} - NO DATA FOUND")
            continue
            
        best = subset.sort_values('f1', ascending=False).head(5)
        print(f"\nMode: {mode}")
        print(best[['calib', 'thresh', 'exit', 'f1', 'prec', 'rec']])

if __name__ == "__main__":
    analyze()
