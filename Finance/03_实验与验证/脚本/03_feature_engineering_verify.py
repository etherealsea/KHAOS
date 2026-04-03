import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import spearmanr
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not found. ADF test will be skipped.")

from khaos_model import KHAOS_Model

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def calculate_ic(feature_series, return_series):
    """
    Calculate Information Coefficient (Spearman Rank Correlation)
    """
    # Align data
    valid_idx = feature_series.notna() & return_series.notna()
    if valid_idx.sum() < 10:
        return 0.0, 1.0 # Not enough data
    
    corr, p_value = spearmanr(feature_series[valid_idx], return_series[valid_idx])
    return corr, p_value

def check_stationarity(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    Null Hypothesis: Series is Non-Stationary (has unit root).
    p-value < 0.05 => Reject Null => Stationary.
    """
    if not HAS_STATSMODELS:
        return "N/A", 1.0
        
    clean_series = series.dropna()
    if len(clean_series) < 30:
        return "Too Short", 1.0
        
    try:
        result = adfuller(clean_series)
        p_value = result[1]
        is_stationary = "YES" if p_value < 0.05 else "NO"
        return is_stationary, p_value
    except Exception as e:
        return "Error", 1.0

def main():
    print("==================================================")
    print("   KHAOS Feature Engineering Verification Tool    ")
    print("==================================================")
    print("Methodology: Statistical Validation (Quantitative)")
    print("1. Stationarity (ADF Test): Required for ML inputs.")
    print("2. Predictive Power (IC): Rank Correlation with Future Returns.")
    print("--------------------------------------------------\n")

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    
    # Select representative subset
    target_files = [
        'AAPL_1d.csv',
        'EURUSD_1h.csv',
        'BTC_15m.csv',
        'SPX_Index_4h.csv',
        'VIX_ETF_5m.csv'
    ]
    
    files_to_process = [f for f in target_files if f in files]
    if not files_to_process:
        print("No target files found. Please run data fetcher first.")
        return

    model = KHAOS_Model()
    
    summary_report = []

    for filename in files_to_process:
        print(f"Analyzing {filename}...")
        file_path = os.path.join(DATA_DIR, filename)
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
            
        # 1. Generate Features
        features = model.process_data(df)
        
        # 2. Create Targets (Future Returns)
        # Shift(-1) means row t contains return of t+1
        features['Ret_1'] = features['Price'].pct_change().shift(-1)
        features['Ret_5'] = features['Price'].pct_change(5).shift(-5)
        
        # 3. Evaluate Each Feature
        eval_features = ['Oscillation', 'Hurst', 'Attention', 'Residuals']
        
        for feat_name in eval_features:
            series = features[feat_name]
            
            # Stationarity
            stat_status, stat_p = check_stationarity(series)
            
            # IC (Predictive Power) vs Next 1 Bar Return
            ic_1, p_1 = calculate_ic(series, features['Ret_1'])
            
            # IC (Predictive Power) vs Next 5 Bars Return
            ic_5, p_5 = calculate_ic(series, features['Ret_5'])
            
            row = {
                'Asset': filename.split('.')[0],
                'Feature': feat_name,
                'Stationary?': stat_status,
                'ADF p-val': f"{stat_p:.4f}",
                'IC (1-step)': f"{ic_1:.4f}",
                'IC (5-step)': f"{ic_5:.4f}"
            }
            summary_report.append(row)
            
    # Display Report
    print("\n================ VERIFICATION REPORT ================")
    report_df = pd.DataFrame(summary_report)
    
    # Adjust display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    
    # Group by Asset for readability or just print
    print(report_df)
    print("=====================================================")
    print("\nInterpretation Guide:")
    print("- Stationary?: MUST be YES for 'Oscillation' and 'Residuals'.")
    print("- IC (Information Coefficient):")
    print("  * > 0.02 or < -0.02: Weak Predictive Power")
    print("  * > 0.05 or < -0.05: Strong Predictive Power")
    print("  * Mean Reversion Features (Oscillator) should often have Negative IC (High value -> Price drops).")
    print("=====================================================")

if __name__ == "__main__":
    main()
