import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import sys
import os

# Add src to path for KAN import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/archive'))
try:
    from khaos_framework.models.kan_model import KAN
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    try:
        from archive.khaos_framework.models.kan_model import KAN
    except ImportError:
        class KAN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, layers=2):
                super(KAN, self).__init__()
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                for _ in range(layers - 2):
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Linear(hidden_dim, output_dim))
                
            def forward(self, x):
                for layer in self.layers[:-1]:
                    x = torch.relu(layer(x))
                x = self.layers[-1](x)
                return x

# ------------------------------------------------------------------------
# 1. Data Loading
# ------------------------------------------------------------------------
def load_raw_data_list():
    """
    Loads raw dataframes from multiple assets and timeframes.
    Assets: BTC, EURUSD, Gold_ETF, SPX_Index, TSLA
    Timeframes: 5m, 15m, 1h, 4h
    """
    base_path = os.path.join(os.path.dirname(__file__), '../../data/model_research/data')
    assets = ["BTC"] # Reduced for debugging
    tfs = ["5m", "15m", "1h", "4h"]
    
    raw_data_list = []
    
    print("Loading raw data files...")
    for asset in assets:
        for tf in tfs:
            fname = f"{asset}_{tf}.csv"
            fpath = os.path.join(base_path, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    # Standardize columns
                    df.columns = [c.capitalize() for c in df.columns]
                    if 'Close' not in df.columns:
                        col = next((c for c in df.columns if c.lower() == 'close'), None)
                        if col: df.rename(columns={col: 'Close'}, inplace=True)
                    
                    if 'Close' in df.columns:
                        df = df[['Close']].dropna()
                        if len(df) > 500: # Minimum length check
                            raw_data_list.append((fname, df))
                            print(f"Loaded {fname} ({len(df)} rows)")
                        else:
                            print(f"Skipped {fname} (Length {len(df)} < 500)")
                    else:
                        print(f"Skipped {fname} (No Close column)")
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
            else:
                 print(f"Skipped {fname} (not found)")
                
    print(f"Loaded {len(raw_data_list)} valid data files.")
    return raw_data_list

# ------------------------------------------------------------------------
# 2. Feature Engineering (EMA-based Normalization)
# ------------------------------------------------------------------------
def process_dataframe(df, norm_window):
    """
    Generates 5D input features: [Hurst, Vol, Bias, Slope, Entropy]
    Uses EMA-based Normalization with `norm_window`.
    """
    df = df.copy()
    close = df['Close']
    
    # --- Raw Indicators ---
    
    # 1. Hurst Proxy (Trend Persistence)
    returns = np.log(close / close.shift(1))
    
    # Hurst Logic (Fixed parameters for indicator stability)
    ma_short = close.rolling(10).mean()
    ma_long = close.rolling(30).mean()
    autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
    hurst_proxy = 0.5 + 0.5 * autocorr.fillna(0)
    
    # 2. Volatility (Energy)
    # Use standard rolling std for the raw indicator, but normalization will be EMA
    vol_proxy = returns.rolling(20).std() * 100
    
    # 3. Bias (Deviation from Fair Value - Now Unified to EMA 20)
    # Using EMA 20 aligns with Price Action theory and the normalization window
    ema20 = close.ewm(span=20, adjust=False).mean()
    bias_raw = (close - ema20) * 100.0 # Raw distance
    
    # 4. Slope (Kinetic Energy)
    slope_raw = (np.log(close) - np.log(close.shift(5))) * 100.0
    
    # 5. Entropy (Chaos/Complexity)
    entropy_proxy = slope_raw.rolling(14).std()
    
    # --- EMA UNIVERSAL NORMALIZATION ---
    # Replacing Rolling Z-Score with EMA Z-Score
    # Z = (X - EMA(X)) / EMA_STD(X)
    # Tanh mapping
    
    def ema_z_tanh(series, span):
        ema = series.ewm(span=span, adjust=False).mean()
        # EMA Standard Deviation Approximation
        # std = sqrt( EMA( (x - ema)^2 ) )
        # Note: pandas ewm.std() calculates bias-weighted std deviation
        std = series.ewm(span=span, adjust=False).std()
        
        z = (series - ema) / (std + 1e-9)
        return np.tanh(z)

    # Apply Normalization
    # Hurst is centered around 0.5, but we normalize it to relative changes
    hurst_norm = ema_z_tanh(hurst_proxy, norm_window)
    vol_norm = ema_z_tanh(vol_proxy, norm_window)
    bias_norm = ema_z_tanh(bias_raw, norm_window)
    slope_norm = ema_z_tanh(slope_raw, norm_window)
    entropy_norm = ema_z_tanh(entropy_proxy, norm_window)

    # Target Construction
    # High Vol + High Hurst = Trend (Q)
    # High Vol + Low Hurst = Chaos (R)
    
    h_weight = (hurst_norm + 1.0) / 2.0
    v_mag = np.abs(vol_norm)
    
    target_Q = v_mag * h_weight
    target_R = v_mag * (1.0 - h_weight)
    
    target_momentum = (close.shift(-5) / close - 1) * 100
    target_M = ema_z_tanh(target_momentum, norm_window)
    
    data = pd.DataFrame({
        'Hurst': hurst_norm,
        'Vol': vol_norm,
        'Bias': bias_norm,
        'Slope': slope_norm,
        'Entropy': entropy_norm,
        'Target_Q': target_Q,
        'Target_R': target_R,
        'Target_M': target_M
    }).dropna()
    
    return data

# ------------------------------------------------------------------------
# 3. Experiment Loop
# ------------------------------------------------------------------------
def train_and_eval(df):
    # Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    X_train = torch.tensor(train_df[['Hurst', 'Vol', 'Bias', 'Slope', 'Entropy']].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[['Target_Q', 'Target_R', 'Target_M']].values, dtype=torch.float32)
    
    X_test = torch.tensor(test_df[['Hurst', 'Vol', 'Bias', 'Slope', 'Entropy']].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[['Target_Q', 'Target_R', 'Target_M']].values, dtype=torch.float32)
    
    # Model
    model = KAN(input_dim=5, hidden_dim=8, output_dim=3, layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(50): # Increased for better convergence
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
        inputs = X_test.numpy()
        
    # Calculate R2 for Linear Distillation (Pine Compatibility)
    r2_scores = []
    features = ['Hurst', 'Vol', 'Bias', 'Slope', 'Entropy']
    outputs = ['Q', 'R', 'M']
    
    weights_log = ""
    
    total_r2 = 0
    
    for i, out_name in enumerate(outputs):
        reg = LinearRegression()
        reg.fit(inputs, preds[:, i])
        score = reg.score(inputs, preds[:, i])
        r2_scores.append(score)
        total_r2 += score
        
        weights_log += f"// {out_name} Weights (R2: {score:.2f})\n"
        for j, f in enumerate(features):
            w = reg.coef_[j]
            weights_log += f"float w_{out_name}_{f} = {w:.5f}\n"
        weights_log += f"float b_{out_name} = {reg.intercept_:.5f}\n\n"
        
    avg_r2 = total_r2 / 3.0
    return avg_r2, weights_log

def main():
    # Write output to file to avoid terminal truncation
    with open('d:/Finance/Finance/training_results.txt', 'w') as f:
        f.write("Starting Training...\n")
        
        raw_data = load_raw_data_list()
        if not raw_data:
            f.write("No data found.\n")
            return
            
        f.write(f"Loaded {len(raw_data)} data files:\n")
        for fname, _ in raw_data:
            f.write(f" - {fname}\n")
        
        # Experiment Loop
        # We only run one window (100) now as it's proven best
        windows = [100]
        best_window = 0
        best_score = -999
        best_weights = ""
        
        # Fast run for weights extraction
        EPOCHS = 5
        
        f.write("\n--- Starting Granular Period Verification ---\n")
        print("Starting Granular Period Verification...")
        
        for w in windows:
            try:
                msg = f"Testing Window: {w}..."
                print(msg)
                f.write(msg + "\n")
                
                processed_dfs = []
                for fname, df in raw_data:
                    # print(f"  Processing {fname}...")
                    processed_dfs.append(process_dataframe(df, w))
                    
                if not processed_dfs:
                    continue
                    
                combined_df = pd.concat(processed_dfs, ignore_index=True)
                score, weights = train_and_eval(combined_df)
                
                res = f"Window {w} -> Avg Distillation R2: {score:.4f}"
                print(res)
                f.write(res + "\n")
                
                if score > best_score:
                    best_score = score
                    best_window = w
                    best_weights = weights
            except Exception as e:
                err = f"Error in Window {w}: {e}"
                print(err)
                f.write(err + "\n")
                import traceback
                traceback.print_exc()
                
        f.write("\n========================================\n")
        f.write(f"BEST WINDOW: {best_window} (R2: {best_score:.4f})\n")
        f.write("========================================\n")
        f.write(best_weights)
        
        # ALSO PRINT TO STDOUT FOR TRAE TO READ
        print("\nFINAL_RESULTS_START")
        print(f"BEST WINDOW: {best_window} (R2: {best_score:.4f})")
        print(best_weights)
        print("FINAL_RESULTS_END")
        
        print("Done. Results written to training_results.txt")

if __name__ == "__main__":
    main()
