import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FinancialDataset(Dataset):
    def __init__(self, df, window_size=64, forecast_horizon=4, volatility_window=40):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        # Ensure columns are standardized
        df.columns = [c.lower().strip() for c in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
                
        self.close = df['close'].values.astype(np.float32)
        self.open = df['open'].values.astype(np.float32)
        self.high = df['high'].values.astype(np.float32)
        self.low = df['low'].values.astype(np.float32)
        self.volume = df['volume'].values.astype(np.float32)
        
        # Calculate Volatility (Sigma_t) for target normalization
        # Using Rolling Std of Returns
        # Log returns for volatility calculation
        log_close = np.log(np.maximum(self.close, 1e-8))
        returns = np.diff(log_close, prepend=log_close[0])
        
        s = pd.Series(returns)
        self.sigma = s.rolling(window=volatility_window).std().bfill().values
        self.sigma = np.maximum(self.sigma, 1e-6).astype(np.float32)
        
        # --- FEATURE ENGINEERING (SMC & PA) ---
        
        # 1. RVOL (Relative Volume)
        # RVOL = Volume / SMA(Volume, 20)
        # Add 1e-8 to avoid div by zero
        vol_series = pd.Series(self.volume)
        vol_sma = vol_series.rolling(window=20).mean().bfill().values
        self.rvol = (self.volume / (vol_sma + 1e-8)).astype(np.float32)
        # Log transform RVOL to squash outliers
        self.rvol = np.log1p(self.rvol)
        
        # 2. EMA20 Calculation
        self.ema20 = pd.Series(self.close).ewm(span=20, adjust=False).mean().values.astype(np.float32)
        
        # 3. Pin Bar (Rejection) Detection
        # Upper Wick = High - max(Open, Close)
        # Lower Wick = min(Open, Close) - Low
        # Body = abs(Close - Open)
        # Pin Bar: Wick > 2 * Body AND Wick > Avg_Body
        upper_wick = self.high - np.maximum(self.open, self.close)
        lower_wick = np.minimum(self.open, self.close) - self.low
        body_size = np.abs(self.close - self.open)
        avg_body = pd.Series(body_size).rolling(10).mean().bfill().values
        
        # Bullish Pin (Long Lower Wick)
        self.pin_bull = ((lower_wick > 2 * body_size) & (lower_wick > 0.5 * avg_body)).astype(np.float32)
        # Bearish Pin (Long Upper Wick)
        self.pin_bear = ((upper_wick > 2 * body_size) & (upper_wick > 0.5 * avg_body)).astype(np.float32)
        
        # 4. Liquidity Sweep (Simple Proxy)
        # High > Rolling Max(10) but Close < Rolling Max(10) (Fakeout)
        roll_high = pd.Series(self.high).rolling(10).max().shift(1).bfill().values
        roll_low = pd.Series(self.low).rolling(10).min().shift(1).bfill().values
        
        self.sweep_high = ((self.high > roll_high) & (self.close < roll_high)).astype(np.float32)
        self.sweep_low = ((self.low < roll_low) & (self.close > roll_low)).astype(np.float32)
        
        # 5. FVG (Fair Value Gap) Strength
        # Bullish FVG: Low[t] > High[t-2]
        # Bearish FVG: High[t] < Low[t-2]
        # We compute gap size relative to volatility
        high_shift2 = np.roll(self.high, 2)
        low_shift2 = np.roll(self.low, 2)
        # Mask first 2
        high_shift2[:2] = self.high[:2]
        low_shift2[:2] = self.low[:2]
        
        # Gap Size (Positive = Gap exists)
        bull_gap = (self.low - high_shift2) / (self.sigma + 1e-8)
        bear_gap = (low_shift2 - self.high) / (self.sigma + 1e-8)
        
        self.fvg_bull = np.maximum(bull_gap, 0).astype(np.float32)
        self.fvg_bear = np.maximum(bear_gap, 0).astype(np.float32)
        
        # 6. Market Structure Shift (MSS) / Break of Structure (BOS)
        # Identify Swings (Fractals): High[t] > High[t-2, t-1, t+1, t+2]
        # Since we can't look ahead (t+1, t+2), we use lagging Swings.
        # Swing High Confirmed at t: High[t-2] was the max of [t-4, t].
        # We need to maintain the "Last Swing High/Low" state.
        # Vectorized approach:
        # 1. Identify Pivot candidates (Rolling Max over 5)
        # 2. Propagate last pivot level forward
        # 3. Check for break
        
        # Rolling Max/Min (window=5, center=True would be ideal but causal requires lag)
        # We define Swing High at t-2 if H[t-2] is max of H[t-4:t+1]. 
        # Causal: We confirm Swing High at 't' that happened at 't-2'.
        # H[t-2] > H[t-4], H[t-3], H[t-1], H[t].
        
        h = self.high
        l = self.low
        c = self.close
        
        # Shifted arrays
        # h0 (t), h1 (t-1), h2 (t-2), h3 (t-3), h4 (t-4)
        # Note: slicing is [start:end], so h[4:] aligns with t=4
        # We need to ensure we compare values at the same relative index
        
        # Slices to represent lagged versions
        # Original: h[0], h[1], h[2], h[3], h[4]
        # At i=0 in sliced array, we are looking at t=4 in original array.
        # h0[0] = h[4] (t)
        # h1[0] = h[3] (t-1)
        # h2[0] = h[2] (t-2) -> Candidate
        # h3[0] = h[1] (t-3)
        # h4[0] = h[0] (t-4)
        
        h0, h1, h2, h3, h4 = h[4:], h[3:-1], h[2:-2], h[1:-3], h[:-4]
        l0, l1, l2, l3, l4 = l[4:], l[3:-1], l[2:-2], l[1:-3], l[:-4]
        
        # 1. Find Swings (Lag 2)
        is_swing_high = (h2 > h0) & (h2 > h1) & (h2 > h3) & (h2 > h4)
        is_swing_low = (l2 < l0) & (l2 < l1) & (l2 < l3) & (l2 < l4)
        
        # Pad to match length
        # These signals appear at index `i` (which corresponds to time t), confirming swing at t-2.
        is_swing_high = np.concatenate(([False]*4, is_swing_high))
        is_swing_low = np.concatenate(([False]*4, is_swing_low))
        
        # 2. Propagate Levels
        # We need the VALUE of the last swing high/low at every point t
        last_sh_level = np.zeros_like(c)
        last_sl_level = np.zeros_like(c)
        
        curr_sh = h[0]
        curr_sl = l[0]
        
        for i in range(len(c)):
            if is_swing_high[i]:
                curr_sh = h[i-2] # The swing happened 2 steps ago
            if is_swing_low[i]:
                curr_sl = l[i-2]
            
            last_sh_level[i] = curr_sh
            last_sl_level[i] = curr_sl
            
        # 3. Detect Breaks
        # Bullish Break: Close crosses above Last Swing High
        # Bearish Break: Close crosses below Last Swing Low
        # Use close > sh AND prev_close <= prev_sh to detect crossover
        # Or just "Distance to Break"
        
        # Let's use "Break Signal" (1.0)
        self.mss_bull = (c > last_sh_level).astype(np.float32)
        self.mss_bear = (c < last_sl_level).astype(np.float32)
        
        # --------------------------------------
        
        # Sample Weights for Training
        # Prioritize: Extreme Price (Distance to EMA > 2*Sigma) OR PA Signal (Pin/Sweep)
        dist_ema = np.abs(self.close - self.ema20) / (self.sigma + 1e-8)
        is_extreme = (dist_ema > 2.0).astype(np.float32)
        has_signal = np.maximum(self.pin_bull, self.pin_bear)
        has_sweep = np.maximum(self.sweep_high, self.sweep_low)
        has_mss = np.maximum(self.mss_bull, self.mss_bear)
        
        # Weight formula: Base(1.0) + Extreme(4.0) + Signal(2.0) + MSS(3.0)
        self.sample_weights = 1.0 + (is_extreme * 4.0) + (has_signal * 2.0) + (has_sweep * 2.0) + (has_mss * 3.0)
        self.sample_weights = self.sample_weights.astype(np.float32)
        
        # Targets: Realized Volatility (Future)
        # Predict the standard deviation of returns over the next 'forecast_horizon' steps
        # Or simpler: The absolute return over horizon (magnitude of move)
        # Volatility is usually computed as StdDev of log returns.
        # Let's compute rolling std of log returns looking AHEAD.
        
        log_ret = np.diff(np.log(self.close + 1e-8), prepend=np.log(self.close[0] + 1e-8))
        s_log_ret = pd.Series(log_ret)
        
        # Future Volatility: Shifted backwards
        # Window size for vol calculation = horizon
        future_vol = s_log_ret.rolling(window=forecast_horizon).std().shift(-forecast_horizon).values
        
        # Fill NaN at the end
        future_vol = np.nan_to_num(future_vol, 0.0)
        
        # Normalize Target: Volatility Return (Log(Future/Current))
        # This makes the target scale-invariant and centered around 0.
        # It forces the model to learn "Change in Volatility" rather than "Level of Volatility".
        # This implicitly uses "Current Volatility" as the baseline (Clustering).
        
        # 1. Future Log Vol
        log_future_vol = np.log(future_vol + 1e-8)
        
        # 2. Current Log Vol (at prediction time t)
        # s_log_ret is aligned with close.
        # At index 'end_idx-1', the 'current' volatility is the recent realized vol.
        # Let's use a short window (e.g. 16) to define "Current Vol".
        # Why 16? Matches the Physics Hurst window roughly.
        current_vol = s_log_ret.rolling(window=16).std().bfill()
        log_current_vol = np.log(current_vol + 1e-8)
        
        # Target = Future - Current
        # We need to align indices.
        # self.targets[i] corresponds to end_idx-1.
        # So we need log_current_vol[end_idx-1].
        
        # Vectorized alignment
        # log_future_vol is already shifted to represent vol at t+horizon.
        # log_current_vol is at t.
        
        target_diff = log_future_vol - log_current_vol.values
        
        # Clip extreme changes (e.g. > 300% jump)
        # Log(3) ~ 1.1. Log(0.3) ~ -1.2.
        # Let's clip to [-2.0, 2.0]
        self.raw_targets = log_future_vol.astype(np.float32)
        self.targets = np.clip(target_diff, -2.0, 2.0).astype(np.float32)
        
        # Data Matrix: [Open, High, Low, Close, Volume, EMA20, RVOL, PinBull, PinBear, SweepHigh, SweepLow, FVGBull, FVGBear, MSSBull, MSSBear]
        # Total 15 Features
        self.data = np.stack([
            self.open, self.high, self.low, self.close, self.volume, self.ema20,
            self.rvol, self.pin_bull, self.pin_bear, 
            self.sweep_high, self.sweep_low,
            self.fvg_bull, self.fvg_bear,
            self.mss_bull, self.mss_bear
        ], axis=1)
        
    def __len__(self):
        # Last index we can start a window at
        # We need idx + window_size + forecast_horizon <= len
        # Actually target is at idx+window_size-1, looking ahead h
        # So we need data up to idx+window_size-1+h
        return len(self.close) - self.window_size - self.forecast_horizon
    
    def __getitem__(self, idx):
        # Input Window: [idx, idx + window_size]
        end_idx = idx + self.window_size
        x = self.data[idx : end_idx]
        
        # Target: computed at time (end_idx - 1), predicting (end_idx - 1 + h)
        y = self.targets[end_idx - 1]
        
        # Also return sigma for analysis (though not used in binary target)
        sigma = self.sigma[end_idx - 1]
        
        # Return sample weight
        weight = self.sample_weights[end_idx - 1]
        
        return torch.from_numpy(x), torch.tensor(y), torch.tensor(sigma), torch.tensor(weight)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Parse Time
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
    return df

def create_rolling_datasets(file_path, train_ratio=0.7, window_size=64, horizon=4, sample_per_year_months=3, subsample=True):
    df = load_data(file_path)
    
    # --- OPTIMIZATION: Subsample Data ---
    # Randomly select 'sample_per_year_months' months for each year
    if subsample and 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Get unique years
        years = df['year'].unique()
        
        sampled_indices = []
        for y in years:
            year_data = df[df['year'] == y]
            months = year_data['month'].unique()
            
            if len(months) > sample_per_year_months:
                # Randomly pick months
                selected_months = np.random.choice(months, sample_per_year_months, replace=False)
                # Filter
                mask = df['month'].isin(selected_months) & (df['year'] == y)
                sampled_indices.extend(df[mask].index.tolist())
            else:
                # Keep all if less than target
                mask = df['year'] == y
                sampled_indices.extend(df[mask].index.tolist())
                
        sampled_indices = sorted(sampled_indices)
        df = df.iloc[sampled_indices].reset_index(drop=True)
        print(f"  Subsampled data to {len(df)} rows ({sample_per_year_months} months/year)")
    # ------------------------------------
    
    total_len = len(df)
    train_size = int(total_len * train_ratio)
    
    train_df = df.iloc[:train_size]
    # Test set needs overlap to start predicting immediately after train set
    test_df = df.iloc[train_size - window_size:] 
    
    train_ds = FinancialDataset(train_df, window_size, horizon)
    test_ds = FinancialDataset(test_df, window_size, horizon)
    
    return train_ds, test_ds
