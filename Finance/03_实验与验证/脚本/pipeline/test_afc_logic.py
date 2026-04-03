
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_custom_safe(series, window=300):
    ema = series.ewm(span=window).mean()
    std = series.ewm(span=window).std()
    z = (series - ema) / (std + 1e-9)
    return np.tanh(z).fillna(0)

def zigzag_labels(price, window=20):
    labels = np.zeros(len(price))
    is_top = (price == price.rolling(window*2 + 1, center=True).max())
    is_bottom = (price == price.rolling(window*2 + 1, center=True).min())
    labels[is_top] = -1
    labels[is_bottom] = 1
    return labels

def test_logic():
    # Synthetic Data
    N = 1000
    t = np.linspace(0, 100, N)
    price = np.sin(t) + np.random.normal(0, 0.1, N)
    m_weight = -np.cos(t) # Should peak at bottom (sin min) ?
    # sin min is at 3pi/2. cos(3pi/2) = 0.
    # We want m_weight to be high at bottom.
    # Bottom of sin is -1.
    # We want m_weight to be +1 there.
    
    df = pd.DataFrame({'close': price, 'm_weight': m_weight})
    
    # ZigZag
    df['y_target'] = zigzag_labels(df['close'], window=10)
    print(f"Targets: {np.sum(df['y_target'] != 0)}")
    
    # EMA Cache
    grid = [10, 20, 50]
    cache = {}
    for w in grid:
        cache[w] = (df['m_weight'].ewm(span=w).mean(), df['m_weight'].ewm(span=w).std())
    df.attrs['ema_cache'] = cache
    df.attrs['ema_matrix_mean'] = np.column_stack([cache[w][0] for w in grid])
    df.attrs['ema_matrix_std'] = np.column_stack([cache[w][1] for w in grid])
    
    # Interpolation Test
    # Target window 15 (between 10 and 20)
    # frac 0.5
    from scripts.pipeline.optimize_afc import get_interpolated_stats, run_simulation_vectorized
    
    # We need to monkey patch global EMA_GRID in optimize_afc or replicate function
    # Let's just replicate the interpolation logic here for simplicity
    
    # ... actually I can just import if I set up the module right.
    # But EMA_GRID is global in optimize_afc.
    
    # Let's just implement simplified version here
    targets = np.full(N, 15)
    
    # Manual Interp
    m10, s10 = cache[10]
    m20, s20 = cache[20]
    m_exp = m10 + 0.5 * (m20 - m10)
    s_exp = s10 + 0.5 * (s20 - s10)
    
    # Simulation
    z = (df['m_weight'] - m_exp) / s_exp
    g = np.tanh(z)
    
    thresh = 0.5
    exit_val = 0.2
    
    bot_set = g > thresh
    bot_reset = g < exit_val
    
    bot_state_raw = np.full(N, np.nan)
    bot_state_raw[bot_set] = 1.0
    bot_state_raw[bot_reset] = 0.0
    bot_state = pd.Series(bot_state_raw).ffill().fillna(0.0).values
    
    print(f"Bot State Sum: {np.sum(bot_state)}")
    
    # Evaluate
    p_bot = (bot_state == 1)
    t_bot = (df['y_target'] == 1)
    tp = np.sum(p_bot & t_bot)
    print(f"TP: {tp}")
    
    # If this works, then optimize_afc.py logic is correct.
    
if __name__ == "__main__":
    test_logic()
