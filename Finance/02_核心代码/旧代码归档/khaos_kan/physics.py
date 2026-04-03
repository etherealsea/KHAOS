import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableEKF(nn.Module):
    def __init__(self, dt=1.0):
        super().__init__()
        self.dt = dt
        # Observation matrix H_obs (1x2): observe price [1, 0]
        self.register_buffer('H_obs', torch.tensor([[1.0, 0.0]]))
        self.register_buffer('I', torch.eye(2))

    def forward(self, price, hurst, vol_sigma):
        """
        price: [B, T]
        hurst: [B, T] - Hurst exponent for damping
        vol_sigma: [B, T] - Volatility for adaptive Q/R
        """
        B, T = price.shape
        device = price.device
        
        # Initial State [p, v]
        # Estimate initial v from first two points
        v0 = (price[:, 1] - price[:, 0]) / self.dt
        x = torch.stack([price[:, 0], v0], dim=1).unsqueeze(2) # [B, 2, 1]
        P = torch.eye(2, device=device).unsqueeze(0).repeat(B, 1, 1) * 1.0
        
        # Store states
        states = []
        
        # Loop over time
        for t in range(T):
            z = price[:, t].view(B, 1, 1)
            h_val = hurst[:, t].view(B, 1, 1)
            sigma = vol_sigma[:, t].view(B, 1, 1)
            
            # 1. Damping Function rho(H)
            # rho = 0.5 + 0.5 / (1 + exp(-10 * (H - 0.5)))
            rho = 0.5 + 0.5 * torch.sigmoid(10 * (h_val - 0.5))
            
            # 2. State Transition Matrix F
            # p_{t+1} = p_t + v_t * dt * rho
            # v_{t+1} = v_t
            # F = [[1, dt*rho], [0, 1]]
            F = torch.eye(2, device=device).unsqueeze(0).repeat(B, 1, 1)
            F[:, 0, 1] = self.dt * rho.squeeze()
            
            # 3. Adaptive Q/R
            # Q, R proportional to max(1, 100*sigma) (simplified scaling)
            scale = torch.clamp(100 * sigma, min=1.0)
            Q = torch.eye(2, device=device).unsqueeze(0).repeat(B, 1, 1) * scale * 0.1 # Process noise
            R = scale * 1.0 # Measurement noise (scalar in matrix form)
            R = R.view(B, 1, 1)
            
            # --- Predict ---
            x_pred = torch.bmm(F, x)
            P_pred = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q
            
            # --- Update ---
            # y = z - H x
            H_mat = self.H_obs.unsqueeze(0).repeat(B, 1, 1)
            y_res = z - torch.bmm(H_mat, x_pred)
            
            # S = H P H^T + R
            S = torch.bmm(torch.bmm(H_mat, P_pred), H_mat.transpose(1, 2)) + R
            
            # K = P H^T S^-1
            # S is scalar (1x1), so inverse is 1/S
            S_inv = 1.0 / (S + 1e-6)
            K = torch.bmm(torch.bmm(P_pred, H_mat.transpose(1, 2)), S_inv)
            
            # x = x + K y
            x = x_pred + torch.bmm(K, y_res)
            
            # P = (I - K H) P
            I_mat = self.I.unsqueeze(0).repeat(B, 1, 1)
            P = torch.bmm(I_mat - torch.bmm(K, H_mat), P_pred)
            
            states.append(x.squeeze(2)) # [B, 2]
            
        return torch.stack(states, dim=1) # [B, T, 2]

def calculate_hurst_proxy(price, window=16):
    """
    H ≈ 0.5 + 0.3 * tanh(m/10)
    m: slope of log-log regression of Price Range vs Time Lag
    Simplified: Rolling Rescaled Range or similar.
    Plan says: "Least squares closed form solution" for m.
    We implement a small batch estimator.
    """
    # For speed and differentiability, we use a fixed lag set for m estimation
    # e.g. lags 2, 4, 8.
    # log(R/S) ~ H * log(tau) + C
    # m is the slope H? No, formula says H = 0.5 + 0.3 tanh(m/10). 
    # This implies m is NOT H, but a proxy parameter.
    # Assuming m is the slope of pure price difference?
    # Let's assume m is the standard Hurst slope estimated from raw data, 
    # and the tanh formula is a squashing function to keep it in range.
    
    # We will use a simplified Volatility Scaling proxy for m.
    # m ~ log(High-Low) / log(Window)? 
    # Let's use the simplest differentiable proxy: 
    # log(Mean(|Price_t - Price_{t-tau}|)) vs log(tau)
    
    B, T = price.shape
    # We need to compute H for each time step t (using past window)
    # Using unfold to get windows
    # padding to keep size T
    pad = window - 1
    price_padded = F.pad(price, (pad, 0), mode='replicate')
    windows = price_padded.unfold(dimension=1, size=window, step=1) # [B, T, window]
    
    # Calculate R/S in the window
    # Range: max - min
    w_max = windows.max(dim=2)[0]
    w_min = windows.min(dim=2)[0]
    R = w_max - w_min + 1e-8
    
    # Std dev
    S = windows.std(dim=2) + 1e-8
    
    RS = R / S
    
    # Since window size is fixed (tau = window), we can't do regression on variable tau inside one window 
    # unless we sub-window.
    # Simplified: H_proxy ~ log(RS) / log(window/2)
    # m = log(RS) * const
    
    log_rs = torch.log(RS)
    log_tau = torch.log(torch.tensor(float(window), device=price.device))
    
    m = log_rs # Just use log(R/S) as the driver
    
    # Formula: H = 0.5 + 0.3 * tanh(m/10)
    # We need to calibrate m so that H is reasonable.
    # Normally H ~ 0.5. tanh(0) = 0.
    # If m is centered around 0?
    # R/S grows with sqrt(N). log(R/S) ~ 0.5 log(N).
    # We want m to represent deviation from random walk.
    # Let m = (log(R/S) - 0.5 * log(N)) * 10 ?
    
    expected_log_rs = 0.5 * log_tau
    deviation = log_rs - expected_log_rs
    
    # Scaling factor to make tanh effective
    m_scaled = deviation * 20.0 
    
    h_est = 0.5 + 0.3 * torch.tanh(m_scaled / 10.0)
    
    return h_est

def permutation_entropy(price, order=3, delay=1, window_size=50):
    """
    Calculate PE for each time step using a window.
    Actually, standard PE is calculated ON a window.
    We need a rolling PE.
    Window size for counting patterns? E.g. 50.
    """
    B, T = price.shape
    device = price.device
    
    # Pad
    pad = window_size - 1 + (order - 1) * delay
    price_padded = F.pad(price, (pad, 0), mode='replicate')
    
    # Extract windows for PE calculation
    # Each window is used to compute ONE entropy value
    # Size needed: window_size + (order-1)*delay
    total_window = window_size + (order - 1) * delay
    
    # Unfold to get the sequence for each time step
    windows = price_padded.unfold(dimension=1, size=total_window, step=1) # [B, T, total_window]
    
    # Now for each window, we extract 'window_size' patterns of length 'order'
    # We can do this by unfolding the inner dimension
    # patterns: [B, T, window_size, order]
    patterns = windows.unfold(dimension=2, size=order, step=delay) # This might miss the end if not careful
    # We want exactly window_size patterns.
    # windows size is (window_size + (order-1)*delay)
    # Unfolding with size=order, step=delay
    # Count = (total_window - order) / delay + 1 = (window_size + (order-1)d - order)/d + 1
    # = (window_size - 1)/d + order - 1 ... math check
    # Let's simplify: 
    # We need 'window_size' vectors of length 'order'.
    # We construct them manually.
    
    # Slice approach
    slices = []
    for i in range(order):
        # start = i * delay
        # end = start + window_size
        # We need to slice in the 'total_window' dim
        s = windows[:, :, i*delay : i*delay + window_size]
        slices.append(s)
        
    pattern_stack = torch.stack(slices, dim=3) # [B, T, window_size, order]
    
    # Argsort to get permutations
    # This is non-differentiable
    perms = torch.argsort(pattern_stack, dim=3)
    
    # Map permutations to unique integers
    # For order 3, perms are 0,1,2.
    # Base 10 or similar encoding
    # 3! = 6 permutations.
    # val = p0 * order^0 + p1 * order^1 ...
    coeffs = torch.tensor([order**i for i in range(order)], device=device).float()
    perm_indices = (perms.float() * coeffs).sum(dim=3) # [B, T, window_size]
    
    # Count frequencies
    # Since order is small (3-5), max index is small (3^3=27, 5^5=3125).
    # We can use bincount per row? Too slow.
    # We use scatter_add or simple histogram logic.
    # Since we need batch processing, maybe just Softmax over distances to prototypes?
    # For "Hard" PE, just use histograms.
    # We can treat this part as no_grad.
    
    entropy_list = []
    # Loop over T is unavoidable for variable batch histogram unless we use specialized kernels
    # Or reshape [B*T, window_size]
    flat_perms = perm_indices.view(-1, window_size)
    
    # Max unique ID
    max_id = int(order ** order) # Upper bound
    
    # Histogram: [B*T, max_id]
    # Memory efficient implementation using bincount
    
    num_windows = flat_perms.shape[0] # B*T
    
    # Create offsets for each window to flatten the batch dimension
    # We want to count per-window.
    # indices = flat_perms + row_idx * max_id
    
    offsets = torch.arange(num_windows, device=device).unsqueeze(1) * max_id
    flat_indices = (flat_perms + offsets).view(-1).long()
    
    # Total bins needed
    total_bins = num_windows * max_id
    
    # bincount
    counts_flat = torch.bincount(flat_indices, minlength=total_bins)
    
    # Reshape back to [B*T, max_id]
    counts = counts_flat.view(num_windows, max_id).float()
    
    probs = counts / window_size
    
    # Entropy: - sum p log p
    # add epsilon
    probs = probs + 1e-8
    log_probs = torch.log(probs)
    entropy = - (probs * log_probs).sum(dim=1)
    
    # Normalize by log(factorial(order))
    # 3! = 6. log(6).
    import math
    norm_factor = math.log(math.factorial(order))
    entropy = entropy / norm_factor
    
    return entropy.view(B, T)

class PhysicsLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ekf = DifferentiableEKF()
        
    def forward(self, x):
        # x: [B, Window, 15] 
        # (O, H, L, C, V, EMA20, RVOL, PinBull, PinBear, SweepHigh, SweepLow, FVGBull, FVGBear, MSSBull, MSSBear)
        
        with torch.no_grad():
             open_price = x[:, :, 0]
             high_price = x[:, :, 1]
             low_price = x[:, :, 2]
             close = x[:, :, 3]
             volume = x[:, :, 4]
             ema20 = x[:, :, 5]
             rvol = x[:, :, 6]
             pin_bull = x[:, :, 7]
             pin_bear = x[:, :, 8]
             sweep_high = x[:, :, 9]
             sweep_low = x[:, :, 10]
             fvg_bull = x[:, :, 11]
             fvg_bear = x[:, :, 12]
             mss_bull = x[:, :, 13]
             mss_bear = x[:, :, 14]
             
             # Use Log-Price for EKF to be scale-invariant
             log_close = torch.log(close + 1e-8)
             
             # 1. Hurst
             h_t = calculate_hurst_proxy(log_close)
             
             # 2. Volatility
             ret = torch.diff(log_close, dim=1, prepend=log_close[:, :1])
             vol = ret.abs()
             
             # 3. EKF
             ekf_states = self.ekf(log_close, h_t, vol)
             ekf_v = ekf_states[:, :, 1]
             ekf_p = ekf_states[:, :, 0]
             
             # 4. Entropy
             pe = permutation_entropy(close, order=3)
             
             # 5. Assemble 5D Physics State
             res = log_close - ekf_p
             
             # 6. Strict Price Features (User Constraints: KHAOS + Price + Vol + EMA20)
             
             # 6.1 Price Momentum (Raw Returns)
             price_mom = ret
             
             # 6.2 EMA20 Divergence (Reversal Signal)
             # (Close - EMA20) / EMA20
             ema_div = (close - ema20) / (ema20 + 1e-8)
             
             # 6.3 Volume Change
             prev_vol = torch.roll(volume, 1, dims=1)
             prev_vol[:, 0] = volume[:, 0]
             vol_change = torch.log((volume + 1e-8) / (prev_vol + 1e-8))
             
             # 6.4 High-Low Range (Volatility/Activity)
             hl_range = (high_price - low_price) / (close + 1e-8)
             
             # 6.5 Close-Open Body (Directional Strength)
             body = (close - open_price) / (close + 1e-8)
             
             # 7. SMC/PA Aggregates
             # Combine signals into signed features
             # Pin: +1 Bull, -1 Bear
             pin_signal = pin_bull - pin_bear
             # Sweep: +1 Low Sweep (Bullish), -1 High Sweep (Bearish)
             sweep_signal = sweep_low - sweep_high
             # FVG: +1 Bull, -1 Bear
             fvg_signal = fvg_bull - fvg_bear
             # MSS: +1 Bull, -1 Bear
             mss_signal = mss_bull - mss_bear
             
             psi = torch.stack([
                 h_t, vol, ekf_v, res, pe,
                 price_mom, ema_div, rvol, hl_range, body,
                 pin_signal, sweep_signal, fvg_signal, mss_signal
             ], dim=2)
         
        return psi

