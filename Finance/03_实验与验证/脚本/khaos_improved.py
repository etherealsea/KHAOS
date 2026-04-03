import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class KinematicKalmanFilter:
    """
    A Kinematic Kalman Filter (Position + Velocity) for trending markets.
    State vector x = [position, velocity]^T
    """
    def __init__(self, dt=1.0, R_base=0.01, Q_base=1e-5):
        self.dt = dt
        self.R_base = R_base
        self.Q_base = Q_base
        
        # State transition matrix (Position = Position + Velocity * dt)
        self.F = np.array([[1, dt],
                           [0, 1]])
        
        # Observation matrix (We only observe Position)
        self.H = np.array([[1, 0]])
        
        # Initial state covariance
        self.P = np.eye(2) * 1.0
        
        # Initial state [p, v]
        self.x = np.zeros((2, 1))
        
    def update(self, z, adaptive_R_factor=1.0, adaptive_Q_factor=1.0):
        """
        z: measurement (price)
        adaptive_R_factor: multiplier for R (high noise -> high R)
        adaptive_Q_factor: multiplier for Q (high trend change -> high Q)
        """
        # Adaptive Q (Process Noise Covariance)
        # Q represents uncertainty in the model (e.g., velocity changes)
        # If market is trending (high velocity stability), Q for velocity can be lower?
        # Actually, if trend is *changing*, we need higher Q to adapt.
        # But let's stick to a base structure:
        # Q = [ [dt^4/4, dt^3/2], [dt^3/2, dt^2] ] * sigma_a^2 (Continuous White Noise Acceleration)
        # Simplified discrete Q:
        q_scale = self.Q_base * adaptive_Q_factor
        Q = np.array([[q_scale * (self.dt**3)/3, q_scale * (self.dt**2)/2],
                      [q_scale * (self.dt**2)/2, q_scale * self.dt]])
        
        # Adaptive R (Measurement Noise Covariance)
        R = np.array([[self.R_base * adaptive_R_factor]])
        
        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + Q
        
        # --- Update ---
        y = z - (self.H @ x_pred) # Innovation
        S = self.H @ P_pred @ self.H.T + R # Innovation Covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return self.x[0, 0] # Return filtered position

class KHAOS_Improved:
    def __init__(self):
        pass

    def calculate_efficiency_ratio(self, prices, window=10):
        """
        Kaufman Efficiency Ratio (ER) as a fast proxy for Fractal Dimension.
        ER = Direction / Volatility
        ER -> 1 (Trend), ER -> 0 (Noise)
        """
        diff = np.diff(prices, prepend=prices[0])
        abs_diff = np.abs(diff)
        
        # Direction: Change over window
        direction = np.abs(prices - np.roll(prices, window))
        # Volatility: Sum of absolute changes
        volatility = pd.Series(abs_diff).rolling(window).sum().values
        
        # Avoid division by zero
        er = np.divide(direction, volatility, out=np.zeros_like(direction), where=volatility!=0)
        
        # Fix first 'window' elements
        er[:window] = 0.5 
        return er

    def process_data(self, df):
        if 'Close' not in df.columns and 'close' in df.columns:
            df['Close'] = df['close']
        
        prices = df['Close'].values
        log_prices = np.log(prices) # Work in log space as per whitepaper
        n = len(prices)
        
        # 1. Calculate Adaptive Factors
        # Use Efficiency Ratio (ER) instead of slow DFA
        # High ER (Trend) -> Low R (Trust Measurement), High Q (Allow Velocity)
        # Low ER (Noise) -> High R (Trust Model/Smooth), Low Q (Stable Velocity)
        er = self.calculate_efficiency_ratio(log_prices, window=14)
        
        # Volatility for R scaling
        returns = np.diff(log_prices, prepend=log_prices[0])
        vol = pd.Series(returns).rolling(20).std().fillna(0).values
        # Normalize vol
        avg_vol = pd.Series(vol).rolling(100).mean().fillna(vol.mean()).values
        vol_ratio = vol / (avg_vol + 1e-9)
        
        # 2. Run Kinematic Kalman Filter
        kf = KinematicKalmanFilter(dt=1.0, R_base=0.0001, Q_base=1e-5)
        kalman_out = np.zeros(n)
        
        for i in range(n):
            # Adaptive Logic:
            # If ER is high (Trend), we want less lag -> Lower R, Higher Q
            # If ER is low (Chop), we want smoothing -> Higher R, Lower Q
            
            # Map ER [0, 1] to factors
            # R factor: High ER -> 0.1, Low ER -> 10.0
            # Using a Sigmoid-like mapping or simple power
            r_factor = 1.0 + 10.0 * (1.0 - er[i])**2 # Quadratic decay: ER=1->R_factor=1, ER=0->R_factor=11
            # Also scale by volatility: High vol -> Need higher R to smooth noise? 
            # Or High vol usually means signal? In trends high vol is signal. In chop high vol is noise.
            # Let's stick to ER.
            
            # Q factor: High ER -> High Q (allow state to move fast)
            q_factor = 1.0 + 10.0 * (er[i])**2
            
            kalman_out[i] = kf.update(log_prices[i], adaptive_R_factor=r_factor, adaptive_Q_factor=q_factor)
            
        # Convert back to linear
        kalman_price = np.exp(kalman_out)
        
        # 3. Comparison Metrics (Simple Lag)
        # Compare with EMA(20)
        ema_20 = pd.Series(prices).ewm(span=20).mean().values
        
        result = pd.DataFrame({
            'Price': prices,
            'LogPrice': log_prices,
            'KHAOS_Improved': kalman_price,
            'EMA_20': ema_20,
            'ER': er,
            'VolRatio': vol_ratio
        }, index=df.index)
        
        return result

if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv("d:/Finance/Finance/model_research/data/BTC_1h.csv")
        # Ensure datetime index if needed, but simple index is fine for test
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
        print(f"Loaded {len(df)} rows of BTC data.")
        
        model = KHAOS_Improved()
        res = model.process_data(df)
        
        # Calculate basic lag/error metrics
        # MSE against future price (predictive power?) or just current price (tracking?)
        # Let's check "Zero Lag" claim:
        # Calculate correlation between Price Change and Indicator Change
        
        price_change = res['Price'].diff()
        khaos_change = res['KHAOS_Improved'].diff()
        ema_change = res['EMA_20'].diff()
        
        corr_khaos = price_change.corr(khaos_change)
        corr_ema = price_change.corr(ema_change)
        
        print(f"Correlation (Price Change vs Indicator Change):")
        print(f"KHAOS Improved: {corr_khaos:.4f}")
        print(f"EMA 20:         {corr_ema:.4f}")
        
        # Check RMSE during high volatility
        mse_khaos = ((res['Price'] - res['KHAOS_Improved'])**2).mean()
        mse_ema = ((res['Price'] - res['EMA_20'])**2).mean()
        
        print(f"\nRMSE (Tracking Error):")
        print(f"KHAOS Improved: {np.sqrt(mse_khaos):.2f}")
        print(f"EMA 20:         {np.sqrt(mse_ema):.2f}")
        
        print("\nDone. Improved model prototype runs successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
