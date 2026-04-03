import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class KHAOS_Model:
    def __init__(self):
        self.lookback = 100
    
    def calculate_kalman(self, prices):
        """
        Simple 1D Kalman Filter for price smoothing/state estimation.
        x_t = x_{t-1} + w_t
        z_t = x_t + v_t
        """
        n = len(prices)
        xhat = np.zeros(n)      # a posteriori estimate of x
        P = np.zeros(n)         # a posteriori error estimate
        xhatminus = np.zeros(n) # a priori estimate of x
        Pminus = np.zeros(n)    # a priori error estimate
        K = np.zeros(n)         # gain or blending factor
        
        Q = 1e-5 # process variance
        R = 0.01 # estimate of measurement variance, change to see effect

        # Initial guesses
        xhat[0] = prices[0]
        P[0] = 1.0

        for k in range(1, n):
            # time update
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q

            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (prices[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
            
        return xhat, prices - xhat # Return smoothed and residuals

    def calculate_hurst(self, series, max_lag=20):
        """
        Calculate Hurst exponent using R/S analysis or simple variance ratio.
        Using a simplified approach for rolling window efficiency.
        """
        lags = range(2, max_lag)
        # Avoid division by zero or empty slices
        if len(series) < max_lag:
            return 0.5

        tau = []
        for lag in lags:
            diff = np.subtract(series[lag:], series[:-lag])
            if len(diff) == 0:
                tau.append(0)
            else:
                tau.append(np.sqrt(np.std(diff)))

        H = 0.5 # Default
        try:
            # Very basic implementation for testing structure
            # (In production, use `hurst` library or optimized numba code)
            Y = np.log(tau)
            X = np.log(lags)
            if len(X) > 1 and len(Y) > 1:
                H = np.polyfit(X, Y, 1)[0] * 2 # Rough approximation adjustment
        except:
            pass
        
        return np.clip(H, 0, 1)

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def calculate_khaos(self, residuals):
        """
        K = Normalize(BandPassFilter(residuals))
        Renamed from Oscillation to KHAOS
        """
        # Assume daily data, fs = 1
        # Filter for cycles between 3 days and 30 days
        # f = 1/T
        fs = 1.0
        lowcut = 1.0/30.0
        highcut = 1.0/3.0
        
        try:
            filtered = self.bandpass_filter(residuals, lowcut, highcut, fs)
            
            # Normalize (z-score rolling or min-max)
            # Using simple MinMax over recent window for 0-1 scaling or Z-score
            # Let's use Z-score-like normalization -> tanh for -1 to 1
            norm = np.tanh(filtered / (np.std(filtered) + 1e-6))
            return norm
        except Exception as e:
            print(f"Filter error: {e}")
            return np.zeros_like(residuals)

    def get_dynamic_thresholds(self, khaos_series, window=300, percentile=0.95):
        """
        Calculate dynamic upper and lower thresholds based on rolling percentiles.
        Returns: upper_threshold, lower_threshold (Series)
        """
        # Ensure input is a pandas Series
        if not isinstance(khaos_series, pd.Series):
            khaos_series = pd.Series(khaos_series)
            
        upper = khaos_series.rolling(window=window, min_periods=min(50, window)).quantile(percentile)
        lower = khaos_series.rolling(window=window, min_periods=min(50, window)).quantile(1 - percentile)
        
        # Fallback for initial period (use expanding or global)
        if upper.isnull().any():
            # Fill initial NaNs with expanding quantile or global stats
            expanding_upper = khaos_series.expanding(min_periods=50).quantile(percentile)
            expanding_lower = khaos_series.expanding(min_periods=50).quantile(1 - percentile)
            upper = upper.fillna(expanding_upper).fillna(0.8) # Default to 0.8 if all else fails
            lower = lower.fillna(expanding_lower).fillna(-0.8)
            
        return upper, lower

    def process_data(self, df):
        # Ensure Close is present
        if 'Close' not in df.columns and 'close' in df.columns:
            df['Close'] = df['close']
            
        prices = df['Close'].values
        
        # 1. Kalman
        kalman_est, residuals = self.calculate_kalman(prices)
        
        # 2. KHAOS (formerly Oscillation)
        khaos_indicator = self.calculate_khaos(residuals)
        
        # 3. Hurst (Rolling)
        hurst_series = np.zeros(len(prices))
        for i in range(50, len(prices)):
            window = prices[i-50:i]
            hurst_series[i] = self.calculate_hurst(window)
            
        # 4. Attention (Volume/Volatility based)
        # Simple proxy: Volatility / Moving Average Volatility
        returns = np.diff(prices, prepend=prices[0])
        vol = pd.Series(returns).rolling(20).std().fillna(0).values
        attention = vol / (pd.Series(vol).rolling(100).mean().fillna(1).values + 1e-6)
        
        # Assemble DataFrame
        result = pd.DataFrame({
            'Price': prices,
            'Kalman': kalman_est,
            'Residuals': residuals,
            'KHAOS': khaos_indicator,
            'Hurst': hurst_series,
            'Attention': attention
        }, index=df.index)
        
        return result
