---
name: "financial-model-expert"
description: "Expert in financial time series analysis, quantitative modeling, and PI-KAN/Physics-Informed ML. Invoke for tasks involving market data, quantitative strategies, or physics-informed neural networks."
---

# Financial Model Expert

This skill specializes in financial time series analysis, quantitative strategy development, and advanced machine learning models (PI-KAN, EKF).

## Capabilities

1.  **Time Series Analysis**:
    *   Stationarity tests (ADF, KPSS).
    *   Fractal analysis (Hurst Exponent, R/S analysis).
    *   Chaos theory (Lyapunov exponent, Entropy).
    *   Phase transition detection (Critical slowing down metrics).

2.  **Quantitative Strategy**:
    *   Backtesting frameworks (Backtrader, vn.py).
    *   Performance metrics (Sharpe, Sortino, Max Drawdown).
    *   Risk management (VaR, Position sizing).

3.  **PI-KAN & Neural Networks**:
    *   **KAN (Kolmogorov-Arnold Networks)**: Spline-based activation functions.
    *   **Physics-Informed**: Embedding SDEs (Stochastic Differential Equations) into loss functions.
    *   **EKF (Extended Kalman Filter)**: Differentiable filtering for state estimation.

## Python Stack

-   **Data**: `yfinance`, `akshare`, `pandas`, `numpy`.
-   **Modeling**: `torch` (PyTorch), `torchdiffeq`.
-   **Math**: `scipy.stats`, `sklearn`.

## Usage Guidelines

-   **Phase Transition**: When analyzing "crashes" or "regime changes", suggest computing Permutation Entropy or Local Hurst Exponent.
-   **KAN Implementation**: Use B-splines for activation. Ensure interpretability of weights.
-   **Data Handling**: Always check for `NaN` and handle outliers in financial data (e.g., winsorization).

## Example: Hurst Exponent

```python
import numpy as np

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
```
