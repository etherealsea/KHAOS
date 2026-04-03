---
name: market_sensing
description: Extract physical states (Potential/Kinetic Energy) and regime information (Hurst/Entropy) from market data.
version: 1.0.0
dependencies: 
  - python: ["numpy", "pandas", "scipy"]
  - pine_script: ["KHAOS.pine"]
physics_constraints:
  - "Kinetic Energy cannot exceed Potential Energy change in closed system (without external force)"
  - "Entropy increases in diffusion phase (Hurst < 0.5)"
---

# Instructions

To perform the **Market Sensing** skill, follow these steps:

## 1. Regime Identification (Hurst)
Calculate the Hurst Exponent ($H$) to determine the current market phase:
- **Input**: Log-price series window (default $N=100$).
- **Logic**: 
  - If $H > 0.5$: Market is in **Trend Mode** (Persistent).
  - If $H < 0.5$: Market is in **Mean Reversion Mode** (Anti-persistent).
  - If $H \approx 0.5$: Market is in **Random Walk** (Brownian Motion).
- **Action**: Use $H$ to dynamically adjust the observation noise ($R$) in the EKF step.

## 2. State Separation (EKF)
Run the Extended Kalman Filter to separate price into Trend (Potential) and Momentum (Kinetic):
- **Dynamic Tuning**:
  - $R_t = R_{base} \cdot f(H_t)$
  - When $H \to 1$ (Strong Trend), decrease $R$ to trust observation more.
  - When $H \to 0$ (Mean Reversion), increase $R$ to filter noise.
- **Output**:
  - `x[0]`: Estimated Price (Trend/Potential Energy)
  - `x[1]`: Velocity (Kinetic Energy)
  - `x[2]`: Acceleration (Force)

## 3. Complexity Measurement (Entropy)
Calculate Permutation Entropy (PE) to validate the "validity" of the trend:
- **Threshold**: PE > 0.8 indicates high chaos (Low confidence).
- **Usage**: If PE is high during a breakout, it may be a "Fakeout" (False Break).

# Interface specification

## Input
```json
{
  "symbol": "ES1!",
  "timeframe": "1m",
  "data": [OHLCV_Array]
}
```

## Output
```json
{
  "regime": {
    "hurst": 0.65,
    "type": "TRENDING"
  },
  "physics_state": {
    "potential": 4500.0,
    "kinetic": 15.5,
    "acceleration": 0.2
  },
  "confidence": {
    "entropy": 0.45,
    "is_valid": true
  }
}
```
