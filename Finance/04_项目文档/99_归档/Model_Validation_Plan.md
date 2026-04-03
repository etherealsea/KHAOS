# KHAOS Mathematical Model Validation Plan

This document outlines the process for testing, validating, and refining the "Fractal-State-Space" mathematical model before its implementation in Pine Script.

## 1. Objectives
- **Verify Predictive Power**: Assess if the Adaptive Kalman Filter correctly estimates the "fair value" state.
- **Validate Regime Detection**: Confirm if the Generalized Hurst Exponent correctly identifies Trending vs. Mean Reverting states.
- **Optimize Parameters**: Find optimal values for $Q$, $R$, and attention weights using historical data.

## 2. Data Acquisition
We will use real historical market data (Crypto) to avoid simulation bias.
- **Source**: Binance / OKX Public API (Spot Market).
- **Pairs**: BTC/USDT, ETH/USDT, SOL/USDT (representing High, Medium, and High-Beta Caps).
- **Timeframes**: 
  - 1 Hour (H1): For core trend validation.
  - 15 Minute (M15): For intraday noise testing.
  - 1 Day (D1): For long-term structural stability.
- **Storage**: `model_research/data/*.csv` (Format: Timestamp, Open, High, Low, Close, Volume).

## 3. Mathematical Model Implementation (Python Prototype)
We will build a Python prototype in `model_research/` mimicking the logic defined in `Mathematical_Model_Design.md`.

### Core Components to Test:
1.  **Adaptive Kalman Filter (AKF)**:
    -   Implement the scalar Kalman Filter update equations.
    -   Implement the adaptive $Q$ (Process Noise) based on volatility shock.
    -   **Metric**: Mean Squared Error (MSE) of State Estimate vs. Next Close Price.

2.  **Generalized Hurst Exponent (GHE)**:
    -   Implement `bm_mix` (Ensemble Hurst) or standard R/S analysis.
    -   **Metric**: Correlation between Hurst Exponent and subsequent Absolute Returns (Trend Strength).

3.  **Attention-Based Weighting**:
    -   Implement the Softmax weighting mechanism combining Hurst, $R^2$, and Volatility.
    -   **Metric**: Improvement in Sharpe Ratio of a simple strategy using these weights vs. fixed weights.

## 4. Testing Procedure

### Phase 1: In-Sample Calibration (Training)
-   Use 70% of historical data.
-   Optimize parameters ($Q_{base}$, $R_{base}$, $\lambda$) to minimize state estimation error.

### Phase 2: Out-of-Sample Validation (Testing)
-   Use remaining 30% of data.
-   Run the model without parameter adjustment.
-   **Pass Criteria**:
    -   Residuals (Price - State) should be normally distributed (check Kurtosis/Skew).
    -   Hurst Exponent > 0.6 should correlate with positive serial correlation in returns.

### Phase 3: Stress Testing
-   Test on high volatility periods (e.g., May 2021 crash, Nov 2022 FTX crash).
-   Verify if the "Structural Potential Field" correctly widens bands/supports during shocks.

## 5. Output Artifacts
-   `model_research/results/performance_metrics.json`: Quantitative results.
-   `model_research/results/residual_plots.png`: Visual verification of model fit.
-   `model_research/optimized_params.json`: Final parameters for Pine Script.
