# KHAOS Mathematical Model Reference
> **Version**: 2.0 (PI-KAN Evolution)
> **Status**: Active / Production
> **Date**: 2026-01-24

This document details the mathematical framework of the KHAOS PI-KAN (Physics-Informed Kolmogorov-Arnold Network) system.

---

## 1. System Architecture
KHAOS is a hybrid system combining **Statistical Physics** (EKF) with **Machine Learning** (KAN). Instead of using fixed physical constants, the system learns to dynamically adjust its physical parameters ($Q$, $R$, $M$) based on real-time market conditions.

### 1.1 The Core Equation (EKF)
The system models Price ($p_t$) and Velocity ($v_t$) using an Extended Kalman Filter:

$$
\begin{cases}
p_t = p_{t-1} + v_{t-1} \\
v_t = v_{t-1} \cdot \rho(H_t)
\end{cases}
$$

Where $\rho(H_t)$ is the **Fractal Damping Factor** derived from the Hurst Exponent:
$$ \rho(H) = 0.5 + \frac{0.5}{1 + e^{-10(H - 0.5)}} $$
*   $H \approx 1 \implies \rho \approx 1$ (Trend/Inertia)
*   $H \approx 0 \implies \rho \approx 0.5$ (Mean Reversion/Friction)

### 1.2 PI-KAN Parameter Modulation
Unlike standard EKF which uses static noise covariance, KHAOS modulates Process Noise ($Q$), Measurement Noise ($R$), and Restoring Force ($M$) via a learned linear mapping (Simplified KAN Layer).

The system observes 5 Normalized Features ($\mathbf{f} \in \mathbb{R}^5$):
1.  **n_hurst**: Normalized Hurst Exponent (Structure)
2.  **n_vol**: Normalized Volatility (Energy)
3.  **n_bias**: Normalized Deviation from EMA20 (Potential)
4.  **n_slope**: Normalized Momentum (Kinetic)
5.  **n_pe**: Normalized Permutation Entropy (Information)

**The Modulation Equations:**
$$ Q_t = \text{ReLU}(\mathbf{w}_Q \cdot \mathbf{f}_t + b_Q) \cdot \alpha $$
$$ R_t = \text{ReLU}(\mathbf{w}_R \cdot \mathbf{f}_t + b_R) \cdot \alpha $$
$$ M_t = \mathbf{w}_M \cdot \mathbf{f}_t + b_M $$

*Weights ($\mathbf{w}$) are optimized via Evolutionary Algorithms (Differential Evolution) to maximize Reversal Detection Profitability.*

---

## 2. Signal Generation Logic

### 2.1 Gravity Force (Restoring Force)
The raw restoring force $M_t$ represents the market's "desire" to revert to the mean.
1.  **Adaptive Normalization**: $M_t$ is standardized (Z-Score) using a dynamic window adapted to the Hurst Exponent.
2.  **Activation**: $F_{gravity} = \tanh(Z_{M})$
3.  **Smoothing**: WMA(5) is applied to form continuous zones.

### 2.2 Phase Transition Zones (Signals)
KHAOS detects **Phase Transitions** (Reversals) using a Schmitt Trigger (Latch) mechanism with Physics Gating.

**Trigger Conditions (Entry):**
1.  **Force Threshold**: $|F_{gravity}| > 1.65 \sigma$ (High Confidence)
2.  **Regime Validity (Gating)**:
    *   **Volatility Gate**: $Z_{vol} > -1.07$ (Market must be active, not dead)
    *   **Structure Gate**: $H < 0.52$ (Trend must be weak/breaking)

**Exit Conditions (Relaxation):**
*   **Energy Dissipation**: $|F_{gravity}| < 1.0 \sigma$

This creates "Reversal Zones" rather than single points, indicating periods of high probability for mean reversion.

---

## 3. Feature Definitions

| Feature | Description | Calculation |
| :--- | :--- | :--- |
| **Hurst Proxy** | Fractal Dimension / Trendiness | Linear Regression Slope of Log-Log Plot (Approx) |
| **Volatility** | Market Energy | StdDev of Log Returns (20 period) |
| **Bias** | Potential Energy | Distance from EMA20 |
| **Slope** | Kinetic Momentum | 5-period Rate of Change |
| **Entropy** | Information Content / Chaos | Permutation Entropy (Order 3) |

*All features are normalized using a Universal Tanh-Z Transform to $[-1, 1]$ before entering the KAN.*

---

## 4. Optimization & Synchronization
*   **Optimizer**: `scripts/pipeline/optimize_pikan_evolution.py`
*   **Replica**: `src/khaos/optimization/pine_replica.py`
*   **Production**: `src/khaos/pine/KHAOS.pine`

**Rule**: Any logic change must be implemented and verified in the Python Replica first, then ported to Pine. Parameter defaults in Python must match Pine `input` defaults.
