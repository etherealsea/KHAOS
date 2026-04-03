---
name: "khaos_expert"
description: "Expert knowledge base for the KHAOS system (Physics-Informed Neural-Kalman Filter). Invoke when user asks about KHAOS logic, architecture, physical meaning, or trading signals."
---

# KHAOS System Expert Knowledge Base

## 1. System Definition
KHAOS is a **Physics-Informed Neural-Kalman System (PI-KAN)** designed for financial market analysis. It treats price action not as geometric patterns, but as a physical system governed by energy, momentum, and entropy.

## 2. Five-Layer Architecture

### Layer 1: Sensing Layer (The Senses)
Perceives the physical state of the market using non-linear features:
- **Hurst Exponent (H)**: Measures Fractal Dimension/Roughness.
  - $H > 0.5$: Persistent (Trend).
  - $H < 0.5$: Anti-Persistent (Mean Reversion).
- **Permutation Entropy (PE)**: Measures Information Density/Disorder. High entropy often precedes phase transitions.
- **Volatility ($\sigma$)**: Measures System Temperature/Energy level.
- **Bias/Potential**: Deviation from the gravitational center (EMA20).
- **Interaction Terms**: Non-linear couplings (e.g., Volatility * Hurst) used by PI-KAN.

### Layer 2: PI-KAN Layer (The Brain)
A **Kolmogorov-Arnold Network (KAN)** optimized via evolutionary algorithms (Differential Evolution).
- **Function**: Maps Sensing Layer features to physical control parameters.
- **Outputs**:
  - **$Q$ (Process Noise)**: System flexibility. High $Q$ = High adaptivity to trend changes.
  - **$R$ (Measurement Noise)**: Noise filtering. High $R$ = Strong filtering (ignoring price fluctuations).
  - **$M$ (Restoring Force/Gravity)**: Predicted structural reversal force.

### Layer 3: Physical Core (The Heart)
An **Extended Kalman Filter (EKF)** that performs dynamic state estimation.
- **State Vector**: $x = [x_p, x_v]^T$
  - **$x_p$ (Fair Value)**: The "True" price stripped of noise.
  - **$x_v$ (Velocity)**: The intrinsic speed/momentum of the trend.
- **Dynamics**: Adapts continuously based on $Q$ and $R$ from the PI-KAN layer.

### Layer 4: Signal Generation (The Decision)
Uses **Dual Physics Confirmation** to filter false signals.
- **Trigger Condition**:
  1.  **Structural Stress**: Gravity Force ($M$) > Threshold (e.g., 1.65 $\sigma$).
  2.  **Kinetic Confirmation**: Kinetic Energy (EKF Innovation Z-Score) must confirm the direction.
- **Logic**: A signal is valid ONLY if the structural tension is high AND the immediate kinetic energy confirms the reversal.

### Layer 5: Output Capabilities (The Dashboard)

| Component | Physical Meaning | Trading Significance |
| :--- | :--- | :--- |
| **Main Line** | **EKF Deviation** ($Price - x_p$) | **Potential Energy**. Represents the stress on price. High values indicate overextension. |
| **Fair Value ($x_p$)** | **Gravitational Center** | **Dynamic S/R**. The true value area where price should theoretically be. |
| **Velocity ($x_v$)** | **Intrinsic Momentum** | **Trend Health**. Divergence between Price and $x_v$ predicts exhaustion before price reverses. |
| **Noise Level ($R$)** | **Visibility/Fog** | **Confidence Score**. High $R$ means high chaos; technical patterns are unreliable. |
| **Histogram** | **Kinetic Energy** | **Shock/Surprise**. The immediate force driving price away from fair value. |
| **Phase State** | **Regime (Hurst/Entropy)** | **Strategy Selector**. Classifies market into Trend, Mean Reversion, or Chaos. |

## 3. Integration with ICT/PA
KHAOS acts as the "Physics Engine" underlying geometric Price Action (PA) or ICT concepts.
- **Order Blocks (OB)**: Valid only if KHAOS Potential Energy is high (Repulsion).
- **Fair Value Gaps (FVG)**: Represent Kinetic Energy surges.
- **Liquidity Sweeps**: Often coincide with Phase Transitions ($R > Q$).
- **Equilibrium**: Corresponds to Zero Potential Energy state.

## 4. Key Formulas
- **Main Line**: $(LogPrice - x_p) \times ScalingFactor$
- **Kinetic Energy**: $Z_{score} = \frac{Price - PredictedPrice}{\sqrt{S}}$ where $S = HPH^T + R$
