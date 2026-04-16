import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from khaos.核心引擎.physics import compute_physics_features_bulk, PHYSICS_FEATURE_NAMES

LOCAL_PHYSICS_WINDOW = 20
EKF_ALPHA = 0.1
BREAKOUT_EVENT_THRESHOLD = 0.8
REVERSION_EVENT_THRESHOLD = 0.8

def compute_ekf_track(log_close, alpha=EKF_ALPHA):
    ekf_track = np.empty_like(log_close, dtype=np.float32)
    ekf_track[0] = log_close[0]
    for idx in range(1, len(log_close)):
        ekf_track[idx] = alpha * log_close[idx] + (1.0 - alpha) * ekf_track[idx - 1]
    return ekf_track

def ema_np(values, span):
    return pd.Series(values).ewm(span=span, adjust=False).mean().values.astype(np.float32)

def rolling_entropy_proxy_np(high, low, close, window=LOCAL_PHYSICS_WINDOW):
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - close_prev),
        np.abs(low - close_prev)
    ])
    sum_tr = pd.Series(tr).rolling(window=window, min_periods=1).sum().values
    highest_high = pd.Series(high).rolling(window=window, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=window, min_periods=1).min().values
    rng = highest_high - lowest_low + 1e-8
    entropy = np.log10((sum_tr / rng) + 1e-8)
    return np.nan_to_num(entropy, nan=0.0).astype(np.float32)

def rolling_hurst_proxy_np(log_close, window=LOCAL_PHYSICS_WINDOW):
    series = pd.Series(log_close)
    r = series.rolling(window=window, min_periods=2).max() - series.rolling(window=window, min_periods=2).min()
    s = series.rolling(window=window, min_periods=2).std()
    rs = (r / (s + 1e-8)).fillna(1.0).values
    expected = 0.5 * np.log(float(window))
    deviation = np.log(rs + 1e-8) - expected
    h = 0.5 + 0.3 * np.tanh((deviation * 20.0) / 10.0)
    return np.clip(np.nan_to_num(h, nan=0.5), 0.0, 1.0).astype(np.float32)

def rolling_mle_proxy_np(returns, window=LOCAL_PHYSICS_WINDOW):
    prev = np.roll(returns, 1)
    prev[0] = returns[0]
    local_div = np.log((np.abs(returns) + 1e-8) / (np.abs(prev) + 1e-8))
    return ema_np(local_div, window).astype(np.float32)

def build_future_path(log_values, horizon):
    future_moves = []
    for step in range(1, horizon + 1):
        shifted = np.roll(log_values, -step)
        move = shifted - log_values
        move[-step:] = 0.0
        future_moves.append(move)
    return np.stack(future_moves, axis=1).astype(np.float32)


def _valid_prefix(values, forecast_horizon):
    values = np.asarray(values, dtype=np.float32)
    if forecast_horizon <= 0 or len(values) <= forecast_horizon:
        return values
    return values[:-forecast_horizon]


def _robust_quantile(values, q, floor=0.0):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(floor)
    return float(max(np.quantile(finite, q), floor))


def _tail_zero(*arrays, forecast_horizon):
    if forecast_horizon <= 0:
        return arrays
    for array in arrays:
        array[-forecast_horizon:] = 0.0
    return arrays

def get_breakout_event_config(forecast_horizon):
    horizon_ratio = np.clip((float(forecast_horizon) - 4.0) / 6.0, 0.0, 1.0)
    return {
        'target_threshold': BREAKOUT_EVENT_THRESHOLD,
        'path_efficiency_min': 0.42 - 0.05 * horizon_ratio,
        'directional_efficiency_min': 0.38 - 0.03 * horizon_ratio,
        'continuation_release_min': 0.20 + 0.08 * horizon_ratio,
        'adverse_penalty_max': 1.10 + 0.12 * horizon_ratio,
        'hard_negative_context_min': 0.35,
        'hard_negative_path_max': 0.38 - 0.04 * horizon_ratio,
        'hard_negative_directional_max': 0.38 - 0.02 * horizon_ratio,
        'hard_negative_continuation_max': 0.15 + 0.05 * horizon_ratio,
        'hard_negative_adverse_min': 1.10 + 0.06 * horizon_ratio,
        'hard_negative_target_max': 0.20 + 0.06 * horizon_ratio
    }

def get_reversion_event_config(forecast_horizon):
    horizon_ratio = np.clip((float(forecast_horizon) - 4.0) / 6.0, 0.0, 1.0)
    return {
        'target_threshold': REVERSION_EVENT_THRESHOLD - 0.13 * horizon_ratio,
        'reversal_quality_min': 0.30 - 0.06 * horizon_ratio,
        'terminal_confirmation_min': 0.18 - 0.06 * horizon_ratio,
        'imbalance_alignment_min': 0.40 - 0.04 * horizon_ratio,
        'entropy_rise_min': -0.01 * horizon_ratio,
        'hard_negative_strength_min': 0.80 - 0.10 * horizon_ratio,
        'hard_negative_target_max': 0.20 + 0.06 * horizon_ratio,
        'hard_negative_terminal_max': 0.10 + 0.05 * horizon_ratio,
        'hard_negative_reversal_max': 0.20 + 0.06 * horizon_ratio,
        'hard_negative_continuation_ratio': 1.00 - 0.05 * horizon_ratio
    }

def build_breakout_targets(log_close, log_high, log_low, returns, sigma, entropy, forecast_horizon):
    event_cfg = get_breakout_event_config(forecast_horizon)
    future_vol = pd.Series(returns).rolling(window=forecast_horizon).std().shift(-forecast_horizon).values
    future_vol = np.nan_to_num(future_vol, nan=0.0).astype(np.float32)
    future_path = build_future_path(log_close, forecast_horizon)
    future_high_path = np.stack([
        np.roll(log_high, -step) for step in range(1, forecast_horizon + 1)
    ], axis=1).astype(np.float32)
    future_low_path = np.stack([
        np.roll(log_low, -step) for step in range(1, forecast_horizon + 1)
    ], axis=1).astype(np.float32)
    for step in range(1, forecast_horizon + 1):
        future_high_path[-step:, step - 1] = log_high[-step:]
        future_low_path[-step:, step - 1] = log_low[-step:]
    future_abs_move = np.max(np.abs(future_path), axis=1)
    future_range = np.max(future_high_path, axis=1) - np.min(future_low_path, axis=1)
    net_displacement = np.abs(future_path[:, -1])
    path_efficiency = net_displacement / (future_abs_move + 1e-8)
    max_adverse_excursion = np.minimum(future_path.min(axis=1), 0.0)
    max_favorable_excursion = np.maximum(future_path.max(axis=1), 0.0)
    directional_efficiency = np.maximum(max_favorable_excursion, -max_adverse_excursion) / (future_range + 1e-8)
    adverse_penalty = np.maximum(0.0, (future_abs_move - net_displacement) / (sigma + 1e-8))
    vol_expansion = np.clip(np.log((future_vol + 1e-8) / (sigma + 1e-8)), -5.0, 5.0)
    range_release = np.maximum(0.0, future_range / (sigma + 1e-8) - 1.0)
    displacement_release = np.maximum(0.0, future_abs_move / (sigma + 1e-8) - 1.0)
    continuation_release = np.maximum(0.0, net_displacement / (sigma + 1e-8) - 0.5)
    breakout_target = np.maximum(
        0.0,
        0.35 * np.maximum(vol_expansion, 0.0) +
        0.20 * range_release +
        0.15 * displacement_release +
        0.20 * continuation_release +
        0.20 * np.maximum(0.0, path_efficiency + directional_efficiency - 0.8) -
        0.10 * adverse_penalty
    ).astype(np.float32)
    sigma_ref = ema_np(sigma, LOCAL_PHYSICS_WINDOW)
    entropy_ref = ema_np(entropy, LOCAL_PHYSICS_WINDOW)
    compression = np.maximum(0.0, (sigma_ref - sigma) / (sigma_ref + 1e-8))
    entropy_quiet = np.maximum(0.0, entropy_ref - entropy)
    future_entropy_mean = pd.Series(entropy).rolling(window=forecast_horizon, min_periods=1).mean().shift(-forecast_horizon).ffill().fillna(entropy[-1]).values.astype(np.float32)
    entropy_rise = np.maximum(0.0, future_entropy_mean - entropy)
    breakout_aux = breakout_target * (0.45 + 0.35 * compression + 0.20 * entropy_quiet) * (1.0 + 0.5 * entropy_rise)
    breakout_event = (
        (breakout_target >= event_cfg['target_threshold']) &
        (path_efficiency >= event_cfg['path_efficiency_min']) &
        (directional_efficiency >= event_cfg['directional_efficiency_min']) &
        (continuation_release >= event_cfg['continuation_release_min']) &
        (adverse_penalty <= event_cfg['adverse_penalty_max'])
    ).astype(np.float32)
    breakout_hard_negative = (
        ((compression + entropy_quiet) > event_cfg['hard_negative_context_min']) &
        (
            (path_efficiency < event_cfg['hard_negative_path_max']) |
            (directional_efficiency < event_cfg['hard_negative_directional_max']) |
            (continuation_release < event_cfg['hard_negative_continuation_max']) |
            (adverse_penalty > event_cfg['hard_negative_adverse_min']) |
            (breakout_target < event_cfg['hard_negative_target_max'])
        )
    ).astype(np.float32)
    if forecast_horizon > 0:
        breakout_target[-forecast_horizon:] = 0.0
        breakout_aux[-forecast_horizon:] = 0.0
        breakout_event[-forecast_horizon:] = 0.0
        breakout_hard_negative[-forecast_horizon:] = 0.0
    return np.clip(breakout_target, 0.0, 10.0), np.clip(breakout_aux, 0.0, 10.0).astype(np.float32), breakout_event, breakout_hard_negative

def build_reversion_targets(log_close, ema20, sigma, entropy, hurst, forecast_horizon):
    event_cfg = get_reversion_event_config(forecast_horizon)
    log_ema20 = np.log(np.maximum(ema20, 1e-8))
    ekf_track = compute_ekf_track(log_close)
    ekf_residual = log_close - ekf_track
    ema_log_gap = log_close - log_ema20
    res_score = ekf_residual / (sigma + 1e-8)
    ema_score = ema_log_gap / (sigma + 1e-8)
    imbalance_direction = np.sign(res_score + ema_score).astype(np.float32)
    imbalance_alignment = np.abs(res_score + ema_score) / (np.abs(res_score) + np.abs(ema_score) + 1e-6)
    imbalance_strength = np.maximum(0.0, np.abs(res_score) + np.abs(ema_score) - 1.0)
    future_path = build_future_path(log_close, forecast_horizon)
    signed_future = future_path * imbalance_direction[:, None]
    continuation_excursion = np.maximum(signed_future, 0.0).max(axis=1) / (sigma + 1e-8)
    reversal_excursion = np.maximum(-signed_future, 0.0).max(axis=1) / (sigma + 1e-8)
    forward_return = future_path[:, -1]
    terminal_confirmation = np.maximum(0.0, -(imbalance_direction * forward_return) / (sigma + 1e-8))
    reversal_quality = np.maximum(0.0, reversal_excursion - 0.45 * continuation_excursion)
    entropy_ref = ema_np(entropy, LOCAL_PHYSICS_WINDOW)
    low_entropy_consensus = np.maximum(0.0, entropy_ref - entropy)
    future_entropy_mean = pd.Series(entropy).rolling(window=forecast_horizon, min_periods=1).mean().shift(-forecast_horizon).ffill().fillna(entropy[-1]).values.astype(np.float32)
    entropy_rise = np.maximum(0.0, future_entropy_mean - entropy)
    trend_consensus = np.maximum(0.0, hurst - 0.55)
    fragmentation_bonus = low_entropy_consensus * entropy_rise * (0.5 + trend_consensus)
    target = imbalance_strength * imbalance_alignment * (
        0.55 * reversal_quality +
        0.25 * terminal_confirmation +
        0.20 * fragmentation_bonus
    )
    aux_target = target * (0.5 * low_entropy_consensus + 0.5 * entropy_rise) * (1.0 + trend_consensus)
    reversion_event = (
        (target >= event_cfg['target_threshold']) &
        (reversal_quality >= event_cfg['reversal_quality_min']) &
        (terminal_confirmation >= event_cfg['terminal_confirmation_min']) &
        (imbalance_alignment >= event_cfg['imbalance_alignment_min']) &
        (entropy_rise >= event_cfg['entropy_rise_min'])
    ).astype(np.float32)
    reversion_hard_negative = (
        (imbalance_strength > event_cfg['hard_negative_strength_min']) &
        (
            (target < event_cfg['hard_negative_target_max']) |
            (terminal_confirmation < event_cfg['hard_negative_terminal_max']) |
            (reversal_quality < event_cfg['hard_negative_reversal_max'])
        ) &
        (continuation_excursion >= reversal_excursion * event_cfg['hard_negative_continuation_ratio'])
    ).astype(np.float32)
    if forecast_horizon > 0:
        target[-forecast_horizon:] = 0.0
        aux_target[-forecast_horizon:] = 0.0
        reversion_event[-forecast_horizon:] = 0.0
        reversion_hard_negative[-forecast_horizon:] = 0.0
    return np.clip(target, 0.0, 10.0).astype(np.float32), np.clip(aux_target, 0.0, 10.0).astype(np.float32), reversion_event, reversion_hard_negative


def _compute_breakout_discovery_components(log_close, log_high, log_low, returns, sigma, entropy, forecast_horizon):
    """
    Iter11: Pure MFE/MAE breakout target logic
    Direction d_t is estimated using a 10-period rolling mean of returns.
    """
    future_path = build_future_path(log_close, forecast_horizon)
    sigma_safe = sigma + 1e-8
    
    # 1. Define current direction
    rolling_ret = pd.Series(returns).rolling(window=10, min_periods=1).mean().values
    d_t = np.sign(rolling_ret).astype(np.float32)
    # Default to +1 if 0
    d_t[d_t == 0] = 1.0

    # 2. MFE & MAE relative to d_t
    signed_future = future_path * d_t[:, None]
    mfe = np.maximum(signed_future, 0.0).max(axis=1)
    mae = np.maximum(-signed_future, 0.0).max(axis=1)
    
    mfe_norm = mfe / sigma_safe
    mae_norm = mae / sigma_safe
    
    # 3. Score calculation (MFE_norm * exp(-λ * MAE_norm))
    # Lambda = 1.2 penalizes drawdowns heavily
    score = mfe_norm * np.exp(-1.2 * mae_norm)
    aux_score = np.maximum(mfe_norm - 0.8 * mae_norm, 0.0)

    return {
        'mfe_norm': mfe_norm.astype(np.float32),
        'mae_norm': mae_norm.astype(np.float32),
        'target_default': score.astype(np.float32),
        'aux_default': aux_score.astype(np.float32),
        'target_guarded_v1': score.astype(np.float32),
        'aux_guarded_v1': aux_score.astype(np.float32),
    }


def fit_breakout_discovery_thresholds(log_close, log_high, log_low, returns, sigma, entropy, forecast_horizon, preset='default'):
    components = _compute_breakout_discovery_components(
        log_close,
        log_high,
        log_low,
        returns,
        sigma,
        entropy,
        forecast_horizon,
    )
    target_key = 'target_guarded_v1' if preset == 'guarded_v1' else 'target_default'
    valid_target = _valid_prefix(components[target_key], forecast_horizon)
    valid_mae = _valid_prefix(components['mae_norm'], forecast_horizon)
    
    # We use 0.95 (top 5%) for event threshold as requested by user
    return {
        'event_target_threshold': _robust_quantile(valid_target, 0.95, floor=0.5),
        'hard_negative_target_threshold': _robust_quantile(valid_target, 0.30, floor=0.1),
        'hard_negative_mae_threshold': _robust_quantile(valid_mae, 0.85, floor=1.5),
    }


def build_breakout_discovery_targets(
    log_close,
    log_high,
    log_low,
    returns,
    sigma,
    entropy,
    forecast_horizon,
    threshold_config=None,
    preset='default',
):
    components = _compute_breakout_discovery_components(
        log_close,
        log_high,
        log_low,
        returns,
        sigma,
        entropy,
        forecast_horizon,
    )
    target_key = 'target_guarded_v1' if preset == 'guarded_v1' else 'target_default'
    aux_key = 'aux_guarded_v1' if preset == 'guarded_v1' else 'aux_default'
    breakout_target = components[target_key].copy()
    breakout_aux = components[aux_key].copy()
    mae_norm = components['mae_norm']

    threshold_cfg = dict(
        threshold_config or fit_breakout_discovery_thresholds(
            log_close,
            log_high,
            log_low,
            returns,
            sigma,
            entropy,
            forecast_horizon,
            preset=preset,
        )
    )
    event_target_threshold = float(threshold_cfg['event_target_threshold'])
    hn_target_threshold = float(threshold_cfg['hard_negative_target_threshold'])
    hn_mae_threshold = float(threshold_cfg['hard_negative_mae_threshold'])

    breakout_event = (breakout_target >= event_target_threshold).astype(np.float32)
    breakout_hard_negative = (
        (mae_norm >= hn_mae_threshold) & 
        (breakout_target <= hn_target_threshold)
    ).astype(np.float32)

    _tail_zero(
        breakout_target,
        breakout_aux,
        breakout_event,
        breakout_hard_negative,
        forecast_horizon=forecast_horizon,
    )
    return (
        np.clip(breakout_target, 0.0, 10.0).astype(np.float32),
        np.clip(breakout_aux, 0.0, 10.0).astype(np.float32),
        breakout_event,
        breakout_hard_negative,
    )


def _compute_reversion_discovery_components(log_close, ema20, sigma, entropy, hurst, forecast_horizon):
    """
    Iter11: Pure MFE/MAE reversion target logic
    Direction -d_t is used to assess reversion quality (opposite of recent trend).
    """
    future_path = build_future_path(log_close, forecast_horizon)
    sigma_safe = sigma + 1e-8
    
    # 1. Define recent direction and use its opposite for reversion
    returns = np.diff(log_close, prepend=log_close[0])
    rolling_ret = pd.Series(returns).rolling(window=10, min_periods=1).mean().values
    d_t = np.sign(rolling_ret).astype(np.float32)
    d_t[d_t == 0] = 1.0
    rev_dir = -d_t

    # 2. MFE & MAE relative to rev_dir
    signed_future = future_path * rev_dir[:, None]
    mfe = np.maximum(signed_future, 0.0).max(axis=1)
    mae = np.maximum(-signed_future, 0.0).max(axis=1)
    
    mfe_norm = mfe / sigma_safe
    mae_norm = mae / sigma_safe
    
    # 3. Score calculation (MFE_norm * exp(-λ * MAE_norm))
    score = mfe_norm * np.exp(-1.2 * mae_norm)
    aux_score = np.maximum(mfe_norm - 0.8 * mae_norm, 0.0)

    return {
        'mfe_norm': mfe_norm.astype(np.float32),
        'mae_norm': mae_norm.astype(np.float32),
        'target_default': score.astype(np.float32),
        'aux_default': aux_score.astype(np.float32),
        'target_guarded_v1': score.astype(np.float32),
        'aux_guarded_v1': aux_score.astype(np.float32),
    }


def fit_reversion_discovery_thresholds(log_close, ema20, sigma, entropy, hurst, forecast_horizon, preset='default'):
    components = _compute_reversion_discovery_components(
        log_close,
        ema20,
        sigma,
        entropy,
        hurst,
        forecast_horizon,
    )
    target_key = 'target_guarded_v1' if preset == 'guarded_v1' else 'target_default'
    valid_target = _valid_prefix(components[target_key], forecast_horizon)
    valid_mae = _valid_prefix(components['mae_norm'], forecast_horizon)
    
    # Use 0.95 (top 5%) for high-precision event threshold
    return {
        'event_target_threshold': _robust_quantile(valid_target, 0.95, floor=0.5),
        'hard_negative_target_threshold': _robust_quantile(valid_target, 0.30, floor=0.1),
        'hard_negative_mae_threshold': _robust_quantile(valid_mae, 0.85, floor=1.5),
    }


def build_reversion_discovery_targets(
    log_close,
    ema20,
    sigma,
    entropy,
    hurst,
    forecast_horizon,
    threshold_config=None,
    preset='default',
):
    components = _compute_reversion_discovery_components(
        log_close,
        ema20,
        sigma,
        entropy,
        hurst,
        forecast_horizon,
    )
    target_key = 'target_guarded_v1' if preset == 'guarded_v1' else 'target_default'
    aux_key = 'aux_guarded_v1' if preset == 'guarded_v1' else 'aux_default'
    target = components[target_key].copy()
    aux_target = components[aux_key].copy()
    mae_norm = components['mae_norm']

    threshold_cfg = dict(
        threshold_config or fit_reversion_discovery_thresholds(
            log_close,
            ema20,
            sigma,
            entropy,
            hurst,
            forecast_horizon,
            preset=preset,
        )
    )
    event_target_threshold = float(threshold_cfg['event_target_threshold'])
    hn_target_threshold = float(threshold_cfg['hard_negative_target_threshold'])
    hn_mae_threshold = float(threshold_cfg['hard_negative_mae_threshold'])

    reversion_event = (target >= event_target_threshold).astype(np.float32)
    reversion_hard_negative = (
        (mae_norm >= hn_mae_threshold) & 
        (target <= hn_target_threshold)
    ).astype(np.float32)

    _tail_zero(
        target,
        aux_target,
        reversion_event,
        reversion_hard_negative,
        forecast_horizon=forecast_horizon,
    )
    return (
        np.clip(target, 0.0, 10.0).astype(np.float32),
        np.clip(aux_target, 0.0, 10.0).astype(np.float32),
        reversion_event,
        reversion_hard_negative,
    )

class FinancialDataset(Dataset):
    def __init__(self, df, window_size=20, forecast_horizon=4, volatility_window=LOCAL_PHYSICS_WINDOW, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.feature_names = PHYSICS_FEATURE_NAMES
        df.columns = [c.lower().strip() for c in df.columns]
        self.time = df['time'].values if 'time' in df.columns else None
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        self.close = df['close'].values.astype(np.float32)
        self.open = df['open'].values.astype(np.float32)
        self.high = df['high'].values.astype(np.float32)
        self.low = df['low'].values.astype(np.float32)
        self.volume = df['volume'].values.astype(np.float32)
        log_close = np.log(np.maximum(self.close, 1e-8))
        log_high = np.log(np.maximum(self.high, 1e-8))
        log_low = np.log(np.maximum(self.low, 1e-8))
        returns = np.diff(log_close, prepend=log_close[0])
        sigma = pd.Series(returns).rolling(window=volatility_window, min_periods=1).std()
        sigma = sigma.ffill().fillna(0.0).values
        self.sigma = np.maximum(sigma, 1e-6).astype(np.float32)
        self.ema20 = ema_np(self.close, 20)
        entropy = rolling_entropy_proxy_np(self.high, self.low, self.close, LOCAL_PHYSICS_WINDOW)
        hurst = rolling_hurst_proxy_np(log_close, LOCAL_PHYSICS_WINDOW)
        breakout_target, breakout_aux, breakout_event, breakout_hard_negative = build_breakout_targets(
            log_close,
            log_high,
            log_low,
            returns,
            self.sigma,
            entropy,
            self.forecast_horizon
        )
        reversion_target, reversion_aux, reversion_event, reversion_hard_negative = build_reversion_targets(
            log_close,
            self.ema20,
            self.sigma,
            entropy,
            hurst,
            self.forecast_horizon
        )
        self.targets = np.stack([breakout_target, reversion_target], axis=1).astype(np.float32)
        self.aux_targets = np.stack([breakout_aux, reversion_aux], axis=1).astype(np.float32)
        self.event_flags = np.stack([
            breakout_event,
            reversion_event,
            breakout_hard_negative,
            reversion_hard_negative
        ], axis=1).astype(np.float32)
        self.sample_weights = (
            1.0 +
            1.40 * breakout_target +
            1.20 * reversion_target +
            0.70 * breakout_aux +
            0.55 * reversion_aux +
            0.65 * breakout_event +
            0.75 * reversion_event +
            0.65 * breakout_hard_negative +
            0.55 * reversion_hard_negative
        ).astype(np.float32)
        self.data = np.stack([
            self.open, self.high, self.low, self.close, self.volume, self.ema20
        ], axis=1)
        print(f"  [Pre-computation] Extracting physics features for {len(df)} rows...")
        raw_tensor = torch.tensor(self.data, dtype=torch.float32)
        self.physics_features = compute_physics_features_bulk(raw_tensor, device='cpu')
        print(f"  [Pre-computation] Done. Feature shape: {self.physics_features.shape}")

    def __len__(self):
        return max(0, len(self.close) - self.window_size - self.forecast_horizon)

    def __getitem__(self, idx):
        end_idx = idx + self.window_size
        feat_window = self.physics_features[idx:end_idx]
        y = self.targets[end_idx - 1]
        aux_y = self.aux_targets[end_idx - 1]
        sigma = self.sigma[end_idx - 1]
        weight = self.sample_weights[end_idx - 1]
        flags = self.event_flags[end_idx - 1]
        return (
            feat_window,
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(aux_y, dtype=torch.float32),
            torch.tensor(sigma, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(flags, dtype=torch.float32)
        )

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

def create_rolling_datasets(file_path, train_ratio=0.8, window_size=20, horizon=4, sample_per_year_months=6, subsample=True, fast_full=False):
    print(f"Loading {file_path}...")
    df = load_data(file_path)
    
    # NO TRUNCATION - Load the full ~10 years history to prevent information loss
    if fast_full:
        df = df.iloc[-20000:] # Limit to 20k rows for faster full-pipeline execution
        print(f"  [Fast Mode] Loaded truncated history: {len(df)} rows")
    else:
        print(f"  Loaded full history: {len(df)} rows")
        
    total_len = len(df)
    train_size = int(total_len * train_ratio)
    
    train_df = df.iloc[:train_size]
    # Test set needs overlap to start predicting immediately after train set
    test_df = df.iloc[train_size - window_size:] 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = FinancialDataset(train_df, window_size, horizon, device=device)
    test_ds = FinancialDataset(test_df, window_size, horizon, device=device)
    return train_ds, test_ds
