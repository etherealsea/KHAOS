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
    future_path = build_future_path(log_close, forecast_horizon)
    future_high_path = np.stack(
        [np.roll(log_high, -step) for step in range(1, forecast_horizon + 1)],
        axis=1,
    ).astype(np.float32)
    future_low_path = np.stack(
        [np.roll(log_low, -step) for step in range(1, forecast_horizon + 1)],
        axis=1,
    ).astype(np.float32)
    for step in range(1, forecast_horizon + 1):
        future_high_path[-step:, step - 1] = log_high[-step:]
        future_low_path[-step:, step - 1] = log_low[-step:]

    sigma_safe = sigma + 1e-8
    breakout_space = np.maximum(np.max(future_high_path, axis=1) - log_close, 0.0) / sigma_safe
    downside_excursion = np.maximum(log_close - np.min(future_low_path, axis=1), 0.0) / sigma_safe
    terminal_breakout = np.maximum(future_path[:, -1], 0.0) / sigma_safe
    net_displacement = np.maximum(future_path.max(axis=1), 0.0) / sigma_safe
    path_persistence = np.mean(future_path > 0.0, axis=1).astype(np.float32)
    path_efficiency = terminal_breakout / np.maximum(net_displacement, 1e-6)
    clean_space = np.maximum(breakout_space - 0.40 * downside_excursion - 0.25, 0.0)
    clean_terminal = np.maximum(terminal_breakout - 0.25 * downside_excursion, 0.0)

    future_vol = pd.Series(returns).rolling(window=forecast_horizon, min_periods=1).std().shift(-forecast_horizon).values
    future_vol = np.nan_to_num(future_vol, nan=0.0).astype(np.float32)
    vol_release = np.maximum(np.log((future_vol + 1e-8) / sigma_safe), 0.0)
    entropy_ref = ema_np(entropy, LOCAL_PHYSICS_WINDOW)
    entropy_quiet = np.maximum(entropy_ref - entropy, 0.0)
    future_entropy = pd.Series(entropy).rolling(window=forecast_horizon, min_periods=1).mean().shift(-forecast_horizon).values
    future_entropy = np.nan_to_num(future_entropy, nan=entropy[-1]).astype(np.float32)
    entropy_release = np.maximum(future_entropy - entropy, 0.0)
    quality_gate = np.clip(0.50 + 0.35 * np.clip(path_efficiency, 0.0, 1.5) + 0.15 * path_persistence, 0.0, 1.40)
    guarded_target = np.maximum(
        0.0,
        (
            0.44 * clean_space +
            0.28 * clean_terminal +
            0.12 * np.maximum(path_efficiency, 0.0) +
            0.08 * path_persistence +
            0.08 * vol_release
        ) * quality_gate -
        0.18 * downside_excursion
    ).astype(np.float32)
    guarded_aux = np.maximum(
        0.0,
        0.62 * clean_space +
        0.22 * breakout_space +
        0.10 * entropy_release +
        0.06 * entropy_quiet
    ).astype(np.float32)
    return {
        'breakout_space': breakout_space.astype(np.float32),
        'downside_excursion': downside_excursion.astype(np.float32),
        'terminal_breakout': terminal_breakout.astype(np.float32),
        'path_efficiency': path_efficiency.astype(np.float32),
        'path_persistence': path_persistence.astype(np.float32),
        'clean_space': clean_space.astype(np.float32),
        'clean_terminal': clean_terminal.astype(np.float32),
        'vol_release': vol_release.astype(np.float32),
        'entropy_release': entropy_release.astype(np.float32),
        'entropy_quiet': entropy_quiet.astype(np.float32),
        'target_default': np.maximum(
            0.0,
            0.38 * clean_space +
            0.28 * clean_terminal +
            0.16 * np.maximum(path_efficiency, 0.0) +
            0.10 * path_persistence +
            0.08 * vol_release
        ).astype(np.float32),
        'aux_default': np.maximum(
            0.0,
            0.55 * clean_space +
            0.20 * breakout_space +
            0.15 * entropy_release +
            0.10 * entropy_quiet
        ).astype(np.float32),
        'target_guarded_v1': guarded_target,
        'aux_guarded_v1': guarded_aux,
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
    valid_space = _valid_prefix(components['clean_space'], forecast_horizon)
    valid_downside = _valid_prefix(components['downside_excursion'], forecast_horizon)
    if preset == 'guarded_v1':
        return {
            'event_target_threshold': _robust_quantile(valid_target, 0.88, floor=0.68),
            'event_space_threshold': _robust_quantile(valid_space, 0.80, floor=0.42),
            'hard_negative_space_threshold': _robust_quantile(valid_space, 0.56, floor=0.20),
            'hard_negative_downside_threshold': _robust_quantile(valid_downside, 0.66, floor=0.22),
            'terminal_confirmation_ratio': 0.42,
        }
    return {
        'event_target_threshold': _robust_quantile(valid_target, 0.82, floor=0.55),
        'event_space_threshold': _robust_quantile(valid_space, 0.70, floor=0.30),
        'hard_negative_space_threshold': _robust_quantile(valid_space, 0.55, floor=0.18),
        'hard_negative_downside_threshold': _robust_quantile(valid_downside, 0.70, floor=0.25),
        'terminal_confirmation_ratio': 0.35,
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
    clean_space = components['clean_space']
    downside_excursion = components['downside_excursion']
    breakout_space = components['breakout_space']
    terminal_breakout = components['terminal_breakout']
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
    event_space_threshold = float(threshold_cfg['event_space_threshold'])
    hard_negative_space_threshold = float(threshold_cfg['hard_negative_space_threshold'])
    hard_negative_downside_threshold = float(threshold_cfg['hard_negative_downside_threshold'])
    terminal_confirmation_ratio = float(threshold_cfg.get('terminal_confirmation_ratio', 0.35))

    breakout_event = (
        (breakout_target >= event_target_threshold) &
        (clean_space >= event_space_threshold) &
        (terminal_breakout >= terminal_confirmation_ratio * breakout_space)
    ).astype(np.float32)
    breakout_hard_negative = (
        (breakout_space >= hard_negative_space_threshold) &
        (
            (clean_space < event_space_threshold) |
            (downside_excursion >= hard_negative_downside_threshold) |
            (breakout_target < 0.85 * event_target_threshold)
        )
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
    log_ema20 = np.log(np.maximum(ema20, 1e-8))
    ekf_track = compute_ekf_track(log_close)
    ekf_residual = log_close - ekf_track
    ema_log_gap = log_close - log_ema20
    sigma_safe = sigma + 1e-8
    res_score = ekf_residual / sigma_safe
    ema_score = ema_log_gap / sigma_safe
    imbalance_direction = np.sign(res_score + ema_score).astype(np.float32)
    imbalance_alignment = np.abs(res_score + ema_score) / (np.abs(res_score) + np.abs(ema_score) + 1e-6)
    imbalance_strength = np.maximum(0.0, np.abs(res_score) + np.abs(ema_score) - 0.80)

    future_path = build_future_path(log_close, forecast_horizon)
    signed_future = future_path * imbalance_direction[:, None]
    continuation_excursion = np.maximum(signed_future, 0.0).max(axis=1) / sigma_safe
    reversal_excursion = np.maximum(-signed_future, 0.0).max(axis=1) / sigma_safe
    terminal_reversal = np.maximum(-(imbalance_direction * future_path[:, -1]), 0.0) / sigma_safe
    clean_reversal = np.maximum(reversal_excursion - 0.45 * continuation_excursion - 0.25, 0.0)
    clean_terminal = np.maximum(terminal_reversal - 0.20 * continuation_excursion, 0.0)

    entropy_ref = ema_np(entropy, LOCAL_PHYSICS_WINDOW)
    future_entropy = pd.Series(entropy).rolling(window=forecast_horizon, min_periods=1).mean().shift(-forecast_horizon).values
    future_entropy = np.nan_to_num(future_entropy, nan=entropy[-1]).astype(np.float32)
    entropy_release = np.maximum(future_entropy - entropy, 0.0)
    entropy_quiet = np.maximum(entropy_ref - entropy, 0.0)
    trend_pressure = np.maximum(hurst - 0.55, 0.0)
    imbalance_gate = np.clip(
        0.40 +
        0.40 * np.tanh(imbalance_strength / 1.25) +
        0.20 * imbalance_alignment,
        0.0,
        1.50,
    )
    guarded_target = np.maximum(
        0.0,
        (
            0.40 * clean_reversal +
            0.30 * clean_terminal +
            0.14 * entropy_release +
            0.10 * entropy_quiet +
            0.06 * imbalance_alignment
        ) * imbalance_gate -
        0.20 * continuation_excursion
    ).astype(np.float32)
    guarded_aux = np.maximum(
        0.0,
        0.64 * clean_reversal +
        0.18 * reversal_excursion +
        0.10 * clean_terminal +
        0.08 * trend_pressure
    ).astype(np.float32)
    return {
        'continuation_excursion': continuation_excursion.astype(np.float32),
        'reversal_excursion': reversal_excursion.astype(np.float32),
        'terminal_reversal': terminal_reversal.astype(np.float32),
        'clean_reversal': clean_reversal.astype(np.float32),
        'clean_terminal': clean_terminal.astype(np.float32),
        'imbalance_alignment': imbalance_alignment.astype(np.float32),
        'imbalance_strength': imbalance_strength.astype(np.float32),
        'entropy_release': entropy_release.astype(np.float32),
        'entropy_quiet': entropy_quiet.astype(np.float32),
        'trend_pressure': trend_pressure.astype(np.float32),
        'target_default': np.maximum(
            0.0,
            imbalance_strength * (
                0.34 * clean_reversal +
                0.28 * clean_terminal +
                0.18 * imbalance_alignment +
                0.12 * entropy_release +
                0.08 * entropy_quiet
            )
        ).astype(np.float32),
        'aux_default': np.maximum(
            0.0,
            (0.60 * clean_reversal + 0.25 * reversal_excursion + 0.15 * clean_terminal) *
            (0.55 + 0.30 * imbalance_alignment + 0.15 * trend_pressure)
        ).astype(np.float32),
        'target_guarded_v1': guarded_target,
        'aux_guarded_v1': guarded_aux,
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
    valid_reversal = _valid_prefix(components['clean_reversal'], forecast_horizon)
    valid_continuation = _valid_prefix(components['continuation_excursion'], forecast_horizon)
    if preset == 'guarded_v1':
        return {
            'event_target_threshold': _robust_quantile(valid_target, 0.88, floor=0.52),
            'event_reversal_threshold': _robust_quantile(valid_reversal, 0.80, floor=0.30),
            'hard_negative_reversal_threshold': _robust_quantile(valid_reversal, 0.56, floor=0.14),
            'continuation_threshold': _robust_quantile(valid_continuation, 0.66, floor=0.22),
            'terminal_confirmation_ratio': 0.42,
            'alignment_min': 0.34,
            'imbalance_strength_min': 0.76,
        }
    return {
        'event_target_threshold': _robust_quantile(valid_target, 0.82, floor=0.40),
        'event_reversal_threshold': _robust_quantile(valid_reversal, 0.72, floor=0.22),
        'hard_negative_reversal_threshold': _robust_quantile(valid_reversal, 0.50, floor=0.12),
        'continuation_threshold': _robust_quantile(valid_continuation, 0.70, floor=0.25),
        'terminal_confirmation_ratio': 0.35,
        'alignment_min': 0.25,
        'imbalance_strength_min': 0.70,
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
    clean_reversal = components['clean_reversal']
    continuation_excursion = components['continuation_excursion']
    reversal_excursion = components['reversal_excursion']
    terminal_reversal = components['terminal_reversal']
    imbalance_alignment = components['imbalance_alignment']
    imbalance_strength = components['imbalance_strength']
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
    event_reversal_threshold = float(threshold_cfg['event_reversal_threshold'])
    hard_negative_reversal_threshold = float(threshold_cfg['hard_negative_reversal_threshold'])
    continuation_threshold = float(threshold_cfg['continuation_threshold'])
    terminal_confirmation_ratio = float(threshold_cfg.get('terminal_confirmation_ratio', 0.35))
    alignment_min = float(threshold_cfg.get('alignment_min', 0.25))
    imbalance_strength_min = float(threshold_cfg.get('imbalance_strength_min', 0.70))

    reversion_event = (
        (target >= event_target_threshold) &
        (clean_reversal >= event_reversal_threshold) &
        (terminal_reversal >= terminal_confirmation_ratio * reversal_excursion) &
        (imbalance_alignment >= alignment_min)
    ).astype(np.float32)
    reversion_hard_negative = (
        (imbalance_strength >= imbalance_strength_min) &
        (
            (clean_reversal < hard_negative_reversal_threshold) |
            (target < 0.85 * event_target_threshold) |
            (continuation_excursion >= np.maximum(reversal_excursion, continuation_threshold))
        )
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
