import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from khaos.数据处理.ashare_support import (
    TIMEFRAME_TO_MINUTES,
    infer_asset_code_from_filename,
    infer_timeframe_from_filename,
    normalize_ohlcv_dataframe,
    normalize_timeframe_label,
    read_csv_with_fallback,
)
from khaos.数据处理.data_loader import (
    LOCAL_PHYSICS_WINDOW,
    build_breakout_targets,
    build_reversion_targets,
    compute_ekf_track,
    ema_np,
    rolling_entropy_proxy_np,
    rolling_hurst_proxy_np,
)
from khaos.核心引擎.physics import compute_physics_features_bulk, PHYSICS_FEATURE_NAMES

EVENT_FLAG_NAMES = (
    'breakout_event',
    'reversion_event',
    'breakout_hard_negative',
    'reversion_hard_negative',
    'reversion_down_context',
    'reversion_up_context',
    'continuation_pressure',
)
EVENT_FLAG_INDEX = {name: idx for idx, name in enumerate(EVENT_FLAG_NAMES)}
TRADE_MASK_NAMES = (
    'trade_block',
    'session_reset',
    'one_word_limit',
    'midday_break',
)
TRADE_MASK_INDEX = {name: idx for idx, name in enumerate(TRADE_MASK_NAMES)}

ITERA5_TIMEFRAME_WEIGHT = {
    '5m': 0.85,
    '15m': 1.00,
    '60m': 1.20,
    '240m': 1.05,
    '1d': 1.00,
}

SHORT_T_TIMEFRAME_WEIGHT = {
    '5m': 1.18,
    '15m': 1.20,
    '60m': 1.12,
    '240m': 0.90,
    '1d': 0.78,
}

SHORT_T_BALANCED_V1_TIMEFRAME_WEIGHT = {
    '5m': 1.05,
    '15m': 1.15,
    '60m': 1.18,
    '240m': 1.00,
    '1d': 0.95,
}

SHORT_T_BALANCED_V2_TIMEFRAME_WEIGHT = {
    '5m': 0.92,
    '15m': 1.10,
    '60m': 1.22,
    '240m': 1.02,
    '1d': 0.98,
}

SHORT_T_PRECISION_V1_TIMEFRAME_WEIGHT = {
    '5m': 0.82,
    '15m': 1.08,
    '60m': 1.34,
    '240m': 1.08,
    '1d': 1.02,
}

ROLLING_RECENT_SPLITS = {
    'fold_1': {
        'train_end': '2023-06-30',
        'val_start': '2023-07-01',
        'val_end': '2023-12-31',
        'test_start': '2024-01-01',
        'test_end': '2024-06-30',
    },
    'fold_2': {
        'train_end': '2023-12-31',
        'val_start': '2024-01-01',
        'val_end': '2024-06-30',
        'test_start': '2024-07-01',
        'test_end': '2024-12-31',
    },
    'fold_3': {
        'train_end': '2024-06-30',
        'val_start': '2024-07-01',
        'val_end': '2024-12-31',
        'test_start': '2025-01-01',
        'test_end': '2025-06-30',
    },
    'fold_4': {
        'train_end': '2024-12-31',
        'val_start': '2025-01-01',
        'val_end': '2025-06-30',
        'test_start': '2025-07-01',
        'test_end': '2025-12-31',
    },
    'final_holdout': {
        'train_end': '2025-06-30',
        'val_start': '2025-07-01',
        'val_end': '2025-12-31',
        'test_start': '2026-01-01',
        'test_end': None,
    },
}

HORIZON_TASK_ORDER = ('breakout', 'reversion')


def _softmax_np(values, axis=-1):
    values = np.asarray(values, dtype=np.float32)
    max_values = np.max(values, axis=axis, keepdims=True)
    exps = np.exp(values - max_values)
    denom = np.sum(exps, axis=axis, keepdims=True)
    denom = np.clip(denom, 1e-6, None)
    return (exps / denom).astype(np.float32)


def _ema_np(values, span):
    return pd.Series(values).ewm(span=max(int(span), 1), adjust=False).mean().values.astype(np.float32)


def _parse_kv_string(value):
    parsed = {}
    if not value:
        return parsed
    for item in str(value).split(','):
        item = item.strip()
        if not item or '=' not in item:
            continue
        key, raw_value = item.split('=', 1)
        parsed[key.strip()] = raw_value.strip()
    return parsed


def normalize_horizon_search_spec(spec):
    defaults = {
        'min_horizon': 20,
        'max_horizon': 120,
        'max_fraction': 0.12,
        'saturation_ratio': 0.01,
        'saturation_patience': 8,
        'saturation_ema': 5,
    }
    if spec is None:
        return defaults
    if isinstance(spec, dict):
        parsed = dict(spec)
    elif isinstance(spec, str):
        text = spec.strip()
        if not text:
            parsed = {}
        else:
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    parsed = {}
            except Exception:
                parsed = _parse_kv_string(text)
    else:
        parsed = {}
    merged = dict(defaults)
    for key, value in parsed.items():
        if key not in merged:
            continue
        if key in {'min_horizon', 'max_horizon', 'saturation_patience', 'saturation_ema'}:
            merged[key] = int(value)
        else:
            merged[key] = float(value)
    merged['min_horizon'] = max(int(merged['min_horizon']), 20)
    merged['max_horizon'] = max(int(merged['max_horizon']), merged['min_horizon'])
    merged['max_fraction'] = float(np.clip(merged['max_fraction'], 0.02, 0.5))
    merged['saturation_ratio'] = float(np.clip(merged['saturation_ratio'], 1e-4, 0.5))
    merged['saturation_patience'] = max(int(merged['saturation_patience']), 2)
    merged['saturation_ema'] = max(int(merged['saturation_ema']), 1)
    return merged


def _quantile_from_distribution(prob, horizons, q):
    cdf = np.cumsum(np.asarray(prob, dtype=np.float64))
    horizons = np.asarray(horizons, dtype=np.float64)
    if cdf.size == 0:
        return 0.0
    cdf[-1] = 1.0
    return float(np.interp(float(q), cdf, horizons))


def summarize_horizon_distribution(prob, horizons):
    prob = np.asarray(prob, dtype=np.float64)
    horizons = np.asarray(horizons, dtype=np.int64)
    if prob.size == 0 or horizons.size == 0:
        return {
            'mode_mass': 0.0,
            'h_mode': 0,
            'h_mode_index': 0,
            'iqr': 0.0,
            'distribution': [],
        }
    mode_index = int(np.argmax(prob))
    h_mode = int(horizons[mode_index])
    q25 = _quantile_from_distribution(prob, horizons, 0.25)
    q75 = _quantile_from_distribution(prob, horizons, 0.75)
    return {
        'mode_mass': float(prob[mode_index]),
        'h_mode': h_mode,
        'h_mode_index': mode_index,
        'iqr': float(max(q75 - q25, 0.0)),
        'distribution': [
            {'horizon': int(horizon), 'prob': float(weight)}
            for horizon, weight in zip(horizons.tolist(), prob.tolist())
        ],
    }


def build_rolling_recent_split_config(split_label):
    if split_label not in ROLLING_RECENT_SPLITS:
        raise KeyError(f'Unsupported rolling split label: {split_label}')
    return dict(ROLLING_RECENT_SPLITS[split_label])


def _infer_expected_bar_minutes(df, timeframe_label=None):
    normalized = normalize_timeframe_label(timeframe_label)
    if normalized in TIMEFRAME_TO_MINUTES:
        return TIMEFRAME_TO_MINUTES[normalized]
    if 'time' in df.columns and len(df) > 2:
        delta = df['time'].diff().dropna()
        delta = delta[delta > pd.Timedelta(0)]
        if not delta.empty:
            return max(int(round(delta.median().total_seconds() / 60.0)), 1)
    return 60


def _build_itera5_boundary_penalty(df, timeframe_label=None):
    normalized = normalize_timeframe_label(timeframe_label)
    penalty = np.ones(len(df), dtype=np.float32)
    if normalized != '5m' or len(df) == 0:
        return penalty
    hours = df['time'].dt.hour.to_numpy()
    minutes = df['time'].dt.minute.to_numpy()
    boundary_mask = (
        ((hours == 9) & np.isin(minutes, [35, 40])) |
        ((hours == 14) & (minutes == 55)) |
        ((hours == 15) & (minutes == 0))
    )
    penalty[boundary_mask] = 0.75
    return penalty


def _build_shortt_boundary_penalty(df, timeframe_label=None):
    normalized = normalize_timeframe_label(timeframe_label)
    penalty = np.ones(len(df), dtype=np.float32)
    if normalized != '5m' or len(df) == 0:
        return penalty
    hours = df['time'].dt.hour.to_numpy()
    minutes = df['time'].dt.minute.to_numpy()
    boundary_mask = (
        ((hours == 9) & np.isin(minutes, [35, 40, 45])) |
        ((hours == 14) & np.isin(minutes, [50, 55])) |
        ((hours == 15) & (minutes == 0))
    )
    penalty[boundary_mask] = 0.60
    return penalty


def _build_ashare_trade_profile(df, close, open_, high, low, volume):
    expected_minutes = _infer_expected_bar_minutes(df)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    price_gap = np.abs(open_ - prev_close) / np.maximum(prev_close, 1e-6)
    intraday_range = (high - low) / np.maximum(prev_close, 1e-6)
    one_word_limit = (intraday_range <= 5e-4) & (price_gap >= 0.08)

    rolling_volume = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    low_liquidity = volume <= np.maximum(rolling_volume * 0.08, 1.0)

    time_delta_minutes = df['time'].diff().dt.total_seconds().div(60).fillna(0.0).values.astype(np.float32)
    prev_time = df['time'].shift(1)
    same_day = (df['time'].dt.normalize() == prev_time.dt.normalize()).fillna(False).values
    prev_hour = prev_time.dt.hour.fillna(df['time'].dt.hour.iloc[0]).values
    current_hour = df['time'].dt.hour.values
    midday_break = (
        same_day &
        (time_delta_minutes >= 80.0) &
        (time_delta_minutes <= 180.0) &
        (prev_hour <= 11) &
        (current_hour >= 13)
    )
    long_gap = (time_delta_minutes > expected_minutes * 3.2) & ~midday_break & (np.arange(len(df)) > 0)
    overnight_gap = (~same_day) & (price_gap >= 0.07)
    announcement_gap = price_gap >= 0.095
    session_reset = ((~same_day) & (np.arange(len(df)) > 0)) | midday_break

    breakout_soft = 1.0
    breakout_soft -= 0.85 * one_word_limit.astype(np.float32)
    breakout_soft -= 0.30 * low_liquidity.astype(np.float32)
    breakout_soft -= 0.55 * long_gap.astype(np.float32)
    breakout_soft -= 0.30 * overnight_gap.astype(np.float32)
    breakout_soft -= 0.20 * announcement_gap.astype(np.float32)
    breakout_soft -= 0.18 * midday_break.astype(np.float32)
    breakout_soft = np.clip(breakout_soft, 0.05, 1.0).astype(np.float32)

    reversion_soft = 1.0
    reversion_soft -= 0.60 * one_word_limit.astype(np.float32)
    reversion_soft -= 0.20 * low_liquidity.astype(np.float32)
    reversion_soft -= 0.45 * long_gap.astype(np.float32)
    reversion_soft -= 0.15 * overnight_gap.astype(np.float32)
    reversion_soft -= 0.25 * announcement_gap.astype(np.float32)
    reversion_soft -= 0.15 * midday_break.astype(np.float32)
    reversion_soft = np.clip(reversion_soft, 0.05, 1.0).astype(np.float32)

    neutral_break = (midday_break & (price_gap < 0.02)).astype(np.float32)
    breakout_event_mask = (
        (breakout_soft >= 0.45) &
        ~one_word_limit &
        ~(neutral_break > 0.5)
    ).astype(np.float32)
    reversion_event_mask = (
        (reversion_soft >= 0.40) &
        ~one_word_limit &
        ~(neutral_break > 0.5)
    ).astype(np.float32)
    breakout_hard_negative_mask = (
        (breakout_soft >= 0.28) &
        ~one_word_limit &
        ~midday_break
    ).astype(np.float32)
    reversion_hard_negative_mask = (
        (reversion_soft >= 0.25) &
        ~one_word_limit &
        ~midday_break
    ).astype(np.float32)
    reversion_context_mask = (
        (~one_word_limit) &
        (reversion_soft >= 0.20) &
        ~(session_reset & (price_gap < 0.03))
    ).astype(np.float32)
    trade_block = np.clip(
        1.00 * one_word_limit.astype(np.float32) +
        0.72 * midday_break.astype(np.float32) +
        0.78 * long_gap.astype(np.float32) +
        0.82 * overnight_gap.astype(np.float32) +
        0.76 * announcement_gap.astype(np.float32) +
        0.60 * session_reset.astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    return {
        'breakout_soft': breakout_soft,
        'reversion_soft': reversion_soft,
        'breakout_event_mask': breakout_event_mask,
        'reversion_event_mask': reversion_event_mask,
        'breakout_hard_negative_mask': breakout_hard_negative_mask,
        'reversion_hard_negative_mask': reversion_hard_negative_mask,
        'reversion_context_mask': reversion_context_mask,
        'session_reset': session_reset.astype(np.float32),
        'one_word_limit': one_word_limit.astype(np.float32),
        'midday_break': midday_break.astype(np.float32),
        'long_gap': long_gap.astype(np.float32),
        'overnight_gap': overnight_gap.astype(np.float32),
        'announcement_gap': announcement_gap.astype(np.float32),
        'trade_block': trade_block,
        'sample_weight': (0.55 + 0.45 * np.minimum(breakout_soft, reversion_soft)).astype(np.float32),
    }


def _safe_horizon_upper_bound(train_length, spec):
    min_horizon = int(spec['min_horizon'])
    raw_upper = min(
        int(spec['max_horizon']),
        int(np.floor(float(train_length) * float(spec['max_fraction']))),
    )
    return max(raw_upper, min_horizon)


def _discover_candidate_horizons(utility_by_horizon, candidate_horizons, spec, train_length):
    candidate_horizons = np.asarray(candidate_horizons, dtype=np.int64)
    if candidate_horizons.size == 0:
        return [int(spec['min_horizon'])]
    if candidate_horizons.size == 1:
        return candidate_horizons.tolist()
    positive_delta = np.maximum(0.0, utility_by_horizon[:, 1:] - utility_by_horizon[:, :-1])
    g = positive_delta.mean(axis=0)
    g_ema = _ema_np(g, spec['saturation_ema'])
    baseline = max(float(g_ema[0]), 1e-6)
    threshold = float(spec['saturation_ratio']) * baseline
    patience = int(spec['saturation_patience'])
    limit_idx = None
    streak = 0
    for idx, value in enumerate(g_ema):
        if float(value) < threshold:
            streak += 1
            if streak >= patience:
                limit_idx = idx - patience + 2
                break
        else:
            streak = 0
    if limit_idx is None:
        max_safe = _safe_horizon_upper_bound(train_length, spec)
    else:
        max_safe = int(candidate_horizons[min(limit_idx, candidate_horizons.size - 1)])
    selected = candidate_horizons[candidate_horizons <= max_safe]
    if selected.size == 0:
        selected = candidate_horizons[:1]
    return selected.tolist()


def build_horizon_candidates(train_length, horizon_search_spec):
    spec = normalize_horizon_search_spec(horizon_search_spec)
    min_horizon = int(spec['min_horizon'])
    upper = _safe_horizon_upper_bound(train_length, spec)
    if upper <= min_horizon:
        return [min_horizon], spec
    return list(range(min_horizon, upper + 1)), spec


def build_global_horizon_grid(horizon_search_spec, train_lengths=None):
    spec = normalize_horizon_search_spec(horizon_search_spec)
    min_horizon = int(spec['min_horizon'])
    max_horizon = int(spec['max_horizon'])
    if train_lengths:
        safe_uppers = [
            _safe_horizon_upper_bound(int(train_length), spec)
            for train_length in train_lengths
            if int(train_length) > 0
        ]
        if safe_uppers:
            max_horizon = min(max_horizon, max(safe_uppers))
    if max_horizon <= min_horizon:
        return [min_horizon]
    return list(range(min_horizon, max_horizon + 1))


def _collect_horizon_targets(
    log_close,
    log_high,
    log_low,
    returns,
    ema20,
    sigma,
    entropy,
    hurst,
    trade_profile,
    continuation_pressure,
    candidate_horizons,
):
    breakout_targets = []
    breakout_aux_targets = []
    breakout_events = []
    breakout_hard_negatives = []
    reversion_targets = []
    reversion_aux_targets = []
    reversion_events = []
    reversion_hard_negatives = []
    breakout_utilities = []
    reversion_utilities = []

    for horizon in candidate_horizons:
        breakout_target, breakout_aux, breakout_event, breakout_hard_negative = build_breakout_targets(
            log_close,
            log_high,
            log_low,
            returns,
            sigma,
            entropy,
            horizon,
        )
        reversion_target, reversion_aux, reversion_event, reversion_hard_negative = build_reversion_targets(
            log_close,
            ema20,
            sigma,
            entropy,
            hurst,
            horizon,
        )

        breakout_target = breakout_target * trade_profile['breakout_soft']
        breakout_aux = breakout_aux * trade_profile['breakout_soft']
        reversion_target = reversion_target * trade_profile['reversion_soft']
        reversion_aux = reversion_aux * trade_profile['reversion_soft']
        breakout_event = (breakout_event * trade_profile['breakout_event_mask']).astype(np.float32)
        breakout_hard_negative = (breakout_hard_negative * trade_profile['breakout_hard_negative_mask']).astype(np.float32)
        reversion_event = (reversion_event * trade_profile['reversion_event_mask']).astype(np.float32)
        reversion_hard_negative = (reversion_hard_negative * trade_profile['reversion_hard_negative_mask']).astype(np.float32)

        breakout_targets.append(breakout_target.astype(np.float32))
        breakout_aux_targets.append(breakout_aux.astype(np.float32))
        breakout_events.append(breakout_event.astype(np.float32))
        breakout_hard_negatives.append(breakout_hard_negative.astype(np.float32))
        reversion_targets.append(reversion_target.astype(np.float32))
        reversion_aux_targets.append(reversion_aux.astype(np.float32))
        reversion_events.append(reversion_event.astype(np.float32))
        reversion_hard_negatives.append(reversion_hard_negative.astype(np.float32))

        breakout_utilities.append(
            (
                1.00 * breakout_event +
                0.35 * breakout_aux -
                1.20 * breakout_hard_negative -
                0.30 * trade_profile['trade_block']
            ).astype(np.float32)
        )
        reversion_utilities.append(
            (
                1.00 * reversion_event +
                0.35 * reversion_aux -
                1.30 * reversion_hard_negative -
                0.40 * continuation_pressure -
                0.30 * trade_profile['trade_block']
            ).astype(np.float32)
        )

    breakout_targets = np.stack(breakout_targets, axis=1).astype(np.float32)
    breakout_aux_targets = np.stack(breakout_aux_targets, axis=1).astype(np.float32)
    breakout_events = np.stack(breakout_events, axis=1).astype(np.float32)
    breakout_hard_negatives = np.stack(breakout_hard_negatives, axis=1).astype(np.float32)
    reversion_targets = np.stack(reversion_targets, axis=1).astype(np.float32)
    reversion_aux_targets = np.stack(reversion_aux_targets, axis=1).astype(np.float32)
    reversion_events = np.stack(reversion_events, axis=1).astype(np.float32)
    reversion_hard_negatives = np.stack(reversion_hard_negatives, axis=1).astype(np.float32)
    breakout_utilities = np.stack(breakout_utilities, axis=1).astype(np.float32)
    reversion_utilities = np.stack(reversion_utilities, axis=1).astype(np.float32)

    return {
        'breakout_targets': breakout_targets,
        'breakout_aux_targets': breakout_aux_targets,
        'breakout_events': breakout_events,
        'breakout_hard_negatives': breakout_hard_negatives,
        'reversion_targets': reversion_targets,
        'reversion_aux_targets': reversion_aux_targets,
        'reversion_events': reversion_events,
        'reversion_hard_negatives': reversion_hard_negatives,
        'breakout_utilities': breakout_utilities,
        'reversion_utilities': reversion_utilities,
    }


def discover_horizon_profile(train_df, window_size, timeframe_label, horizon_search_spec):
    train_df = normalize_ohlcv_dataframe(train_df)
    close = train_df['close'].values.astype(np.float32)
    open_ = train_df['open'].values.astype(np.float32)
    high = train_df['high'].values.astype(np.float32)
    low = train_df['low'].values.astype(np.float32)
    volume = train_df['volume'].values.astype(np.float32)
    log_close = np.log(np.maximum(close, 1e-8))
    log_high = np.log(np.maximum(high, 1e-8))
    log_low = np.log(np.maximum(low, 1e-8))
    returns = np.diff(log_close, prepend=log_close[0])
    sigma = pd.Series(returns).rolling(window=LOCAL_PHYSICS_WINDOW, min_periods=1).std().bfill().values
    sigma = np.maximum(sigma, 1e-6).astype(np.float32)
    ema20 = ema_np(close, 20)
    entropy = rolling_entropy_proxy_np(high, low, close, LOCAL_PHYSICS_WINDOW)
    hurst = rolling_hurst_proxy_np(log_close, LOCAL_PHYSICS_WINDOW)
    log_ema20 = np.log(np.maximum(ema20, 1e-8))
    ekf_track = compute_ekf_track(log_close)
    ekf_residual = log_close - ekf_track
    ema_log_gap = log_close - log_ema20
    res_score = ekf_residual / (sigma + 1e-8)
    ema_score = ema_log_gap / (sigma + 1e-8)
    imbalance_alignment = np.abs(res_score + ema_score) / (np.abs(res_score) + np.abs(ema_score) + 1e-6)
    imbalance_strength = np.maximum(0.0, np.abs(res_score) + np.abs(ema_score) - 0.8)
    abs_returns = np.abs(returns) / (sigma + 1e-8)
    abr = np.abs(returns) + 1e-6
    abr_prev = np.roll(abr, 1)
    abr_prev[0] = abr[0]
    mle_proxy = ema_np(np.log((abr + 1e-6) / (abr_prev + 1e-6)), LOCAL_PHYSICS_WINDOW)
    reversion_setup = np.maximum(np.abs(res_score) - 1.0, 0.0) * np.maximum(np.abs(ema_score) - 0.5, 0.0) * imbalance_alignment
    trade_profile = _build_ashare_trade_profile(train_df, close, open_, high, low, volume)
    continuation_pressure = (
        (hurst >= 0.57) &
        (mle_proxy >= 0.03) &
        (abs_returns >= 0.55) &
        (reversion_setup < 0.45) &
        (trade_profile['reversion_context_mask'] > 0.5) &
        ~(trade_profile['session_reset'] > 0.5)
    ).astype(np.float32)

    initial_candidates, spec = build_horizon_candidates(len(train_df), horizon_search_spec)
    collected = _collect_horizon_targets(
        log_close=log_close,
        log_high=log_high,
        log_low=log_low,
        returns=returns,
        ema20=ema20,
        sigma=sigma,
        entropy=entropy,
        hurst=hurst,
        trade_profile=trade_profile,
        continuation_pressure=continuation_pressure,
        candidate_horizons=initial_candidates,
    )
    selected_candidates = _discover_candidate_horizons(
        utility_by_horizon=collected['breakout_utilities'] + collected['reversion_utilities'],
        candidate_horizons=initial_candidates,
        spec=spec,
        train_length=len(train_df),
    )
    if len(selected_candidates) != len(initial_candidates):
        keep_mask = np.isin(np.asarray(initial_candidates, dtype=np.int64), np.asarray(selected_candidates, dtype=np.int64))
        for key in list(collected.keys()):
            collected[key] = collected[key][:, keep_mask]

    breakout_q = _softmax_np(
        (collected['breakout_utilities'] - collected['breakout_utilities'].mean(axis=1, keepdims=True)) /
        (collected['breakout_utilities'].std(axis=1, keepdims=True) + 1e-6),
        axis=1,
    )
    reversion_q = _softmax_np(
        (collected['reversion_utilities'] - collected['reversion_utilities'].mean(axis=1, keepdims=True)) /
        (collected['reversion_utilities'].std(axis=1, keepdims=True) + 1e-6),
        axis=1,
    )
    breakout_prob = breakout_q.mean(axis=0)
    reversion_prob = reversion_q.mean(axis=0)
    breakout_summary = summarize_horizon_distribution(breakout_prob, selected_candidates)
    reversion_summary = summarize_horizon_distribution(reversion_prob, selected_candidates)
    return {
        'timeframe': normalize_timeframe_label(timeframe_label),
        'candidate_horizons': [int(item) for item in selected_candidates],
        'search_spec': spec,
        'task_stats': {
            'breakout': breakout_summary,
            'reversion': reversion_summary,
        },
        'q_distribution_mean': {
            'breakout': [float(item) for item in breakout_prob.tolist()],
            'reversion': [float(item) for item in reversion_prob.tolist()],
        },
    }

class AshareFinancialDataset(Dataset):
    def __init__(
        self,
        df,
        window_size=20,
        forecast_horizon=10,
        volatility_window=LOCAL_PHYSICS_WINDOW,
        timeframe_label=None,
        start_index=0,
        dataset_profile='iterA2',
        horizon_search_spec=None,
        horizon_family_mode='legacy',
        horizon_profile=None,
        global_horizon_grid=None,
    ):
        self.window_size = window_size
        self.feature_names = PHYSICS_FEATURE_NAMES
        self.timeframe_label = normalize_timeframe_label(timeframe_label)
        self.start_index = max(int(start_index), 0)
        self.dataset_profile = str(dataset_profile or 'iterA2')
        self.horizon_search_spec = normalize_horizon_search_spec(horizon_search_spec) if horizon_search_spec is not None else None
        self.horizon_family_mode = str(horizon_family_mode or 'legacy')
        self.horizon_aware = self.horizon_search_spec is not None
        self.global_horizon_grid = [int(item) for item in global_horizon_grid] if global_horizon_grid is not None else None
        self.use_continuation_flag = self.dataset_profile in {
            'iterA3',
            'iterA4',
            'iterA5',
            'shortT_v1',
            'shortT_balanced_v1',
            'shortT_balanced_v2',
            'shortT_precision_v1',
        }
        self.forecast_horizon = int(forecast_horizon) if not self.horizon_aware else 0

        df = normalize_ohlcv_dataframe(df)
        self.time = df['time'].copy()
        self.close = df['close'].values.astype(np.float32)
        self.open = df['open'].values.astype(np.float32)
        self.high = df['high'].values.astype(np.float32)
        self.low = df['low'].values.astype(np.float32)
        self.volume = df['volume'].values.astype(np.float32)

        log_close = np.log(np.maximum(self.close, 1e-8))
        log_high = np.log(np.maximum(self.high, 1e-8))
        log_low = np.log(np.maximum(self.low, 1e-8))
        returns = np.diff(log_close, prepend=log_close[0])
        sigma = pd.Series(returns).rolling(window=volatility_window, min_periods=1).std().bfill().values
        self.sigma = np.maximum(sigma, 1e-6).astype(np.float32)
        self.ema20 = ema_np(self.close, 20)
        entropy = rolling_entropy_proxy_np(self.high, self.low, self.close, LOCAL_PHYSICS_WINDOW)
        hurst = rolling_hurst_proxy_np(log_close, LOCAL_PHYSICS_WINDOW)

        breakout_target = breakout_aux = breakout_event = breakout_hard_negative = None
        reversion_target = reversion_aux = reversion_event = reversion_hard_negative = None
        if not self.horizon_aware:
            breakout_target, breakout_aux, breakout_event, breakout_hard_negative = build_breakout_targets(
                log_close,
                log_high,
                log_low,
                returns,
                self.sigma,
                entropy,
                self.forecast_horizon,
            )
            reversion_target, reversion_aux, reversion_event, reversion_hard_negative = build_reversion_targets(
                log_close,
                self.ema20,
                self.sigma,
                entropy,
                hurst,
                self.forecast_horizon,
            )

        log_ema20 = np.log(np.maximum(self.ema20, 1e-8))
        ekf_track = compute_ekf_track(log_close)
        ekf_residual = log_close - ekf_track
        ema_log_gap = log_close - log_ema20
        res_score = ekf_residual / (self.sigma + 1e-8)
        ema_score = ema_log_gap / (self.sigma + 1e-8)
        imbalance_direction = np.sign(res_score + ema_score).astype(np.float32)
        imbalance_alignment = np.abs(res_score + ema_score) / (np.abs(res_score) + np.abs(ema_score) + 1e-6)
        imbalance_strength = np.maximum(0.0, np.abs(res_score) + np.abs(ema_score) - 0.8)
        abs_returns = np.abs(returns) / (self.sigma + 1e-8)
        abr = np.abs(returns) + 1e-6
        abr_prev = np.roll(abr, 1)
        abr_prev[0] = abr[0]
        mle_proxy = ema_np(np.log((abr + 1e-6) / (abr_prev + 1e-6)), LOCAL_PHYSICS_WINDOW)
        reversion_setup = np.maximum(np.abs(res_score) - 1.0, 0.0) * np.maximum(np.abs(ema_score) - 0.5, 0.0) * imbalance_alignment

        trade_profile = _build_ashare_trade_profile(df, self.close, self.open, self.high, self.low, self.volume)
        reversion_down_context = (
            (imbalance_direction > 0) &
            (imbalance_strength >= 0.30) &
            (imbalance_alignment >= 0.35)
        ).astype(np.float32) * trade_profile['reversion_context_mask']
        reversion_up_context = (
            (imbalance_direction < 0) &
            (imbalance_strength >= 0.30) &
            (imbalance_alignment >= 0.35)
        ).astype(np.float32) * trade_profile['reversion_context_mask']
        continuation_pressure = (
            (hurst >= 0.57) &
            (mle_proxy >= 0.03) &
            (abs_returns >= 0.55) &
            (reversion_setup < 0.45) &
            (trade_profile['reversion_context_mask'] > 0.5) &
            ~(trade_profile['session_reset'] > 0.5)
        ).astype(np.float32)
        self.trade_masks = np.stack(
            [
                trade_profile['trade_block'],
                trade_profile['session_reset'],
                trade_profile['one_word_limit'],
                trade_profile['midday_break'],
            ],
            axis=1,
        ).astype(np.float32)

        if self.horizon_aware:
            if horizon_profile is None:
                horizon_profile = discover_horizon_profile(
                    train_df=df,
                    window_size=window_size,
                    timeframe_label=self.timeframe_label,
                    horizon_search_spec=self.horizon_search_spec,
                )
            self.horizon_profile = horizon_profile
            self.candidate_horizons = [int(item) for item in horizon_profile['candidate_horizons']]
            if self.global_horizon_grid is None:
                self.global_horizon_grid = list(self.candidate_horizons)
            horizon_to_global = {int(horizon): idx for idx, horizon in enumerate(self.global_horizon_grid)}
            global_indices = [int(horizon_to_global[horizon]) for horizon in self.candidate_horizons if int(horizon) in horizon_to_global]
            self.valid_horizon_mask = np.zeros(len(self.global_horizon_grid), dtype=np.float32)
            self.valid_horizon_mask[global_indices] = 1.0
            self.forecast_horizon = max(self.candidate_horizons)
            collected = _collect_horizon_targets(
                log_close=log_close,
                log_high=log_high,
                log_low=log_low,
                returns=returns,
                ema20=self.ema20,
                sigma=self.sigma,
                entropy=entropy,
                hurst=hurst,
                trade_profile=trade_profile,
                continuation_pressure=continuation_pressure,
                candidate_horizons=self.candidate_horizons,
            )
            breakout_q = _softmax_np(
                (collected['breakout_utilities'] - collected['breakout_utilities'].mean(axis=1, keepdims=True)) /
                (collected['breakout_utilities'].std(axis=1, keepdims=True) + 1e-6),
                axis=1,
            )
            reversion_q = _softmax_np(
                (collected['reversion_utilities'] - collected['reversion_utilities'].mean(axis=1, keepdims=True)) /
                (collected['reversion_utilities'].std(axis=1, keepdims=True) + 1e-6),
                axis=1,
            )
            horizon_count = len(self.global_horizon_grid)
            sample_count = len(df)
            self.targets_by_horizon = np.zeros((sample_count, 2, horizon_count), dtype=np.float32)
            self.aux_targets_by_horizon = np.zeros((sample_count, 2, horizon_count), dtype=np.float32)
            self.event_flags_by_horizon = np.zeros((sample_count, 2, horizon_count), dtype=np.float32)
            self.hard_negative_by_horizon = np.zeros((sample_count, 2, horizon_count), dtype=np.float32)
            self.q_horizon = np.zeros((sample_count, 2, horizon_count), dtype=np.float32)
            for local_idx, global_idx in enumerate(global_indices):
                self.targets_by_horizon[:, 0, global_idx] = collected['breakout_targets'][:, local_idx]
                self.targets_by_horizon[:, 1, global_idx] = collected['reversion_targets'][:, local_idx]
                self.aux_targets_by_horizon[:, 0, global_idx] = collected['breakout_aux_targets'][:, local_idx]
                self.aux_targets_by_horizon[:, 1, global_idx] = collected['reversion_aux_targets'][:, local_idx]
                self.event_flags_by_horizon[:, 0, global_idx] = collected['breakout_events'][:, local_idx]
                self.event_flags_by_horizon[:, 1, global_idx] = collected['reversion_events'][:, local_idx]
                self.hard_negative_by_horizon[:, 0, global_idx] = collected['breakout_hard_negatives'][:, local_idx]
                self.hard_negative_by_horizon[:, 1, global_idx] = collected['reversion_hard_negatives'][:, local_idx]
                self.q_horizon[:, 0, global_idx] = breakout_q[:, local_idx]
                self.q_horizon[:, 1, global_idx] = reversion_q[:, local_idx]
            breakout_fixed_idx = int(horizon_to_global[self.horizon_profile['task_stats']['breakout']['h_mode']])
            reversion_fixed_idx = int(horizon_to_global[self.horizon_profile['task_stats']['reversion']['h_mode']])
            if self.horizon_family_mode == 'single_cycle':
                self.selected_horizon_indices = np.tile(
                    np.array([breakout_fixed_idx, reversion_fixed_idx], dtype=np.int64),
                    (len(df), 1),
                )
                self.horizon_prior = np.zeros_like(self.q_horizon, dtype=np.float32)
                self.horizon_prior[:, 0, breakout_fixed_idx] = 1.0
                self.horizon_prior[:, 1, reversion_fixed_idx] = 1.0
            else:
                self.selected_horizon_indices = np.stack(
                    [
                        np.argmax(self.q_horizon[:, 0, :], axis=1),
                        np.argmax(self.q_horizon[:, 1, :], axis=1),
                    ],
                    axis=1,
                ).astype(np.int64)
                self.horizon_prior = self.q_horizon.copy()

            breakout_idx = self.selected_horizon_indices[:, 0]
            reversion_idx = self.selected_horizon_indices[:, 1]
            row_index = np.arange(len(df), dtype=np.int64)
            breakout_target = self.targets_by_horizon[row_index, 0, breakout_idx]
            reversion_target = self.targets_by_horizon[row_index, 1, reversion_idx]
            breakout_aux = self.aux_targets_by_horizon[row_index, 0, breakout_idx]
            reversion_aux = self.aux_targets_by_horizon[row_index, 1, reversion_idx]
            breakout_event = self.event_flags_by_horizon[row_index, 0, breakout_idx]
            reversion_event = self.event_flags_by_horizon[row_index, 1, reversion_idx]
            breakout_hard_negative = self.hard_negative_by_horizon[row_index, 0, breakout_idx]
            reversion_hard_negative = self.hard_negative_by_horizon[row_index, 1, reversion_idx]
        else:
            self.horizon_profile = None
            self.candidate_horizons = [int(self.forecast_horizon)]
            self.global_horizon_grid = list(self.candidate_horizons)
            self.valid_horizon_mask = np.ones(len(self.global_horizon_grid), dtype=np.float32)
            self.targets_by_horizon = None
            self.aux_targets_by_horizon = None
            self.event_flags_by_horizon = None
            self.hard_negative_by_horizon = None
            self.q_horizon = None
            self.horizon_prior = None
            self.selected_horizon_indices = None
            breakout_target *= trade_profile['breakout_soft']
            breakout_aux *= trade_profile['breakout_soft']
            reversion_target *= trade_profile['reversion_soft']
            reversion_aux *= trade_profile['reversion_soft']
            breakout_event = (breakout_event * trade_profile['breakout_event_mask']).astype(np.float32)
            breakout_hard_negative = (breakout_hard_negative * trade_profile['breakout_hard_negative_mask']).astype(np.float32)
            reversion_event = (reversion_event * trade_profile['reversion_event_mask']).astype(np.float32)
            reversion_hard_negative = (reversion_hard_negative * trade_profile['reversion_hard_negative_mask']).astype(np.float32)

        self.targets = np.stack([breakout_target, reversion_target], axis=1).astype(np.float32)
        self.aux_targets = np.stack([breakout_aux, reversion_aux], axis=1).astype(np.float32)
        event_columns = [
            breakout_event,
            reversion_event,
            breakout_hard_negative,
            reversion_hard_negative,
            reversion_down_context,
            reversion_up_context,
        ]
        self.event_flag_names = EVENT_FLAG_NAMES[:-1]
        if self.use_continuation_flag:
            event_columns.append(continuation_pressure)
            self.event_flag_names = EVENT_FLAG_NAMES
        self.event_flags = np.stack(event_columns, axis=1).astype(np.float32)
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
        ).astype(np.float32) * trade_profile['sample_weight']
        if self.use_continuation_flag:
            self.sample_weights = self.sample_weights * (1.0 + 0.60 * continuation_pressure.astype(np.float32))
        self._apply_profile_weights(
            df=df,
            breakout_event=breakout_event,
            reversion_event=reversion_event,
            breakout_aux=breakout_aux,
            breakout_hard_negative=breakout_hard_negative,
            reversion_hard_negative=reversion_hard_negative,
            continuation_pressure=continuation_pressure,
        )

        raw_data = np.stack([
            self.open, self.high, self.low, self.close, self.volume, self.ema20,
        ], axis=1)
        print(f'  [Pre-computation] Extracting A-share physics features for {len(df)} rows...')
        raw_tensor = torch.tensor(raw_data, dtype=torch.float32)
        self.physics_features = compute_physics_features_bulk(raw_tensor, device='cpu')
        print(f'  [Pre-computation] Done. Feature shape: {self.physics_features.shape}')

        self._max_start = max(0, len(self.close) - self.window_size - self.forecast_horizon)
        self.start_index = min(self.start_index, self._max_start)
        self.sample_bar_indices = np.arange(
            self.start_index + self.window_size - 1,
            self.start_index + self.window_size - 1 + len(self),
            dtype=np.int64,
        )
        self._finalize_tensor_storage()

    @staticmethod
    def _as_tensor(value, dtype=None):
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous()

    def _finalize_tensor_storage(self):
        self.physics_features = self._as_tensor(self.physics_features, dtype=torch.float32)
        self.targets = self._as_tensor(self.targets, dtype=torch.float32)
        self.aux_targets = self._as_tensor(self.aux_targets, dtype=torch.float32)
        self.sigma = self._as_tensor(self.sigma, dtype=torch.float32)
        self.sample_weights = self._as_tensor(self.sample_weights, dtype=torch.float32)
        self.event_flags = self._as_tensor(self.event_flags, dtype=torch.float32)
        self.trade_masks = self._as_tensor(self.trade_masks, dtype=torch.float32)
        if self.targets_by_horizon is not None:
            self.targets_by_horizon = self._as_tensor(self.targets_by_horizon, dtype=torch.float32)
        if self.aux_targets_by_horizon is not None:
            self.aux_targets_by_horizon = self._as_tensor(self.aux_targets_by_horizon, dtype=torch.float32)
        if self.event_flags_by_horizon is not None:
            self.event_flags_by_horizon = self._as_tensor(self.event_flags_by_horizon, dtype=torch.float32)
        if self.hard_negative_by_horizon is not None:
            self.hard_negative_by_horizon = self._as_tensor(self.hard_negative_by_horizon, dtype=torch.float32)
        if self.q_horizon is not None:
            self.q_horizon = self._as_tensor(self.q_horizon, dtype=torch.float32)
        if self.horizon_prior is not None:
            self.horizon_prior = self._as_tensor(self.horizon_prior, dtype=torch.float32)
        if self.selected_horizon_indices is not None:
            self.selected_horizon_indices = self._as_tensor(self.selected_horizon_indices, dtype=torch.int64)
        self.valid_horizon_mask = self._as_tensor(self.valid_horizon_mask, dtype=torch.float32)
        self.candidate_horizons_tensor = self._as_tensor(
            np.asarray(self.candidate_horizons, dtype=np.int64),
            dtype=torch.int64,
        )
        self.global_horizon_grid_tensor = self._as_tensor(
            np.asarray(self.global_horizon_grid, dtype=np.int64),
            dtype=torch.int64,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ('time', 'close', 'open', 'high', 'low', 'volume', 'ema20'):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'candidate_horizons_tensor') or not hasattr(self, 'global_horizon_grid_tensor'):
            self._finalize_tensor_storage()

    def _apply_profile_weights(
        self,
        df,
        breakout_event,
        reversion_event,
        breakout_aux,
        breakout_hard_negative,
        reversion_hard_negative,
        continuation_pressure,
    ):
        if self.dataset_profile == 'iterA5':
            timeframe_weight = ITERA5_TIMEFRAME_WEIGHT.get(self.timeframe_label, 1.0)
            boundary_penalty = _build_itera5_boundary_penalty(df, self.timeframe_label)
            self.sample_weights = self.sample_weights * timeframe_weight * boundary_penalty
        elif self.dataset_profile == 'shortT_v1':
            timeframe_weight = SHORT_T_TIMEFRAME_WEIGHT.get(self.timeframe_label, 1.0)
            boundary_penalty = _build_shortt_boundary_penalty(df, self.timeframe_label)
            breakout_bias = (
                1.0 +
                0.20 * breakout_event.astype(np.float32) +
                0.10 * breakout_aux.astype(np.float32) +
                0.16 * breakout_hard_negative.astype(np.float32)
            )
            self.sample_weights = self.sample_weights * timeframe_weight * boundary_penalty * breakout_bias
        elif self.dataset_profile == 'shortT_balanced_v1':
            timeframe_weight = SHORT_T_BALANCED_V1_TIMEFRAME_WEIGHT.get(self.timeframe_label, 1.0)
            boundary_penalty = _build_shortt_boundary_penalty(df, self.timeframe_label)
            self.sample_weights = self.sample_weights * timeframe_weight * boundary_penalty
        elif self.dataset_profile == 'shortT_balanced_v2':
            timeframe_weight = SHORT_T_BALANCED_V2_TIMEFRAME_WEIGHT.get(self.timeframe_label, 1.0)
            boundary_penalty = _build_shortt_boundary_penalty(df, self.timeframe_label)
            self.sample_weights = self.sample_weights * timeframe_weight * boundary_penalty
        elif self.dataset_profile == 'shortT_precision_v1':
            timeframe_weight = SHORT_T_PRECISION_V1_TIMEFRAME_WEIGHT.get(self.timeframe_label, 1.0)
            boundary_penalty = _build_shortt_boundary_penalty(df, self.timeframe_label)
            precision_bias = (
                1.0 +
                0.18 * breakout_event.astype(np.float32) +
                0.24 * reversion_event.astype(np.float32) +
                0.22 * breakout_hard_negative.astype(np.float32) +
                0.42 * reversion_hard_negative.astype(np.float32) +
                0.18 * continuation_pressure.astype(np.float32)
            )
            self.sample_weights = self.sample_weights * timeframe_weight * boundary_penalty * precision_bias

    def __len__(self):
        return max(0, self._max_start - self.start_index)

    def __getitem__(self, idx):
        actual_idx = idx + self.start_index
        end_idx = actual_idx + self.window_size
        feat_window = self.physics_features[actual_idx:end_idx]
        y = self.targets[end_idx - 1]
        aux_y = self.aux_targets[end_idx - 1]
        sigma = self.sigma[end_idx - 1]
        weight = self.sample_weights[end_idx - 1]
        flags = self.event_flags[end_idx - 1]
        payload = (
            feat_window,
            y,
            aux_y,
            sigma,
            weight,
            flags,
        )
        if not self.horizon_aware:
            return payload
        horizon_payload = {
            'targets_by_horizon': self.targets_by_horizon[end_idx - 1],
            'aux_by_horizon': self.aux_targets_by_horizon[end_idx - 1],
            'event_flags_by_horizon': self.event_flags_by_horizon[end_idx - 1],
            'hard_negative_by_horizon': self.hard_negative_by_horizon[end_idx - 1],
            'q_horizon': self.q_horizon[end_idx - 1],
            'trade_masks': self.trade_masks[end_idx - 1],
            'horizon_prior': self.horizon_prior[end_idx - 1],
            'selected_horizon_index': self.selected_horizon_indices[end_idx - 1],
            'candidate_horizons': self.candidate_horizons_tensor,
            'valid_horizon_mask': self.valid_horizon_mask,
            'global_horizon_grid': self.global_horizon_grid_tensor,
        }
        return payload + (horizon_payload,)


def load_ashare_data(file_path):
    return normalize_ohlcv_dataframe(read_csv_with_fallback(file_path))


def _build_segment(df, core_start_idx, core_end_idx, window_size):
    if core_start_idx is None or core_end_idx is None or core_start_idx > core_end_idx:
        return None
    segment_start = max(0, core_start_idx - window_size)
    segment_df = df.iloc[segment_start:core_end_idx + 1].copy()
    local_start_index = max(0, core_start_idx - segment_start - window_size + 1)
    return {'df': segment_df, 'start_index': local_start_index}


def _slice_indices(time_values, start=None, end=None):
    mask = np.ones(len(time_values), dtype=bool)
    if start is not None:
        mask &= time_values >= pd.Timestamp(start)
    if end is not None:
        mask &= time_values <= pd.Timestamp(end)
    return np.flatnonzero(mask)


def _build_rolling_split_payloads(df, split_label, window_size, horizon_profile):
    config = build_rolling_recent_split_config(split_label)
    train_end_ts = pd.Timestamp(config['train_end'])
    val_start_ts = pd.Timestamp(config['val_start'])
    val_end_ts = pd.Timestamp(config['val_end'])
    test_start_ts = pd.Timestamp(config['test_start'])
    test_end_ts = pd.Timestamp(config['test_end']) if config.get('test_end') else None
    time_values = df['time']

    train_indices = np.flatnonzero(time_values <= train_end_ts)
    if len(train_indices) == 0:
        return {}, config

    purge_gap = int(window_size + max(horizon_profile.get('candidate_horizons', [window_size])))
    val_indices = _slice_indices(time_values, start=val_start_ts, end=val_end_ts)
    val_indices = val_indices[val_indices > int(train_indices[-1]) + purge_gap]
    base_test_indices = _slice_indices(time_values, start=test_start_ts, end=test_end_ts)
    base_test_start = int(val_indices[-1]) if len(val_indices) else int(train_indices[-1])
    test_indices = base_test_indices[base_test_indices > base_test_start + purge_gap]

    split_payloads = {
        'train': {'df': df.iloc[:train_indices[-1] + 1].copy(), 'start_index': 0},
    }
    if len(val_indices):
        split_payloads['val'] = _build_segment(df, int(val_indices[0]), int(val_indices[-1]), window_size)
    if len(test_indices):
        split_payloads['test'] = _build_segment(df, int(test_indices[0]), int(test_indices[-1]), window_size)
    return split_payloads, config


def create_ashare_dataset_splits(
    file_path,
    window_size=20,
    horizon=10,
    train_end='2023-12-31',
    val_end='2024-12-31',
    test_start='2025-01-01',
    fast_full=False,
    return_metadata=False,
    dataset_profile='iterA2',
    split_scheme='time',
    split_label=None,
    horizon_search_spec=None,
    horizon_family_mode='legacy',
    global_horizon_grid=None,
):
    print(f'Loading {file_path}...')
    df = load_ashare_data(file_path)
    if fast_full:
        df = df.iloc[-20000:].reset_index(drop=True)
        print(f'  [Fast Mode] Loaded truncated history: {len(df)} rows')
    else:
        print(f'  Loaded full history: {len(df)} rows')

    timeframe_label = infer_timeframe_from_filename(file_path)
    metadata = {
        'asset_code': infer_asset_code_from_filename(file_path),
        'timeframe': normalize_timeframe_label(timeframe_label),
        'rows': len(df),
    }

    if split_scheme == 'rolling_recent_v1':
        if split_label is None:
            raise ValueError('`split_label` is required when `split_scheme=rolling_recent_v1`.')
        rolling_config = build_rolling_recent_split_config(split_label)
        train_df = df[df['time'] <= pd.Timestamp(rolling_config['train_end'])].copy()
        horizon_profile = discover_horizon_profile(
            train_df=train_df,
            window_size=window_size,
            timeframe_label=timeframe_label,
            horizon_search_spec=horizon_search_spec,
        )
        split_payloads, resolved_config = _build_rolling_split_payloads(
            df=df,
            split_label=split_label,
            window_size=window_size,
            horizon_profile=horizon_profile,
        )
        datasets = {
            split_name: AshareFinancialDataset(
                payload['df'],
                window_size=window_size,
                forecast_horizon=max(horizon_profile['candidate_horizons']),
                timeframe_label=timeframe_label,
                start_index=payload['start_index'],
                dataset_profile=dataset_profile,
                horizon_search_spec=horizon_search_spec,
                horizon_family_mode=horizon_family_mode,
                horizon_profile=horizon_profile,
                global_horizon_grid=global_horizon_grid,
            )
            for split_name, payload in split_payloads.items()
            if payload is not None
        }
        metadata.update(
            {
                'split_scheme': split_scheme,
                'split_label': split_label,
                'split_config': resolved_config,
                'split_rows': {
                    split_name: len(payload['df']) for split_name, payload in split_payloads.items() if payload is not None
                },
                'horizon_profile': horizon_profile,
                'candidate_horizons': horizon_profile['candidate_horizons'],
                'global_horizon_grid': global_horizon_grid or horizon_profile['candidate_horizons'],
                'horizon_family_mode': horizon_family_mode,
            }
        )
        if return_metadata:
            return datasets, metadata
        return datasets

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_start_ts = pd.Timestamp(test_start)
    time_values = df['time']

    train_indices = np.flatnonzero(time_values <= train_end_ts)
    val_indices = np.flatnonzero((time_values > train_end_ts) & (time_values <= val_end_ts))
    test_indices = np.flatnonzero(time_values >= test_start_ts)

    split_payloads = {}
    if len(train_indices):
        split_payloads['train'] = {'df': df.iloc[:train_indices[-1] + 1].copy(), 'start_index': 0}
    if len(val_indices):
        split_payloads['val'] = _build_segment(df, int(val_indices[0]), int(val_indices[-1]), window_size)
    if len(test_indices):
        split_payloads['test'] = _build_segment(df, int(test_indices[0]), len(df) - 1, window_size)

    datasets = {
        split_name: AshareFinancialDataset(
            payload['df'],
            window_size=window_size,
            forecast_horizon=horizon,
            timeframe_label=timeframe_label,
            start_index=payload['start_index'],
            dataset_profile=dataset_profile,
        )
        for split_name, payload in split_payloads.items()
        if payload is not None
    }

    metadata['split_rows'] = {
        split_name: len(payload['df']) for split_name, payload in split_payloads.items() if payload is not None
    }
    if return_metadata:
        return datasets, metadata
    return datasets
