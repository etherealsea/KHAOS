import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
    ema_np,
    rolling_entropy_proxy_np,
    rolling_hurst_proxy_np,
)
from khaos.核心引擎.physics import compute_physics_features_bulk, PHYSICS_FEATURE_NAMES


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

    breakout_soft = 1.0
    breakout_soft -= 0.85 * one_word_limit.astype(np.float32)
    breakout_soft -= 0.30 * low_liquidity.astype(np.float32)
    breakout_soft -= 0.55 * long_gap.astype(np.float32)
    breakout_soft -= 0.30 * overnight_gap.astype(np.float32)
    breakout_soft -= 0.20 * announcement_gap.astype(np.float32)
    breakout_soft = np.clip(breakout_soft, 0.05, 1.0).astype(np.float32)

    reversion_soft = 1.0
    reversion_soft -= 0.60 * one_word_limit.astype(np.float32)
    reversion_soft -= 0.20 * low_liquidity.astype(np.float32)
    reversion_soft -= 0.45 * long_gap.astype(np.float32)
    reversion_soft -= 0.15 * overnight_gap.astype(np.float32)
    reversion_soft -= 0.25 * announcement_gap.astype(np.float32)
    reversion_soft = np.clip(reversion_soft, 0.05, 1.0).astype(np.float32)

    return {
        'breakout_soft': breakout_soft,
        'reversion_soft': reversion_soft,
        'breakout_event_mask': (breakout_soft >= 0.45).astype(np.float32),
        'reversion_event_mask': (reversion_soft >= 0.40).astype(np.float32),
        'sample_weight': (0.55 + 0.45 * np.minimum(breakout_soft, reversion_soft)).astype(np.float32),
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
    ):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.feature_names = PHYSICS_FEATURE_NAMES
        self.timeframe_label = normalize_timeframe_label(timeframe_label)
        self.start_index = max(int(start_index), 0)

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

        trade_profile = _build_ashare_trade_profile(df, self.close, self.open, self.high, self.low, self.volume)
        breakout_target *= trade_profile['breakout_soft']
        breakout_aux *= trade_profile['breakout_soft']
        reversion_target *= trade_profile['reversion_soft']
        reversion_aux *= trade_profile['reversion_soft']
        breakout_event = (breakout_event * trade_profile['breakout_event_mask']).astype(np.float32)
        breakout_hard_negative = (breakout_hard_negative * trade_profile['breakout_event_mask']).astype(np.float32)
        reversion_event = (reversion_event * trade_profile['reversion_event_mask']).astype(np.float32)
        reversion_hard_negative = (reversion_hard_negative * trade_profile['reversion_event_mask']).astype(np.float32)

        self.targets = np.stack([breakout_target, reversion_target], axis=1).astype(np.float32)
        self.aux_targets = np.stack([breakout_aux, reversion_aux], axis=1).astype(np.float32)
        self.event_flags = np.stack([
            breakout_event,
            reversion_event,
            breakout_hard_negative,
            reversion_hard_negative,
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
        ).astype(np.float32) * trade_profile['sample_weight']

        raw_data = np.stack([
            self.open, self.high, self.low, self.close, self.volume, self.ema20,
        ], axis=1)
        print(f'  [Pre-computation] Extracting A-share physics features for {len(df)} rows...')
        raw_tensor = torch.tensor(raw_data, dtype=torch.float32)
        self.physics_features = compute_physics_features_bulk(raw_tensor, device='cpu')
        print(f'  [Pre-computation] Done. Feature shape: {self.physics_features.shape}')

        self._max_start = max(0, len(self.close) - self.window_size - self.forecast_horizon)
        self.start_index = min(self.start_index, self._max_start)

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
        return (
            feat_window,
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(aux_y, dtype=torch.float32),
            torch.tensor(sigma, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(flags, dtype=torch.float32),
        )


def load_ashare_data(file_path):
    return normalize_ohlcv_dataframe(read_csv_with_fallback(file_path))


def _build_segment(df, core_start_idx, core_end_idx, window_size):
    if core_start_idx is None or core_end_idx is None or core_start_idx > core_end_idx:
        return None
    segment_start = max(0, core_start_idx - window_size)
    segment_df = df.iloc[segment_start:core_end_idx + 1].copy()
    local_start_index = max(0, core_start_idx - segment_start - window_size + 1)
    return {'df': segment_df, 'start_index': local_start_index}


def create_ashare_dataset_splits(
    file_path,
    window_size=20,
    horizon=10,
    train_end='2023-12-31',
    val_end='2024-12-31',
    test_start='2025-01-01',
    fast_full=False,
    return_metadata=False,
):
    print(f'Loading {file_path}...')
    df = load_ashare_data(file_path)
    if fast_full:
        df = df.iloc[-20000:].reset_index(drop=True)
        print(f'  [Fast Mode] Loaded truncated history: {len(df)} rows')
    else:
        print(f'  Loaded full history: {len(df)} rows')

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

    timeframe_label = infer_timeframe_from_filename(file_path)
    datasets = {
        split_name: AshareFinancialDataset(
            payload['df'],
            window_size=window_size,
            forecast_horizon=horizon,
            timeframe_label=timeframe_label,
            start_index=payload['start_index'],
        )
        for split_name, payload in split_payloads.items()
        if payload is not None
    }

    metadata = {
        'asset_code': infer_asset_code_from_filename(file_path),
        'timeframe': normalize_timeframe_label(timeframe_label),
        'rows': len(df),
        'split_rows': {
            split_name: len(payload['df']) for split_name, payload in split_payloads.items() if payload is not None
        },
    }
    if return_metadata:
        return datasets, metadata
    return datasets
