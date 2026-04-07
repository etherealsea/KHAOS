import importlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


ASHARE_PRIMARY_ASSETS = [
    '600036', '601166', '600030', '601318',
    '600519', '000858', '600887', '000333', '600690',
    '600309', '601899', '600031', '600900', '600028',
    '300750', '002594', '002475', '002415', '300059',
    '600276', '300760', '300124', '601012', '603288',
]

ASHARE_FALLBACK_ASSETS = [
    '000651', '000725', '601668', '601088', '603986', '002142',
]

LEGACY_ITER9_ASSETS = ['BTC', 'ETH', 'GOLD', 'XAU', 'OIL', 'WTI', 'SPX', 'NDX', 'DJI', 'ES']
DEFAULT_ASHARE_TIMEFRAMES = ['15m', '60m', '1d']
DEFAULT_TRAIN_END = '2023-12-31'
DEFAULT_VAL_END = '2024-12-31'
DEFAULT_TEST_START = '2025-01-01'

TIMEFRAME_TO_SUFFIX = {
    '5m': '5m',
    '15m': '15m',
    '60m': '1h',
    '240m': '4h',
    '1d': '1d',
}

SUFFIX_TO_TIMEFRAME = {
    '5m': '5m',
    '15m': '15m',
    '1h': '60m',
    '4h': '240m',
    '1d': '1d',
}

TIMEFRAME_TO_MINUTES = {
    '5m': 5,
    '15m': 15,
    '60m': 60,
    '240m': 240,
    '1d': 1440,
}

STANDARD_REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']
STANDARD_OPTIONAL_COLUMNS = ['amount', 'turnover', 'adj_factor']

COLUMN_ALIASES = {
    'time': ['time', 'datetime', 'date', 'trade_time', 'trading_time', '日期', '时间', '交易时间'],
    'open': ['open', '开盘', 'open_price'],
    'high': ['high', '最高', 'high_price'],
    'low': ['low', '最低', 'low_price'],
    'close': ['close', '收盘', 'close_price', 'price'],
    'volume': ['volume', 'vol', '成交量', '成交量(股)', 'volume_shares'],
    'amount': ['amount', 'amt', '成交额', '成交金额'],
    'turnover': ['turnover', '换手率', 'turnover_rate'],
    'adj_factor': ['adj_factor', '复权因子', 'adjfactor'],
}


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(',') if part.strip()]
    return [str(value).strip()]


def normalize_timeframe_label(label):
    if label is None:
        return None
    value = str(label).strip().lower()
    replacements = {
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '60min': '60m',
        '1h': '60m',
        '4h': '240m',
        '240min': '240m',
        'day': '1d',
        'd': '1d',
    }
    return replacements.get(value, value)


def timeframe_to_suffix(timeframe):
    normalized = normalize_timeframe_label(timeframe)
    if normalized not in TIMEFRAME_TO_SUFFIX:
        raise ValueError(f'Unsupported timeframe: {timeframe}')
    return TIMEFRAME_TO_SUFFIX[normalized]


def canonical_training_filename(asset_code, timeframe):
    return f'{asset_code}_{timeframe_to_suffix(timeframe)}.csv'


def infer_timeframe_from_filename(file_path):
    basename = os.path.splitext(os.path.basename(file_path))[0].lower()
    for suffix, timeframe in SUFFIX_TO_TIMEFRAME.items():
        if basename.endswith(f'_{suffix}'):
            return timeframe
    return None


def infer_asset_code_from_filename(file_path):
    basename = os.path.splitext(os.path.basename(file_path))[0]
    basename = re.sub(r'_(5m|15m|1h|60m|4h|240m|1d)$', '', basename, flags=re.IGNORECASE)
    match = re.search(r'(?<!\d)(\d{6})(?!\d)', basename)
    if match:
        return match.group(1)
    return basename.upper()


def resolve_training_ready_dir(data_dir, market=None, training_subdir=None):
    base_dir = os.path.join(data_dir, 'training_ready')
    if training_subdir:
        return os.path.join(base_dir, training_subdir)
    if market == 'ashare':
        return os.path.join(base_dir, 'ashare')
    return base_dir


def discover_training_files(data_dir, market=None, assets=None, timeframes=None, training_subdir=None):
    target_dir = resolve_training_ready_dir(data_dir, market=market, training_subdir=training_subdir)
    asset_filter = {item.upper() for item in ensure_list(assets)}
    timeframe_filter = {normalize_timeframe_label(item) for item in ensure_list(timeframes)}
    discovered = []
    if not os.path.isdir(target_dir):
        return discovered
    for root, _, files in os.walk(target_dir):
        for name in files:
            if not name.lower().endswith('.csv'):
                continue
            path = os.path.join(root, name)
            asset_code = infer_asset_code_from_filename(path)
            timeframe = infer_timeframe_from_filename(path)
            if asset_filter and asset_code.upper() not in asset_filter:
                continue
            if timeframe_filter and timeframe not in timeframe_filter:
                continue
            discovered.append({
                'path': path,
                'asset_code': asset_code,
                'timeframe': timeframe,
                'basename': name,
            })
    discovered.sort(key=lambda item: (item['asset_code'], item['timeframe'] or '', item['basename']))
    return discovered


def read_csv_with_fallback(file_path):
    last_error = None
    for encoding in ('utf-8', 'utf-8-sig', 'gbk', 'gb18030'):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(file_path)


def _rename_columns(df):
    renamed = {}
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.strip().lower()
            if alias_lower in lowered:
                renamed[lowered[alias_lower]] = standard_name
                break
    return df.rename(columns=renamed)


def normalize_ohlcv_dataframe(df):
    df = _rename_columns(df.copy())
    df.columns = [str(col).strip().lower() for col in df.columns]
    missing_columns = [col for col in STANDARD_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    keep_columns = STANDARD_REQUIRED_COLUMNS + [col for col in STANDARD_OPTIONAL_COLUMNS if col in df.columns]
    normalized = df[keep_columns].copy()
    normalized['time'] = pd.to_datetime(normalized['time'], errors='coerce')
    normalized = normalized.dropna(subset=['time'])
    for column in keep_columns:
        if column == 'time':
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors='coerce')
    normalized = normalized.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    normalized = normalized.sort_values('time').drop_duplicates(subset=['time'], keep='last').reset_index(drop=True)

    if getattr(normalized['time'].dt, 'tz', None) is not None:
        normalized['time'] = normalized['time'].dt.tz_localize(None)

    return normalized


def infer_timeframe_from_dataframe(df):
    if 'time' not in df.columns or len(df) < 3:
        return None
    deltas = df['time'].diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    median_minutes = deltas.median().total_seconds() / 60.0
    if median_minutes <= 7.5:
        return '5m'
    if median_minutes <= 22.5:
        return '15m'
    if median_minutes <= 90:
        return '60m'
    if median_minutes <= 360:
        return '240m'
    return '1d'


def resample_ohlcv_dataframe(df, timeframe):
    normalized = normalize_timeframe_label(timeframe)
    if normalized == '1d':
        rule = '1D'
    elif normalized == '240m':
        rule = None
    elif normalized == '60m':
        rule = '1h'
    elif normalized == '15m':
        rule = '15min'
    elif normalized == '5m':
        rule = '5min'
    else:
        raise ValueError(f'Unsupported resample timeframe: {timeframe}')

    indexed = df.set_index('time').sort_index()
    aggregations = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    if 'amount' in indexed.columns:
        aggregations['amount'] = 'sum'
    if 'turnover' in indexed.columns:
        aggregations['turnover'] = 'last'
    if 'adj_factor' in indexed.columns:
        aggregations['adj_factor'] = 'last'

    if normalized == '240m':
        # A-share "4h/240m" is one full trading-day bar (240 trading minutes),
        # not two clock-based 4-hour buckets split by the lunch break.
        rows = []
        for _, group in indexed.groupby(indexed.index.normalize(), sort=True):
            if group.empty:
                continue
            row = {
                'time': group.index.max(),
                'open': group['open'].iloc[0],
                'high': group['high'].max(),
                'low': group['low'].min(),
                'close': group['close'].iloc[-1],
                'volume': group['volume'].sum(),
            }
            if 'amount' in group.columns:
                row['amount'] = group['amount'].sum()
            if 'turnover' in group.columns:
                row['turnover'] = group['turnover'].iloc[-1]
            if 'adj_factor' in group.columns:
                row['adj_factor'] = group['adj_factor'].iloc[-1]
            rows.append(row)
        resampled = pd.DataFrame(rows)
        if 'volume' in resampled.columns:
            resampled['volume'] = resampled['volume'].fillna(0.0)
        return resampled

    resampled = indexed.resample(rule).agg(aggregations)
    resampled = resampled.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
    if 'volume' in resampled.columns:
        resampled['volume'] = resampled['volume'].fillna(0.0)
    return resampled


def detect_ifind_sdk():
    for module_name in ('iFinDPy', 'ifind', 'ths'):
        try:
            module = importlib.import_module(module_name)
            return {
                'available': True,
                'module_name': module_name,
                'module_path': getattr(module, '__file__', None),
            }
        except ModuleNotFoundError:
            continue
    return {
        'available': False,
        'module_name': None,
        'module_path': None,
    }


def to_baostock_symbol(asset_code):
    code = str(asset_code).strip()
    if not re.fullmatch(r'\d{6}', code):
        raise ValueError(f'Unsupported A-share code: {asset_code}')
    if code.startswith(('5', '6', '9')):
        return f'sh.{code}'
    return f'sz.{code}'


def _normalize_akshare_frame(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    将 AkShare 返回的行情表转换为统一 OHLCV 格式（time/open/high/low/close/volume/amount）。
    AkShare 的列名通常为中文（时间/开盘/收盘/最高/最低/成交量/成交额/换手率...）。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    normalized = normalize_ohlcv_dataframe(df)
    if timeframe == '1d':
        normalized['time'] = normalized['time'].dt.normalize()
    return normalized


def fetch_akshare_ashare_history(
    asset_codes,
    output_dir,
    timeframes=None,
    daily_start='2018-01-01',
    minute_start='2021-01-01',
    end_date=None,
    overwrite=False,
    pause_seconds=0.35,
):
    """
    使用 AkShare 作为公开数据源拉取 A 股行情（用于 GitHub-only 场景的数据自助生成）。
    - 日线：ak.stock_zh_a_hist
    - 分钟线：ak.stock_zh_a_hist_min_em（period=5/15/60）
    """
    ak = importlib.import_module('akshare')
    timeframes = [normalize_timeframe_label(item) for item in (ensure_list(timeframes) or DEFAULT_ASHARE_TIMEFRAMES)]
    end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    os.makedirs(output_dir, exist_ok=True)

    minute_period_map = {'5m': '5', '15m': '15', '60m': '60'}
    daily_start_compact = pd.Timestamp(daily_start).strftime('%Y%m%d')
    daily_end_compact = pd.Timestamp(end_date).strftime('%Y%m%d')
    minute_start_ts = pd.Timestamp(minute_start).normalize()
    minute_end_ts = pd.Timestamp(end_date).normalize()

    written_files = []
    skipped_files = []
    for asset_code in ensure_list(asset_codes):
        code = str(asset_code).strip()
        for timeframe in timeframes:
            output_path = os.path.join(output_dir, canonical_training_filename(code, timeframe))
            if os.path.exists(output_path) and not overwrite and os.path.getsize(output_path) > 128:
                skipped_files.append({'asset_code': code, 'timeframe': timeframe, 'reason': 'existing_file'})
                continue

            try:
                if timeframe == '1d':
                    frame = None
                    last_exc = None
                    for attempt in range(1, 6):
                        try:
                            frame = ak.stock_zh_a_hist(
                                symbol=code,
                                period='daily',
                                start_date=daily_start_compact,
                                end_date=daily_end_compact,
                                adjust='',
                            )
                            break
                        except Exception as exc:
                            last_exc = exc
                            time.sleep(min(2.0 * attempt, 12.0))
                    if frame is None and last_exc is not None:
                        raise last_exc
                else:
                    period = minute_period_map.get(timeframe)
                    if period is None:
                        skipped_files.append({'asset_code': code, 'timeframe': timeframe, 'reason': 'unsupported_timeframe'})
                        continue
                    frames = []
                    for chunk_start, chunk_end in _iter_year_chunks(minute_start_ts.strftime('%Y-%m-%d'), minute_end_ts.strftime('%Y-%m-%d')):
                        start_date = f'{chunk_start} 09:30:00'
                        end_date_str = f'{chunk_end} 15:00:00'
                        print(f'[AKSHARE] {code} {timeframe} chunk {start_date} -> {end_date_str}')
                        chunk_frame = None
                        last_exc = None
                        for attempt in range(1, 6):
                            try:
                                chunk_frame = ak.stock_zh_a_hist_min_em(
                                    symbol=code,
                                    start_date=start_date,
                                    end_date=end_date_str,
                                    period=period,
                                    adjust='',
                                )
                                break
                            except Exception as exc:
                                last_exc = exc
                                time.sleep(min(2.0 * attempt, 12.0))
                        if chunk_frame is None and last_exc is not None:
                            raise last_exc
                        if chunk_frame is not None and not chunk_frame.empty:
                            frames.append(chunk_frame)
                        time.sleep(pause_seconds)
                    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                normalized = _normalize_akshare_frame(frame, timeframe)
                if normalized.empty:
                    skipped_files.append({'asset_code': code, 'timeframe': timeframe, 'reason': 'empty_result'})
                    continue
                normalized.to_csv(output_path, index=False, encoding='utf-8')
                written_files.append({
                    'asset_code': code,
                    'timeframe': timeframe,
                    'path': output_path,
                    'rows': int(len(normalized)),
                    'start_time': normalized['time'].iloc[0].isoformat(),
                    'end_time': normalized['time'].iloc[-1].isoformat(),
                    'source': 'akshare',
                })
            except Exception as exc:
                skipped_files.append({'asset_code': code, 'timeframe': timeframe, 'reason': repr(exc)})
            time.sleep(pause_seconds)

    return {
        'written_files': written_files,
        'skipped_files': skipped_files,
        'written_count': len(written_files),
        'source': 'akshare',
    }


def fetch_public_ashare_history(
    asset_codes,
    output_dir,
    timeframes=None,
    daily_start='2018-01-01',
    minute_start='2021-01-01',
    end_date=None,
    overwrite=False,
    pause_seconds=0.2,
    baostock_retries: int = 3,
):
    """
    公开数据兜底：优先 BaoStock（与既有实现保持一致），失败后自动降级到 AkShare。
    """
    last_error = None
    for attempt in range(1, baostock_retries + 1):
        try:
            return fetch_baostock_ashare_history(
                asset_codes=asset_codes,
                output_dir=output_dir,
                timeframes=timeframes,
                daily_start=daily_start,
                minute_start=minute_start,
                end_date=end_date,
                overwrite=overwrite,
                pause_seconds=pause_seconds,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(min(5.0 * attempt, 20.0))

    try:
        return fetch_akshare_ashare_history(
            asset_codes=asset_codes,
            output_dir=output_dir,
            timeframes=timeframes,
            daily_start=daily_start,
            minute_start=minute_start,
            end_date=end_date,
            overwrite=overwrite,
            pause_seconds=max(pause_seconds, 0.35),
        )
    except Exception as exc:
        raise RuntimeError(f'public fetch failed: baostock_error={last_error!r}, akshare_error={exc!r}') from exc


def _iter_year_chunks(start_date, end_date):
    current = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    while current <= end_ts:
        chunk_end = min(pd.Timestamp(year=current.year, month=12, day=31), end_ts)
        yield current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')
        current = chunk_end + pd.Timedelta(days=1)


def _query_baostock_rows(bs_module, symbol, frequency, start_date, end_date):
    if frequency == 'd':
        fields = 'date,code,open,high,low,close,volume,amount'
    else:
        fields = 'date,time,code,open,high,low,close,volume,amount'
    rs = bs_module.query_history_k_data_plus(
        symbol,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        adjustflag='2',
    )
    if rs.error_code != '0':
        raise RuntimeError(f'baostock query failed: {rs.error_code} {rs.error_msg}')

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=[field.strip() for field in fields.split(',')])
    return pd.DataFrame(rows, columns=[field.strip() for field in fields.split(',')])


def _normalize_baostock_frame(df, timeframe):
    if df.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    normalized = df.copy()
    if 'time' in normalized.columns:
        normalized['time'] = pd.to_datetime(normalized['time'].str.slice(0, 14), format='%Y%m%d%H%M%S', errors='coerce')
    else:
        normalized['time'] = pd.to_datetime(normalized['date'], errors='coerce')
    for column in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors='coerce')
    normalized = normalized[['time', 'open', 'high', 'low', 'close', 'volume', 'amount']].dropna(subset=['time', 'open', 'high', 'low', 'close'])
    normalized = normalized.sort_values('time').drop_duplicates(subset=['time'], keep='last').reset_index(drop=True)
    if timeframe == '1d':
        normalized['time'] = normalized['time'].dt.normalize()
    return normalized


def fetch_baostock_ashare_history(
    asset_codes,
    output_dir,
    timeframes=None,
    daily_start='2018-01-01',
    minute_start='2021-01-01',
    end_date=None,
    overwrite=False,
    pause_seconds=0.2,
):
    bs_module = importlib.import_module('baostock')
    timeframes = [normalize_timeframe_label(item) for item in (ensure_list(timeframes) or DEFAULT_ASHARE_TIMEFRAMES)]
    end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    os.makedirs(output_dir, exist_ok=True)

    frequency_map = {'5m': '5', '15m': '15', '60m': '60', '1d': 'd'}
    login_result = None
    for attempt in range(1, 6):
        login_result = bs_module.login()
        if login_result.error_code == '0':
            break
        time.sleep(min(3.0 * attempt, 15.0))
    if login_result is None or login_result.error_code != '0':
        raise RuntimeError(f'baostock login failed: {login_result.error_code} {login_result.error_msg}')

    written_files = []
    skipped_files = []
    try:
        for asset_code in ensure_list(asset_codes):
            symbol = to_baostock_symbol(asset_code)
            for timeframe in timeframes:
                output_path = os.path.join(output_dir, canonical_training_filename(asset_code, timeframe))
                if os.path.exists(output_path) and not overwrite and os.path.getsize(output_path) > 128:
                    skipped_files.append({'asset_code': asset_code, 'timeframe': timeframe, 'reason': 'existing_file'})
                    continue

                if timeframe not in frequency_map:
                    skipped_files.append({'asset_code': asset_code, 'timeframe': timeframe, 'reason': 'unsupported_timeframe'})
                    continue

                if timeframe == '1d':
                    frame = _query_baostock_rows(bs_module, symbol, frequency_map[timeframe], daily_start, end_date)
                else:
                    frames = []
                    for chunk_start, chunk_end in _iter_year_chunks(minute_start, end_date):
                        chunk_frame = _query_baostock_rows(bs_module, symbol, frequency_map[timeframe], chunk_start, chunk_end)
                        if not chunk_frame.empty:
                            frames.append(chunk_frame)
                        time.sleep(pause_seconds)
                    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                normalized = _normalize_baostock_frame(frame, timeframe)
                if normalized.empty:
                    skipped_files.append({'asset_code': asset_code, 'timeframe': timeframe, 'reason': 'empty_result'})
                    continue
                normalized.to_csv(output_path, index=False, encoding='utf-8')
                written_files.append({
                    'asset_code': asset_code,
                    'timeframe': timeframe,
                    'path': output_path,
                    'rows': int(len(normalized)),
                    'start_time': normalized['time'].iloc[0].isoformat(),
                    'end_time': normalized['time'].iloc[-1].isoformat(),
                })
                time.sleep(pause_seconds)
    finally:
        try:
            bs_module.logout()
        except Exception:
            pass

    return {
        'written_files': written_files,
        'skipped_files': skipped_files,
        'written_count': len(written_files),
    }


def prepare_imported_ashare_data(import_dir, normalized_dir, training_ready_dir, assets=None, target_timeframes=None):
    asset_filter = {item for item in ensure_list(assets)}
    target_timeframes = [normalize_timeframe_label(item) for item in (ensure_list(target_timeframes) or DEFAULT_ASHARE_TIMEFRAMES)]
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(training_ready_dir, exist_ok=True)

    generated = {}
    source_files = []
    skipped = []
    for root, _, files in os.walk(import_dir):
        for name in files:
            if name.lower().endswith('.csv'):
                source_files.append(os.path.join(root, name))
    source_files.sort()

    def _source_preference_key(source_timeframe, target_timeframe):
        source = normalize_timeframe_label(source_timeframe)
        target = normalize_timeframe_label(target_timeframe)
        source_minutes = TIMEFRAME_TO_MINUTES.get(source)
        target_minutes = TIMEFRAME_TO_MINUTES.get(target)
        if source_minutes is None or target_minutes is None or source_minutes > target_minutes:
            return None
        if source == target:
            return (0, 0, source_minutes)
        return (1, target_minutes - source_minutes, -source_minutes)

    for file_path in source_files:
        try:
            raw_df = read_csv_with_fallback(file_path)
            normalized_df = normalize_ohlcv_dataframe(raw_df)
            asset_code = infer_asset_code_from_filename(file_path)
            if asset_filter and asset_code not in asset_filter:
                continue
            timeframe = infer_timeframe_from_filename(file_path) or infer_timeframe_from_dataframe(normalized_df)
            if timeframe is None:
                skipped.append({'path': file_path, 'reason': 'unable_to_infer_timeframe'})
                continue

            normalized_name = canonical_training_filename(asset_code, timeframe)
            normalized_path = os.path.join(normalized_dir, normalized_name)
            normalized_df.to_csv(normalized_path, index=False)

            source_minutes = TIMEFRAME_TO_MINUTES.get(timeframe)
            for target_timeframe in target_timeframes:
                target_minutes = TIMEFRAME_TO_MINUTES[target_timeframe]
                if source_minutes is None or source_minutes > target_minutes:
                    continue
                priority = _source_preference_key(timeframe, target_timeframe)
                if priority is None:
                    continue
                output_df = normalized_df if timeframe == target_timeframe else resample_ohlcv_dataframe(normalized_df, target_timeframe)
                if output_df.empty:
                    continue
                target_key = (asset_code, target_timeframe)
                current = generated.get(target_key)
                if current is None or priority < current['priority']:
                    generated[target_key] = {
                        'df': output_df,
                        'source_minutes': source_minutes,
                        'source_timeframe': timeframe,
                        'priority': priority,
                        'source_path': file_path,
                    }
        except Exception as exc:
            skipped.append({'path': file_path, 'reason': repr(exc)})

    written_files = []
    for (asset_code, timeframe), payload in sorted(generated.items()):
        output_name = canonical_training_filename(asset_code, timeframe)
        output_path = os.path.join(training_ready_dir, output_name)
        payload['df'].to_csv(output_path, index=False)
        written_files.append({
            'path': output_path,
            'asset_code': asset_code,
            'timeframe': timeframe,
            'source_path': payload['source_path'],
        })

    return {
        'source_file_count': len(source_files),
        'generated_file_count': len(written_files),
        'written_files': written_files,
        'skipped_files': skipped,
    }


def collect_coverage_records(file_records, train_end=None, val_end=None, test_start=None):
    train_end_ts = pd.Timestamp(train_end) if train_end else None
    val_end_ts = pd.Timestamp(val_end) if val_end else None
    test_start_ts = pd.Timestamp(test_start) if test_start else None
    coverage_records = []

    for record in file_records:
        path = record['path']
        try:
            df = normalize_ohlcv_dataframe(read_csv_with_fallback(path))
            row_count = int(len(df))
            start_time = df['time'].iloc[0].isoformat() if row_count else None
            end_time = df['time'].iloc[-1].isoformat() if row_count else None
            if train_end_ts is not None:
                has_train = bool((df['time'] <= train_end_ts).any())
            else:
                has_train = row_count > 0
            if train_end_ts is not None and val_end_ts is not None:
                has_val = bool(((df['time'] > train_end_ts) & (df['time'] <= val_end_ts)).any())
            else:
                has_val = row_count > 0
            if test_start_ts is not None:
                has_test = bool((df['time'] >= test_start_ts).any())
            else:
                has_test = row_count > 0
            coverage_records.append({
                'path': path,
                'asset_code': record['asset_code'],
                'timeframe': record['timeframe'],
                'rows': row_count,
                'start_time': start_time,
                'end_time': end_time,
                'has_train': has_train,
                'has_val': has_val,
                'has_test': has_test,
                'complete': row_count > 0 and has_train and has_val and has_test,
            })
        except Exception as exc:
            coverage_records.append({
                'path': path,
                'asset_code': record.get('asset_code'),
                'timeframe': record.get('timeframe'),
                'rows': 0,
                'start_time': None,
                'end_time': None,
                'has_train': False,
                'has_val': False,
                'has_test': False,
                'complete': False,
                'error': repr(exc),
            })
    coverage_records.sort(key=lambda item: (item.get('asset_code') or '', item.get('timeframe') or ''))
    return coverage_records


def resolve_asset_universe(coverage_records, primary_assets=None, fallback_assets=None, timeframes=None):
    primary_assets = ensure_list(primary_assets) or list(ASHARE_PRIMARY_ASSETS)
    fallback_assets = ensure_list(fallback_assets) or list(ASHARE_FALLBACK_ASSETS)
    timeframes = [normalize_timeframe_label(item) for item in (ensure_list(timeframes) or DEFAULT_ASHARE_TIMEFRAMES)]

    completeness = defaultdict(dict)
    for record in coverage_records:
        completeness[record['asset_code']][record['timeframe']] = bool(record.get('complete'))

    complete_assets = {
        asset_code
        for asset_code, per_timeframe in completeness.items()
        if all(per_timeframe.get(timeframe, False) for timeframe in timeframes)
    }

    selected_assets = [asset for asset in primary_assets if asset in complete_assets]
    missing_primary_assets = [asset for asset in primary_assets if asset not in complete_assets]
    fallback_used = []
    for asset in fallback_assets:
        if len(selected_assets) + len(fallback_used) >= len(primary_assets):
            break
        if asset in complete_assets and asset not in selected_assets and asset not in fallback_used:
            fallback_used.append(asset)

    final_assets = selected_assets + fallback_used
    return {
        'selected_assets': final_assets,
        'missing_primary_assets': missing_primary_assets,
        'fallback_used': fallback_used,
        'complete_assets': sorted(complete_assets),
        'sufficient': len(final_assets) >= len(primary_assets),
    }


def build_market_coverage_report(
    data_dir,
    market='ashare',
    primary_assets=None,
    fallback_assets=None,
    timeframes=None,
    train_end=None,
    val_end=None,
    test_start=None,
    training_subdir=None,
):
    primary_assets = ensure_list(primary_assets) or list(ASHARE_PRIMARY_ASSETS)
    fallback_assets = ensure_list(fallback_assets) or list(ASHARE_FALLBACK_ASSETS)
    timeframes = [normalize_timeframe_label(item) for item in (ensure_list(timeframes) or DEFAULT_ASHARE_TIMEFRAMES)]
    all_assets = primary_assets + [asset for asset in fallback_assets if asset not in primary_assets]
    file_records = discover_training_files(
        data_dir=data_dir,
        market=market,
        assets=all_assets,
        timeframes=timeframes,
        training_subdir=training_subdir,
    )
    coverage_records = collect_coverage_records(
        file_records=file_records,
        train_end=train_end,
        val_end=val_end,
        test_start=test_start,
    )
    asset_resolution = resolve_asset_universe(
        coverage_records=coverage_records,
        primary_assets=primary_assets,
        fallback_assets=fallback_assets,
        timeframes=timeframes,
    )
    missing_combinations = []
    index = {(item['asset_code'], item['timeframe']): item for item in coverage_records}
    for asset_code in all_assets:
        for timeframe in timeframes:
            record = index.get((asset_code, timeframe))
            if record is None:
                missing_combinations.append({
                    'asset_code': asset_code,
                    'timeframe': timeframe,
                    'reason': 'missing_file',
                })
            elif not record['complete']:
                reason = 'insufficient_split_coverage'
                if record.get('error'):
                    reason = record['error']
                missing_combinations.append({
                    'asset_code': asset_code,
                    'timeframe': timeframe,
                    'reason': reason,
                })

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'market': market,
        'timeframes': timeframes,
        'train_end': train_end,
        'val_end': val_end,
        'test_start': test_start,
        'file_count': len(file_records),
        'coverage_records': coverage_records,
        'asset_resolution': asset_resolution,
        'missing_combinations': missing_combinations,
    }


def write_coverage_reports(report_dir, report_name, report):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, f'{report_name}.json')
    md_path = os.path.join(report_dir, f'{report_name}.md')

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    lines = [
        f"# {report_name}",
        '',
        f"- 生成时间: `{report.get('generated_at')}`",
        f"- 市场: `{report.get('market')}`",
        f"- 文件数: `{report.get('file_count')}`",
        f"- 时间切分: train<=`{report.get('train_end')}`, val<=`{report.get('val_end')}`, test>=`{report.get('test_start')}`",
        '',
        '## 标的池解析',
        '',
        f"- 可直接训练: `{len(report.get('asset_resolution', {}).get('selected_assets', []))}` 只",
        f"- 主标的缺失: `{', '.join(report.get('asset_resolution', {}).get('missing_primary_assets', [])) or '无'}`",
        f"- 备选补位: `{', '.join(report.get('asset_resolution', {}).get('fallback_used', [])) or '无'}`",
        f"- 是否满足正式训练: `{report.get('asset_resolution', {}).get('sufficient')}`",
        '',
        '## 缺失项',
        '',
    ]
    if report.get('missing_combinations'):
        for item in report['missing_combinations']:
            lines.append(f"- `{item['asset_code']}` / `{item['timeframe']}`: {item['reason']}")
    else:
        lines.append('- 无')

    lines.extend(['', '## 文件覆盖', ''])
    for item in report.get('coverage_records', []):
        lines.append(
            f"- `{item['asset_code']}` / `{item['timeframe']}` | rows={item['rows']} | "
            f"{item['start_time']} -> {item['end_time']} | "
            f"train/val/test={item['has_train']}/{item['has_val']}/{item['has_test']}"
        )

    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    return {
        'json_path': json_path,
        'md_path': md_path,
    }
