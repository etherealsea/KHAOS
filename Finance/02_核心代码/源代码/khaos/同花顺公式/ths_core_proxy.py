from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from khaos.数据处理.ashare_support import normalize_ohlcv_dataframe


@dataclass(frozen=True)
class THSCoreParams:
    n: int = 20
    eps: float = 1e-6
    bk_evt_th: float = 2.336719
    rv_evt_th: float = 1.185428
    res_node: float = -0.005
    ent_bull: float = 0.58
    ent_bear: float = 0.64
    mle_gate: float = 0.05
    h_trend: float = 0.55
    dir_gap: float = 0.18
    bk_rule_weight: float = 1.10
    bk_comp_weight: float = 1.15
    bk_mle_weight: float = 0.8625
    bk_trend_weight: float = 0.6325
    bk_press_weight: float = 0.5175
    bk_ent_turn_weight: float = 0.4025
    bk_vol_quiet_weight: float = 0.35
    bk_soft_scale: float = 0.72
    bk_comp_gate: float = 0.01
    cont_press_mle_weight: float = 0.60
    cont_press_press_weight: float = 0.55
    cont_press_comp_weight: float = 0.45
    cont_press_trend_weight: float = 0.35
    cont_press_ent_turn_weight: float = 0.20
    uprev_rev_setup_weight: float = 1.15
    uprev_ent_weight: float = 0.4675
    uprev_confirm_weight: float = 0.30
    uprev_ent_rise_weight: float = 0.2875
    uprev_trend_weight: float = 0.20
    dnrev_rev_setup_weight: float = 1.15
    dnrev_ent_weight: float = 0.3825
    dnrev_mom_weight: float = 0.4025
    dnrev_confirm_weight: float = 0.25
    dnrev_trend_weight: float = 0.15
    uprev_bk_penalty: float = 0.2975
    uprev_cont_press_penalty: float = 0.25
    dnrev_bk_penalty: float = 0.2975
    dnrev_cont_press_penalty: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)

    def updated(self, **kwargs) -> 'THSCoreParams':
        return replace(self, **kwargs)


def load_ths_core_params(params_path: str | Path) -> THSCoreParams:
    payload = json.loads(Path(params_path).read_text(encoding='utf-8'))
    kwargs = {}
    for key, value in payload.items():
        if not hasattr(THSCoreParams, key):
            continue
        if key == 'n':
            kwargs[key] = int(value)
        else:
            kwargs[key] = float(value) if isinstance(value, (int, float)) else value
    return THSCoreParams(**kwargs)


def dump_ths_core_params(params: THSCoreParams, params_path: str | Path) -> None:
    path = Path(params_path)
    path.write_text(
        json.dumps(params.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding='utf-8',
    )


DEFAULT_THS_CORE_PARAMS = THSCoreParams()
_DEFAULT_PARAMS_PATH = Path(__file__).with_name('params.json')
if _DEFAULT_PARAMS_PATH.exists():
    try:
        DEFAULT_THS_CORE_PARAMS = load_ths_core_params(_DEFAULT_PARAMS_PATH)
    except Exception:
        # 保持向后兼容：若 params.json 不可用，则回退到 dataclass 默认值
        DEFAULT_THS_CORE_PARAMS = THSCoreParams()

PHASE_NAME_MAP = {
    -2: 'bull_reversion',
    0: 'neutral',
    1: 'breakout',
    2: 'bear_reversion',
}

TUNABLE_THS_FIELDS = (
    'bk_evt_th',
    'rv_evt_th',
    'res_node',
    'ent_bull',
    'ent_bear',
    'mle_gate',
    'dir_gap',
    'bk_rule_weight',
    'bk_comp_weight',
    'bk_mle_weight',
    'bk_trend_weight',
    'bk_press_weight',
    'bk_ent_turn_weight',
    'bk_vol_quiet_weight',
    'cont_press_mle_weight',
    'cont_press_press_weight',
    'cont_press_comp_weight',
    'cont_press_trend_weight',
    'cont_press_ent_turn_weight',
    'uprev_rev_setup_weight',
    'uprev_ent_weight',
    'uprev_confirm_weight',
    'uprev_ent_rise_weight',
    'uprev_trend_weight',
    'dnrev_rev_setup_weight',
    'dnrev_ent_weight',
    'dnrev_mom_weight',
    'dnrev_confirm_weight',
    'dnrev_trend_weight',
    'uprev_bk_penalty',
    'uprev_cont_press_penalty',
    'dnrev_bk_penalty',
    'dnrev_cont_press_penalty',
)


def phase_value_to_name(value: int) -> str:
    return PHASE_NAME_MAP.get(int(value), 'unknown')


def normalize_phase_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=np.float64))


def _shift(series: pd.Series, periods: int = 1) -> pd.Series:
    shifted = series.shift(periods)
    if len(series) == 0:
        return shifted.fillna(0.0)
    return shifted.fillna(float(series.iloc[0]))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).sum()


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).max()


def _rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).min()


def _clip(values, lower=None, upper=None):
    array = np.asarray(values, dtype=np.float64)
    if lower is not None:
        array = np.maximum(array, lower)
    if upper is not None:
        array = np.minimum(array, upper)
    return array


def _ensure_ohlcv_frame(df: pd.DataFrame, normalize_input: bool = True) -> pd.DataFrame:
    if normalize_input:
        return normalize_ohlcv_dataframe(df)
    return df.copy()


def compute_ths_core_frame(
    df: pd.DataFrame,
    params: THSCoreParams = DEFAULT_THS_CORE_PARAMS,
    normalize_input: bool = True,
) -> pd.DataFrame:
    frame = _ensure_ohlcv_frame(df, normalize_input=normalize_input)
    out = frame.copy()

    close = _series(out['close'].values)
    high = _series(out['high'].values)
    low = _series(out['low'].values)
    n = params.n
    eps = params.eps

    lc = np.log(np.maximum(close.values, 0.01))
    lc_s = _series(lc)
    ret = lc_s - _shift(lc_s, 1)
    vol = ret.abs()
    sigma = _rolling_std(ret, n).clip(lower=eps)
    sigma_ref = _ema(sigma, n)

    ema20 = _ema(close, 20)
    log_ema20 = np.log(np.maximum(ema20.values, 0.01))
    ema_gap = lc_s - _series(log_ema20)
    ema_div = (close - ema20) / (ema20 + eps)

    hmax = _rolling_max(lc_s, n)
    hmin = _rolling_min(lc_s, n)
    hstd = _rolling_std(lc_s, n)
    rsv = np.where(hstd.values > eps, (hmax.values - hmin.values) / (hstd.values + eps), 1.0)
    hbase = np.log(rsv + eps) - 0.5 * np.log(float(n))
    hx = _clip(4.0 * hbase, lower=-8.0, upper=8.0)
    htanh = np.tanh(hx)
    hurst = _clip(0.5 + 0.3 * htanh, lower=0.0, upper=1.0)
    rho_input = _clip(-10.0 * (hurst - 0.5), lower=-20.0, upper=20.0)
    rho_raw = 0.5 + 0.5 / (1.0 + np.exp(rho_input))

    ekf_track = _ema(lc_s, 19)
    ekf_res = lc_s - ekf_track
    ekf_vel = ekf_track - _shift(ekf_track, 1)

    price_vel = close - _shift(close, 1)
    ekf_pred = close.astype(np.float64).copy()
    if len(ekf_pred) >= 3:
        ekf_pred.iloc[2:] = (
            _shift(close, 1).iloc[2:]
            + _shift(_series(rho_raw), 1).iloc[2:] * _shift(price_vel, 1).iloc[2:]
        )

    abr = vol + eps
    abr1 = _shift(abr, 1) + eps
    mle = _ema(_series(np.log((abr.values + 0.0) / abr1.values)), n)

    prev_close = _shift(close, 1)
    tr1 = np.maximum.reduce([
        (high - low).values,
        (high - prev_close).abs().values,
        (low - prev_close).abs().values,
    ])
    tr1_s = _series(tr1)
    sum_tr = _rolling_sum(tr1_s, n)
    rng = _rolling_max(high, n) - _rolling_min(low, n)
    entropy_ratio = np.zeros(len(out), dtype=np.float64)
    entropy_mask = (rng.values > eps) & (sum_tr.values > eps)
    np.divide(
        sum_tr.values,
        rng.values,
        out=entropy_ratio,
        where=entropy_mask,
    )
    ent_raw = np.where(entropy_mask, np.log10(entropy_ratio + eps), 0.0)
    ent_raw_s = _series(ent_raw)
    ent = _ema(ent_raw_s, 2)
    ent_ref = _ema(ent, n)
    low_ent = np.maximum(ent_ref.values - ent.values, 0.0)
    ent_d = ent - _shift(ent, 1)
    ent_dd = ent_d - _shift(ent_d, 1)

    vol_quiet = np.maximum((sigma_ref.values - sigma.values) / (sigma_ref.values + eps), 0.0)
    comp = vol_quiet * low_ent

    trend_edge = np.maximum(hurst - params.h_trend, 0.0)
    trend_n = np.minimum(trend_edge / 0.05, 4.0)
    mle_n = np.minimum(np.maximum(mle.values - params.mle_gate, 0.0) / 0.08, 4.0)
    press_n = np.minimum(np.maximum(np.abs(ret.values) / (sigma.values + eps) - 0.35, 0.0), 4.0)
    ent_turn_n = np.minimum(np.maximum(-ent_d.values, 0.0) / 0.03 + np.maximum(ent_dd.values, 0.0) / 0.03, 4.0)
    vol_quiet_n = np.minimum(vol_quiet / 0.10, 4.0)
    comp_n = np.minimum(comp / 0.03, 4.0)
    cont_press = (
        params.cont_press_mle_weight * mle_n +
        params.cont_press_press_weight * press_n +
        params.cont_press_comp_weight * comp_n +
        params.cont_press_trend_weight * trend_n +
        params.cont_press_ent_turn_weight * ent_turn_n
    )

    bk_rule = (vol.values <= sigma_ref.values) & (ekf_res.values > params.res_node) & (ekf_res.values <= 0.0)
    bk_score_raw = (
        params.bk_rule_weight * bk_rule.astype(np.float64) +
        params.bk_comp_weight * comp_n +
        params.bk_mle_weight * mle_n +
        params.bk_trend_weight * trend_n +
        params.bk_press_weight * press_n +
        params.bk_ent_turn_weight * ent_turn_n +
        params.bk_vol_quiet_weight * vol_quiet_n
    )
    bk_score = np.maximum(
        np.where(bk_rule | (comp > params.bk_comp_gate), bk_score_raw, bk_score_raw * params.bk_soft_scale),
        0.0,
    )

    res_score = np.abs(ekf_res.values) / (sigma.values + eps)
    ema_score = np.abs(ema_gap.values) / (sigma.values + eps)
    align = np.abs(ekf_res.values + ema_gap.values) / (np.abs(ekf_res.values) + np.abs(ema_gap.values) + eps)
    rev_setup = align * np.maximum(res_score - 1.0, 0.0) * np.maximum(ema_score - 0.50, 0.0)
    rev_setup_n = np.minimum(rev_setup, 8.0)

    uprev_ent_n = np.minimum(np.maximum(params.ent_bull - ent.values, 0.0) / 0.08, 4.0)
    dnrev_ent_n = np.minimum(np.maximum(params.ent_bear - ent.values, 0.0) / 0.08, 4.0)
    uprev_confirm_n = np.minimum(np.maximum(ekf_vel.values, 0.0) / (sigma.values + eps), 4.0)
    dnrev_confirm_n = np.minimum(np.maximum(-ekf_vel.values, 0.0) / (sigma.values + eps), 4.0)
    dnrev_mom_n = np.minimum(np.maximum(ret.values, 0.0) / (sigma.values + eps), 4.0)
    ent_rise_n = np.minimum(np.maximum(ent_d.values, 0.0) / 0.02, 4.0)

    uprev_rule = (ema_div.values <= 0.0) & (ekf_res.values <= params.res_node) & (ent.values <= params.ent_bull)
    dnrev_rule = (ema_div.values > 0.0) & ((ent.values <= params.ent_bear) | (ret.values > 0.0))

    uprev_raw_base = np.where(
        uprev_rule,
        params.uprev_rev_setup_weight * rev_setup_n +
        params.uprev_ent_weight * uprev_ent_n +
        params.uprev_confirm_weight * uprev_confirm_n +
        params.uprev_ent_rise_weight * ent_rise_n +
        params.uprev_trend_weight * trend_n,
        0.0,
    )
    dnrev_raw_base = np.where(
        dnrev_rule,
        params.dnrev_rev_setup_weight * rev_setup_n +
        params.dnrev_ent_weight * dnrev_ent_n +
        params.dnrev_mom_weight * dnrev_mom_n +
        params.dnrev_confirm_weight * dnrev_confirm_n +
        params.dnrev_trend_weight * trend_n,
        0.0,
    )
    uprev_raw = np.maximum(
        uprev_raw_base - params.uprev_bk_penalty * bk_score - params.uprev_cont_press_penalty * cont_press,
        0.0,
    )
    dnrev_raw = np.maximum(
        dnrev_raw_base - params.dnrev_bk_penalty * bk_score - params.dnrev_cont_press_penalty * cont_press,
        0.0,
    )

    bk_dom = bk_score
    bear_dom = dnrev_raw
    bull_dom = uprev_raw

    bk_on = (bk_dom >= params.bk_evt_th) & (bk_dom >= bear_dom) & (bk_dom >= bull_dom)
    bear_on = (bear_dom >= params.rv_evt_th) & ((bear_dom - bull_dom) >= params.dir_gap) & (bear_dom > bk_dom)
    bull_on = (bull_dom >= params.rv_evt_th) & ((bull_dom - bear_dom) >= params.dir_gap) & (bull_dom > bk_dom)

    phase = np.where(bk_on, 1, np.where(bear_on, 2, np.where(bull_on, -2, 0))).astype(np.int32)
    out['LC'] = lc
    out['RET'] = ret.values
    out['VOL'] = vol.values
    out['SIGMA'] = sigma.values
    out['SIGMA_REF'] = sigma_ref.values
    out['EMA20'] = ema20.values
    out['LOG_EMA20'] = log_ema20
    out['EMA_GAP'] = ema_gap.values
    out['EMA_DIV'] = ema_div.values
    out['HURST'] = hurst
    out['RHO_RAW'] = rho_raw
    out['EKF_TRACK'] = ekf_track.values
    out['EKF_RES'] = ekf_res.values
    out['EKF_VEL'] = ekf_vel.values
    out['PRICE_VEL'] = price_vel.values
    out['EKF_PRED'] = ekf_pred.values
    out['MLE'] = mle.values
    out['ENT'] = ent.values
    out['ENT_REF'] = ent_ref.values
    out['ENT_D'] = ent_d.values
    out['ENT_DD'] = ent_dd.values
    out['LOW_ENT'] = low_ent
    out['VOL_QUIET'] = vol_quiet
    out['COMP'] = comp
    out['TREND_N'] = trend_n
    out['MLE_N'] = mle_n
    out['PRESS_N'] = press_n
    out['ENT_TURN_N'] = ent_turn_n
    out['VOL_QUIET_N'] = vol_quiet_n
    out['COMP_N'] = comp_n
    out['CONT_PRESS'] = cont_press
    out['BK_RULE'] = bk_rule.astype(np.int32)
    out['BK_SCORE'] = bk_score
    out['REV_SETUP'] = rev_setup
    out['REV_SETUP_N'] = rev_setup_n
    out['UPREV_RULE'] = uprev_rule.astype(np.int32)
    out['DNREV_RULE'] = dnrev_rule.astype(np.int32)
    out['UPREV_RAW'] = uprev_raw
    out['DNREV_RAW'] = dnrev_raw
    out['BK_DOM'] = bk_dom
    out['BEAR_DOM'] = bear_dom
    out['BULL_DOM'] = bull_dom
    out['BK_ON'] = bk_on.astype(np.int32)
    out['BEAR_ON'] = bear_on.astype(np.int32)
    out['BULL_ON'] = bull_on.astype(np.int32)
    out['PHASE'] = phase
    return out


def extract_formula_constants(formula_text: str) -> dict:
    pattern = re.compile(r'^([A-Z_]+):=([-+]?\d+(?:\.\d+)?);$', re.MULTILINE)
    constants = {}
    for key, value in pattern.findall(formula_text):
        if key in {'N', 'BK_EVT_TH', 'RV_EVT_TH', 'RES_NODE', 'ENT_BULL', 'ENT_BEAR', 'MLE_GATE', 'H_TREND', 'DIR_GAP'}:
            constants[key] = float(value)
    return constants


def load_formula_constants(formula_path: str | Path) -> dict:
    return extract_formula_constants(Path(formula_path).read_text(encoding='utf-8-sig'))
