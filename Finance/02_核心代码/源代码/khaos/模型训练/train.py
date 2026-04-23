import argparse
import gc
import glob
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Sampler, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""

from khaos.鏁版嵁澶勭悊.ashare_dataset import (
    AshareFinancialDataset,
    EVENT_FLAG_INDEX,
    ROLLING_RECENT_SPLITS,
    build_global_horizon_grid,
    create_ashare_dataset_splits,
)
from khaos.鏁版嵁澶勭悊.ashare_support import (
    DEFAULT_ASHARE_TIMEFRAMES,
    DEFAULT_MULTI_ASSETS,
    discover_training_files,
    normalize_timeframe_label,
    resolve_training_ready_dir,
)
from khaos.鏁版嵁澶勭悊.data_loader import create_rolling_datasets
from khaos.鏁版嵁澶勭悊.data_processor import process_multi_timeframe
from khaos.妯″瀷瀹氫箟.kan import KHAOS_KAN
from khaos.妯″瀷璁粌.loss import PhysicsLoss
from khaos.鏍稿績寮曟搸.physics import PHYSICS_FEATURE_NAMES


"""
from khaos.数据处理.ashare_dataset import (
    EVENT_FLAG_INDEX,
    ROLLING_RECENT_SPLITS,
    build_global_horizon_grid,
    create_ashare_dataset_splits,
)
from khaos.数据处理.ashare_support import (
    DEFAULT_ASHARE_TIMEFRAMES,
    DEFAULT_MULTI_ASSETS,
    discover_training_files,
    normalize_timeframe_label,
    resolve_training_ready_dir,
)
from khaos.数据处理.data_loader import create_rolling_datasets
from khaos.数据处理.data_processor import process_multi_timeframe
from khaos.模型定义.kan import KHAOS_KAN
from khaos.模型训练.loss import PhysicsLoss
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES

DEBUG_BATCH_KEYS = (
    'bear_score',
    'bull_score',
    'direction_gate',
    'compression_gate',
    'directional_gate',
    'gate_floor_hit',
    'public_reversion_gate',
    'breakout_residual_gate',
    'directional_reversion',
    'directional_floor',
    'selected_horizon_breakout',
    'selected_horizon_reversion',
    'selected_horizon_breakout_value',
    'selected_horizon_reversion_value',
    'horizon_entropy_breakout',
    'horizon_entropy_reversion',
)

CONSTRAINT_STAT_BASE_KEYS = (
    'bear_over_bull_violation',
    'bull_over_bear_violation',
    'public_below_directional_violation',
    'continuation_public_violation',
)

RECENT_FOLD_ORDER = ('fold_3', 'fold_4')
DATASET_CACHE_VERSION = 2


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# SECTION: METRICS


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


PRECISION_FIRST_SCORE_PROFILES = {
    'short_t_precision_focus',
    'short_t_discovery_focus',
    'short_t_discovery_guarded_focus',
    'recent_precision_v1',
    'iter11_precision_first',
    'iter12_guarded_precision_first',
    'iter12_soft_guarded_precision_first',
    'iter13_precision_first',
    'iter14_precision_first',
}
AUX_GATED_SCORE_PROFILES = {
    'short_t_discovery_guarded_focus',
    'iter12_guarded_precision_first',
    'iter12_soft_guarded_precision_first',
}

PRECISION_FIRST_PROFILE_CONFIG = {
    'default': {
        'min_precision': 0.60,
        'max_hard_negative_rate': 0.15,
        'threshold_grid_min': 0.55,
        'threshold_grid_max': 0.99,
        'threshold_grid_points': 12,
    },
    'iter11_precision_first': {
        'min_precision': 0.70,
        'max_hard_negative_rate': 0.12,
        'threshold_grid_min': 0.75,
        'threshold_grid_max': 0.995,
        'threshold_grid_points': 12,
    },
    'iter12_guarded_precision_first': {
        'min_precision': 0.62,
        'max_hard_negative_rate': 0.16,
        'threshold_grid_min': 0.68,
        'threshold_grid_max': 0.995,
        'threshold_grid_points': 14,
    },
    'iter13_precision_first': {
        'min_precision': 0.0, # 移除不合理的精度下限限制
        'max_hard_negative_rate': 1.0,
        'threshold_grid_min': 0.20, # 下调分位数搜索下限，适应 Sigmoid 真实概率
        'threshold_grid_max': 0.95,
        'threshold_grid_points': 16,
    },
    'iter14_precision_first': {
        'min_precision': 0.0,
        'max_hard_negative_rate': 1.0,
        'threshold_grid_min': 0.85, # 适应EV回归，寻找最头部的15%以内的信号
        'threshold_grid_max': 0.9995, # 放宽到最极端的 0.05%，让稀疏的 Reversion 有机会达到合理的频率
        'threshold_grid_points': 200, # 提升搜索粒度到 200 步，精细捕捉极尾部分布
    },
}


def resolve_event_selection_mode(score_profile='default'):
    if score_profile in PRECISION_FIRST_SCORE_PROFILES:
        return 'precision_first'
    return 'f1'


def is_modern_score_profile(score_profile='default'):
    return str(score_profile or 'default') in ['iter12_soft_guarded_precision_first', 'iter13_precision_first', 'iter14_precision_first']


def resolve_event_oversignal_cap(label_frequency, event_type='generic', score_profile='default'):
    label_frequency = float(label_frequency)
    if score_profile == 'iter11_precision_first':
        return min(label_frequency, 0.05)
    if score_profile == 'iter12_guarded_precision_first':
        if event_type == 'reversion':
            return min(label_frequency + 0.01, 0.18)
        if event_type == 'breakout':
            return min(label_frequency + 0.02, 0.22)
        return min(label_frequency + 0.02, 0.20)
    if score_profile in ('iter12_soft_guarded_precision_first', 'iter13_precision_first', 'iter14_precision_first'):
        if event_type == 'reversion':
            return min(max(label_frequency * 1.6, label_frequency + 0.03, 0.03), 0.28)
        if event_type == 'breakout':
            return min(max(label_frequency * 1.6, label_frequency + 0.04, 0.04), 0.32)
        return min(max(label_frequency * 1.6, label_frequency + 0.04, 0.04), 0.30)
    if event_type == 'reversion':
        return min(label_frequency + 0.02, 0.32)
    if event_type == 'breakout':
        return min(label_frequency + 0.02, 0.46)
    return None


def compute_event_oversignal(metrics):
    return max(
        float(metrics.get('signal_frequency', 0.0)) - float(metrics.get('label_frequency', 0.0)),
        0.0,
    )


def compute_signal_health(signal_frequency, signal_cap, signal_floor=0.02):
    signal_floor = max(float(signal_floor), 1e-6)
    signal_frequency = max(float(signal_frequency), 0.0)
    if signal_cap is None:
        signal_cap = max(signal_floor * 2.0, 0.04)
    signal_cap = max(float(signal_cap), signal_floor + 1e-6)
    if signal_floor <= signal_frequency <= signal_cap:
        return 1.0
    if signal_frequency < signal_floor:
        return float(np.clip(signal_frequency / signal_floor, 0.0, 1.0))
    overflow = (signal_frequency - signal_cap) / max(signal_cap, 1e-6)
    return float(np.clip(1.0 - overflow, 0.0, 1.0))


def compute_threshold_selection_utility(candidate, event_type='generic', score_profile='default'):
    signal_cap = candidate.get('signal_cap')
    signal_health = compute_signal_health(candidate.get('signal_frequency', 0.0), signal_cap)
    candidate['signal_health'] = signal_health
    if is_modern_score_profile(score_profile):
        # 【Iter14 EV Regression】: signal_space_mean 和 signal_quality_mean 变成了连续 EV，不再局限于 [0, 1]
        # 为了保证不同指标权重可比，对 EV 均值做一层软归一化处理（除以 3.0 并截断），以体现高 EV 的正向作用。
        norm_space = float(np.clip(candidate.get('signal_space_mean', 0.0) / 3.0, 0.0, 1.5))
        norm_quality = float(np.clip(candidate.get('signal_quality_mean', 0.0) / 3.0, 0.0, 1.5))
        
        utility = (
            0.42 * float(candidate.get('precision', 0.0)) +
            0.18 * float(candidate.get('recall', 0.0)) +
            0.20 * max(1.0 - float(candidate.get('hard_negative_rate', 0.0)), 0.0) +
            0.12 * norm_space +
            0.08 * norm_quality +
            0.15 * signal_health
        )
        candidate['selection_utility'] = utility
        return utility
    candidate['selection_utility'] = signal_health
    return compute_precision_first_event_quality(candidate, event_type=event_type, score_profile=score_profile)


def score_event_candidate(candidate, selection_mode='f1', event_type='generic', score_profile='default'):
    if selection_mode == 'precision_first':
        return compute_threshold_selection_utility(candidate, event_type=event_type, score_profile=score_profile)
    return float(candidate['f1'])


def is_better_event_candidate(candidate, best, selection_mode='f1', event_type='generic', score_profile='default'):
    if best is None:
        return True
    candidate_score = score_event_candidate(
        candidate,
        selection_mode=selection_mode,
        event_type=event_type,
        score_profile=score_profile,
    )
    best_score = score_event_candidate(
        best,
        selection_mode=selection_mode,
        event_type=event_type,
        score_profile=score_profile,
    )
    if candidate_score > best_score + 1e-12:
        return True
    if best_score > candidate_score + 1e-12:
        return False
    if candidate['precision'] > best['precision'] + 1e-12:
        return True
    if best['precision'] > candidate['precision'] + 1e-12:
        return False
    if best['hard_negative_rate'] > candidate['hard_negative_rate'] + 1e-12:
        return True
    if candidate['hard_negative_rate'] > best['hard_negative_rate'] + 1e-12:
        return False
    if best['signal_frequency'] > candidate['signal_frequency'] + 1e-12:
        return True
    if candidate['signal_frequency'] > best['signal_frequency'] + 1e-12:
        return False
    return candidate['threshold'] > best['threshold']


def compute_event_metrics(
    scores,
    event_flags,
    hard_negative_flags,
    selection_mode='f1',
    event_type='generic',
    score_profile='default',
    target_values=None,
    aux_values=None,
):
    scores = np.asarray(scores, dtype=np.float64)
    event_flags = np.asarray(event_flags, dtype=bool)
    hard_negative_flags = np.asarray(hard_negative_flags, dtype=bool)
    if len(scores) == 0:
        return {
            'threshold': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'event_rate': 0.0,
            'hard_negative_rate': 0.0,
            'signal_frequency': 0.0,
            'label_frequency': 0.0,
            'oversignal': 0.0,
            'signal_space_mean': 0.0,
            'signal_quality_mean': 0.0,
            'signal_health': 0.0,
            'signal_cap': None,
            'selection_utility': 0.0,
        }
    label_frequency = float(np.mean(event_flags)) if len(event_flags) > 0 else 0.0
    precision_cfg = PRECISION_FIRST_PROFILE_CONFIG.get(score_profile) or PRECISION_FIRST_PROFILE_CONFIG['default']
    threshold_grid = (
        np.linspace(
            float(precision_cfg['threshold_grid_min']),
            min(float(precision_cfg['threshold_grid_max']), 0.9999),
            int(precision_cfg['threshold_grid_points']),
        )
        if selection_mode == 'precision_first'
        else np.linspace(0.55, 0.95, 9)
    )
    thresholds = np.unique(np.quantile(scores, threshold_grid))
    best = None
    fallback_best = None
    signal_cap = (
        resolve_event_oversignal_cap(label_frequency, event_type=event_type, score_profile=score_profile)
        if selection_mode == 'precision_first'
        else None
    )
    for threshold in thresholds:
        pred = scores >= threshold
        tp = np.sum(pred & event_flags)
        fp = np.sum(pred & ~event_flags)
        fn = np.sum(~pred & event_flags)
        tn = np.sum(~pred & ~event_flags)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(scores), 1)
        hn_rate = np.mean(pred[hard_negative_flags]) if np.any(hard_negative_flags) else 0.0
        event_rate = np.mean(pred[event_flags]) if np.any(event_flags) else 0.0
        signal_space_summary = compute_signal_space_summary(scores, threshold, target_values, aux_values)
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'event_rate': float(event_rate),
            'hard_negative_rate': float(hn_rate),
            'signal_frequency': float(np.mean(pred)),
            'label_frequency': label_frequency,
            'oversignal': max(float(np.mean(pred)) - label_frequency, 0.0),
            'signal_space_mean': float(signal_space_summary['signal_space_mean']),
            'signal_quality_mean': float(signal_space_summary['signal_quality_mean']),
            'signal_cap': signal_cap,
        }
        if is_better_event_candidate(
            candidate,
            fallback_best,
            selection_mode=selection_mode,
            event_type=event_type,
            score_profile=score_profile,
        ):
            fallback_best = candidate
        if (
            signal_cap is not None and
            not is_modern_score_profile(score_profile) and
            candidate['signal_frequency'] > signal_cap + 1e-12
        ):
            continue
        if is_better_event_candidate(
            candidate,
            best,
            selection_mode=selection_mode,
            event_type=event_type,
            score_profile=score_profile,
        ):
            best = candidate
    return best or fallback_best


def compute_event_metrics_at_threshold(
    scores,
    event_flags,
    hard_negative_flags,
    threshold,
    target_values=None,
    aux_values=None,
    event_type='generic',
    score_profile='default',
):
    scores = np.asarray(scores, dtype=np.float64)
    event_flags = np.asarray(event_flags, dtype=bool)
    hard_negative_flags = np.asarray(hard_negative_flags, dtype=bool)
    if len(scores) == 0:
        return {
            'threshold': float(threshold or 0.0),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'event_rate': 0.0,
            'hard_negative_rate': 0.0,
            'signal_frequency': 0.0,
            'label_frequency': 0.0,
            'oversignal': 0.0,
            'signal_space_mean': 0.0,
            'signal_quality_mean': 0.0,
            'signal_health': 0.0,
            'signal_cap': None,
            'selection_utility': 0.0,
        }
    label_frequency = float(np.mean(event_flags)) if len(event_flags) > 0 else 0.0
    threshold = float(threshold or 0.0)
    pred = scores >= threshold
    tp = np.sum(pred & event_flags)
    fp = np.sum(pred & ~event_flags)
    fn = np.sum(~pred & event_flags)
    tn = np.sum(~pred & ~event_flags)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(scores), 1)
    hn_rate = np.mean(pred[hard_negative_flags]) if np.any(hard_negative_flags) else 0.0
    event_rate = np.mean(pred[event_flags]) if np.any(event_flags) else 0.0
    signal_frequency = float(np.mean(pred)) if len(pred) else 0.0
    signal_space_summary = compute_signal_space_summary(scores, threshold, target_values, aux_values)
    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'event_rate': float(event_rate),
        'hard_negative_rate': float(hn_rate),
        'signal_frequency': signal_frequency,
        'label_frequency': label_frequency,
        'oversignal': max(signal_frequency - label_frequency, 0.0),
        'signal_space_mean': float(signal_space_summary['signal_space_mean']),
        'signal_quality_mean': float(signal_space_summary['signal_quality_mean']),
        'signal_cap': resolve_event_oversignal_cap(label_frequency, event_type=event_type, score_profile=score_profile),
    }
    compute_threshold_selection_utility(metrics, event_type=event_type, score_profile=score_profile)
    return metrics


def compute_event_quality(metrics):
    signal_gap = abs(metrics['signal_frequency'] - metrics['label_frequency'])
    return metrics['f1'] - 0.20 * metrics['hard_negative_rate'] - 0.05 * signal_gap


def compute_precision_first_event_quality(metrics, event_type='generic', score_profile='default'):
    """
    Iter11 EDL: Naturally prioritize Precision by letting the model optimize.
    Removed the hard -999 constraints, letting EDL sort naturally.
    """
    # Focus purely on finding the most accurate signal
    return (
        0.80 * metrics['precision'] +           # Massive weight on precision
        0.10 * metrics['f1'] +                  # Slight weight on F1 to break ties
        0.10 * metrics['accuracy'] -
        0.40 * metrics['hard_negative_rate']    # Heavy penalty for bad signals
    )


def compute_signal_space_summary(scores, threshold, target_values, aux_values):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target_values, dtype=np.float64).reshape(-1)
    aux_values = np.asarray(aux_values, dtype=np.float64).reshape(-1)
    if scores.size == 0 or target_values.size != scores.size or aux_values.size != scores.size:
        return {
            'signal_target_mean': 0.0,
            'signal_space_mean': 0.0,
            'signal_quality_mean': 0.0,
            'all_target_mean': 0.0,
            'all_space_mean': 0.0,
            'all_quality_mean': 0.0,
        }
    pred = scores >= float(threshold)
    quality_values = 0.65 * target_values + 0.35 * aux_values

    def _masked_mean(values, mask):
        if values.size == 0 or not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    return {
        'signal_target_mean': _masked_mean(target_values, pred),
        'signal_space_mean': _masked_mean(aux_values, pred),
        'signal_quality_mean': _masked_mean(quality_values, pred),
        'all_target_mean': float(np.mean(target_values)) if target_values.size else 0.0,
        'all_space_mean': float(np.mean(aux_values)) if aux_values.size else 0.0,
        'all_quality_mean': float(np.mean(quality_values)) if quality_values.size else 0.0,
    }


def compose_metric_scores(main_scores, aux_scores, score_profile='default', event_type='generic'):
    primary = np.asarray(main_scores, dtype=np.float64).reshape(-1)
    if score_profile not in AUX_GATED_SCORE_PROFILES:
        return primary
    aux_values = np.asarray(aux_scores, dtype=np.float64).reshape(-1)
    if aux_values.size != primary.size:
        aux_values = np.zeros_like(primary)
    
    # 【Iter14 EV Regression】：取消对 primary 概率的 [0, 1] 强行截断，
    # 允许连续 EV 目标保持自身的数值范围。
    # primary = np.clip(primary, 0.0, 1.0)
    
    # 同样取消对 aux_values 的 tanh 归一化，直接使用其连续值。
    # aux_values = np.tanh(np.maximum(aux_values, 0.0) / 2.0)
    
    coherent = np.minimum(primary, aux_values)
    if event_type == 'reversion':
        return 0.48 * primary + 0.22 * aux_values + 0.30 * coherent
    return 0.52 * primary + 0.18 * aux_values + 0.30 * coherent


def compute_discovery_space_quality(summary, event_type='generic'):
    signal_quality_mean = float(summary.get('signal_quality_mean', 0.0))
    signal_space_mean = float(summary.get('signal_space_mean', 0.0))
    all_quality_mean = float(summary.get('all_quality_mean', 0.0))
    all_space_mean = float(summary.get('all_space_mean', 0.0))
    quality_lift = max(signal_quality_mean - all_quality_mean, 0.0)
    space_lift = max(signal_space_mean - all_space_mean, 0.0)
    if event_type == 'reversion':
        return (
            0.58 * np.tanh(signal_quality_mean / 1.20) +
            0.22 * np.tanh(signal_space_mean / 1.10) +
            0.12 * np.tanh(quality_lift / 0.60) +
            0.08 * np.tanh(space_lift / 0.50)
        )
    return (
        0.60 * np.tanh(signal_quality_mean / 1.25) +
        0.20 * np.tanh(signal_space_mean / 1.15) +
        0.12 * np.tanh(quality_lift / 0.70) +
        0.08 * np.tanh(space_lift / 0.60)
    )


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def compute_iter14_structural_components(summary):
    summary = summary or {}
    public_violation_rate = _clip01(summary.get('public_below_directional_violation_rate', 1.0))
    public_feasibility = _clip01(1.0 - public_violation_rate)
    
    # Iter14: We completely disabled directional gates and outputs are pure regression EV logits.
    # Therefore, we force them to 1.0 to eliminate any arbitrary penalization on the final score.
    directional_floor_mean = 1.0
    directional_floor_event_mean = 1.0
    directional_support_rate = 1.0
    directional_floor_quality = 1.0
    
    return {
        'public_violation_rate': public_violation_rate,
        'public_feasibility': public_feasibility,
        'directional_floor_mean': directional_floor_mean,
        'directional_floor_reversion_event_mean': directional_floor_event_mean,
        'directional_floor_quality': directional_floor_quality,
        'directional_support_rate': directional_support_rate,
        'structural_quality': public_feasibility,
    }


def _zero_direction_metrics():
    return {
        'accuracy': 0.0,
        'macro_f1': 0.0,
        'bear_f1': 0.0,
        'bull_f1': 0.0,
        'support': 0,
    }


def compute_direction_metrics(bear_scores, bull_scores, flags):
    bear_scores = np.asarray(bear_scores, dtype=np.float64).reshape(-1)
    bull_scores = np.asarray(bull_scores, dtype=np.float64).reshape(-1)
    flags = np.asarray(flags, dtype=np.float64)
    if flags.ndim >= 1:
        aligned_size = min(bear_scores.shape[0], bull_scores.shape[0], flags.shape[0])
        bear_scores = bear_scores[:aligned_size]
        bull_scores = bull_scores[:aligned_size]
        flags = flags[:aligned_size]
    bear_idx = EVENT_FLAG_INDEX['reversion_down_context']
    bull_idx = EVENT_FLAG_INDEX['reversion_up_context']
    if flags.ndim != 2 or flags.shape[1] <= max(bear_idx, bull_idx):
        return _zero_direction_metrics()
    bear_mask = flags[:, bear_idx] > 0.5
    bull_mask = flags[:, bull_idx] > 0.5
    valid_mask = bear_mask | bull_mask
    if not np.any(valid_mask):
        return _zero_direction_metrics()
    truth = np.where(bear_mask[valid_mask], 1, 0)
    pred = np.where(bear_scores[valid_mask] >= bull_scores[valid_mask], 1, 0)

    def _binary_f1(target_label):
        truth_mask = truth == target_label
        pred_mask = pred == target_label
        tp = np.sum(truth_mask & pred_mask)
        fp = np.sum(~truth_mask & pred_mask)
        fn = np.sum(truth_mask & ~pred_mask)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return 2 * precision * recall / max(precision + recall, 1e-8)

    bear_f1 = float(_binary_f1(1))
    bull_f1 = float(_binary_f1(0))
    return {
        'accuracy': float(np.mean(truth == pred)),
        'macro_f1': float((bear_f1 + bull_f1) / 2.0),
        'bear_f1': bear_f1,
        'bull_f1': bull_f1,
        'support': int(np.sum(valid_mask)),
    }


def compute_checkpoint_score(
    breakout_metrics,
    reversion_metrics,
    breakout_corr,
    reversion_corr,
    direction_macro_f1=None,
    breakout_space_summary=None,
    reversion_space_summary=None,
    structural_summary=None,
    profile='default',
):
    breakout_quality = compute_event_quality(breakout_metrics)
    reversion_quality = compute_event_quality(reversion_metrics)
    if profile in {'short_t_precision_focus', 'recent_precision_v1', 'iter11_precision_first'}:
        breakout_quality = compute_precision_first_event_quality(
            breakout_metrics,
            event_type='breakout',
            score_profile=profile,
        )
        reversion_quality = compute_precision_first_event_quality(
            reversion_metrics,
            event_type='reversion',
            score_profile=profile,
        )
        if direction_macro_f1 is None:
            return (
                0.46 * breakout_quality +
                0.46 * reversion_quality +
                0.04 * breakout_corr +
                0.04 * reversion_corr
            )
        return (
            0.40 * breakout_quality +
            0.44 * reversion_quality +
            0.10 * direction_macro_f1 +
            0.03 * breakout_corr +
            0.03 * reversion_corr
        )
    if profile == 'iter12_guarded_precision_first':
        breakout_quality = compute_precision_first_event_quality(
            breakout_metrics,
            event_type='breakout',
            score_profile=profile,
        )
        reversion_quality = compute_precision_first_event_quality(
            reversion_metrics,
            event_type='reversion',
            score_profile=profile,
        )
        structural_components = compute_iter14_structural_components(structural_summary)
        if direction_macro_f1 is None:
            return (
                0.30 * breakout_quality +
                0.32 * reversion_quality +
                0.16 * structural_components['public_feasibility'] +
                0.16 * structural_components['directional_floor_quality'] +
                0.03 * breakout_corr +
                0.03 * reversion_corr
            )
        return (
            0.26 * breakout_quality +
            0.30 * reversion_quality +
            0.10 * direction_macro_f1 +
            0.14 * structural_components['public_feasibility'] +
            0.14 * structural_components['directional_floor_quality'] +
            0.03 * breakout_corr +
            0.03 * reversion_corr
        )
    if profile == 'iter12_soft_guarded_precision_first':
        breakout_quality = compute_precision_first_event_quality(
            breakout_metrics,
            event_type='breakout',
            score_profile=profile,
        )
        reversion_quality = compute_precision_first_event_quality(
            reversion_metrics,
            event_type='reversion',
            score_profile=profile,
        )
        structural_components = compute_iter12_structural_components(structural_summary)
        if direction_macro_f1 is None:
            return (
                0.30 * breakout_quality +
                0.30 * reversion_quality +
                0.18 * structural_components['public_feasibility'] +
                0.12 * structural_components['directional_support_rate'] +
                0.05 * breakout_corr +
                0.05 * reversion_corr
            )
        return (
            0.26 * breakout_quality +
            0.28 * reversion_quality +
            0.10 * direction_macro_f1 +
            0.16 * structural_components['public_feasibility'] +
            0.10 * structural_components['directional_support_rate'] +
            0.05 * breakout_corr +
            0.05 * reversion_corr
        )
    if profile == 'short_t_discovery_focus':
        breakout_quality = compute_precision_first_event_quality(
            breakout_metrics,
            event_type='breakout',
            score_profile=profile,
        )
        reversion_quality = compute_precision_first_event_quality(
            reversion_metrics,
            event_type='reversion',
            score_profile=profile,
        )
        breakout_space_quality = compute_discovery_space_quality(breakout_space_summary or {}, event_type='breakout')
        reversion_space_quality = compute_discovery_space_quality(reversion_space_summary or {}, event_type='reversion')
        if direction_macro_f1 is None:
            return (
                0.31 * breakout_quality +
                0.31 * reversion_quality +
                0.16 * breakout_space_quality +
                0.16 * reversion_space_quality +
                0.03 * breakout_corr +
                0.03 * reversion_corr
            )
        return (
            0.26 * breakout_quality +
            0.28 * reversion_quality +
            0.14 * breakout_space_quality +
            0.16 * reversion_space_quality +
            0.10 * direction_macro_f1 +
            0.03 * breakout_corr +
            0.03 * reversion_corr
        )
    if profile == 'short_t_discovery_guarded_focus':
        breakout_quality = compute_precision_first_event_quality(
            breakout_metrics,
            event_type='breakout',
            score_profile=profile,
        )
        reversion_quality = compute_precision_first_event_quality(
            reversion_metrics,
            event_type='reversion',
            score_profile=profile,
        )
        breakout_space_quality = compute_discovery_space_quality(breakout_space_summary or {}, event_type='breakout')
        reversion_space_quality = compute_discovery_space_quality(reversion_space_summary or {}, event_type='reversion')
        if direction_macro_f1 is None:
            return (
                0.36 * breakout_quality +
                0.40 * reversion_quality +
                0.10 * breakout_space_quality +
                0.10 * reversion_space_quality +
                0.02 * breakout_corr +
                0.02 * reversion_corr
            )
        return (
            0.32 * breakout_quality +
            0.36 * reversion_quality +
            0.08 * breakout_space_quality +
            0.08 * reversion_space_quality +
            0.12 * direction_macro_f1 +
            0.02 * breakout_corr +
            0.02 * reversion_corr
        )
    if profile == 'short_t_breakout_focus':
        if direction_macro_f1 is None:
            return (
                0.66 * breakout_quality +
                0.24 * reversion_quality +
                0.06 * breakout_corr +
                0.04 * reversion_corr
            )
        return (
            0.58 * breakout_quality +
            0.20 * reversion_quality +
            0.12 * direction_macro_f1 +
            0.06 * breakout_corr +
            0.04 * reversion_corr
        )
    if profile == 'short_t_balanced_focus':
        if direction_macro_f1 is None:
            return (
                0.47 * breakout_quality +
                0.47 * reversion_quality +
                0.03 * breakout_corr +
                0.03 * reversion_corr
            )
        return (
            0.43 * breakout_quality +
            0.43 * reversion_quality +
            0.10 * direction_macro_f1 +
            0.02 * breakout_corr +
            0.02 * reversion_corr
        )
    if profile == 'short_t_balanced_guarded':
        if direction_macro_f1 is None:
            return (
                0.45 * breakout_quality +
                0.45 * reversion_quality +
                0.05 * breakout_corr +
                0.05 * reversion_corr
            )
        return (
            0.40 * breakout_quality +
            0.40 * reversion_quality +
            0.14 * direction_macro_f1 +
            0.03 * breakout_corr +
            0.03 * reversion_corr
        )
    if profile == 'iterA4_event_focus':
        if direction_macro_f1 is None:
            return (
                0.55 * breakout_quality +
                0.39 * reversion_quality +
                0.03 * breakout_corr +
                0.03 * reversion_corr
            )
        return (
            0.50 * breakout_quality +
            0.36 * reversion_quality +
            0.08 * direction_macro_f1 +
            0.03 * breakout_corr +
            0.03 * reversion_corr
        )
    if direction_macro_f1 is None:
        return (
            0.46 * breakout_quality +
            0.54 * reversion_quality +
            0.03 * breakout_corr +
            0.04 * reversion_corr
        )
    return (
        0.40 * breakout_quality +
        0.42 * reversion_quality +
        0.12 * direction_macro_f1 +
        0.03 * breakout_corr +
        0.03 * reversion_corr
    )


def build_metric_bucket():
    return {
        'preds': [],
        'aux_preds': [],
        'targets': [],
        'aux_targets': [],
        'flags': [],
        'logs': [],
        'skip_counters': defaultdict(int),
        'debug_batches': {key: [] for key in DEBUG_BATCH_KEYS},
    }


def merge_metric_bucket(destination, source):
    destination['preds'].extend(source['preds'])
    destination['aux_preds'].extend(source['aux_preds'])
    destination['targets'].extend(source['targets'])
    destination['aux_targets'].extend(source['aux_targets'])
    destination['flags'].extend(source['flags'])
    destination['logs'].extend(source['logs'])
    for key, value in source.get('skip_counters', {}).items():
        destination['skip_counters'][key] += int(value)
    for key in DEBUG_BATCH_KEYS:
        destination['debug_batches'][key].extend(source['debug_batches'][key])


def _flatten_debug_batches(batches):
    if not batches:
        return np.array([], dtype=np.float32)
    return np.concatenate([np.asarray(batch, dtype=np.float32).reshape(-1) for batch in batches]).astype(np.float32)


def _flatten_per_sample_direction_batches(batches):
    if not batches:
        return np.array([], dtype=np.float32)
    flattened = []
    for batch in batches:
        values = np.asarray(batch, dtype=np.float32)
        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim > 1:
            if values.shape[1] == 2:
                values = values[:, 1, ...]
            values = values.reshape(values.shape[0], -1).mean(axis=1)
        flattened.append(values.reshape(-1))
    return np.concatenate(flattened).astype(np.float32)


def _metric_bucket_mean(logs, key):
    if not logs or key not in logs[0]:
        return 0.0
    total_weight = sum(item.get('batch_size', 1) for item in logs)
    if total_weight == 0:
        return 0.0
    total_value = sum(float(item[key]) * item.get('batch_size', 1) for item in logs)
    return float(total_value / total_weight)


def _masked_array_mean(values, mask):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0 or mask.size == 0 or values.size != mask.size or not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def summarize_metric_bucket(bucket, score_profile='default', use_direction_metrics=False, frozen_thresholds=None):
    zero_summary = {
        'sample_count': 0,
        'avg_val_loss': 0.0,
        'breakout_corr': 0.0,
        'reversion_corr': 0.0,
        'pred_rev_mean': 0.0,
        'pred_rev_event_mean': 0.0,
        'breakout_event_mean': 0.0,
        'reversion_event_mean': 0.0,
        'breakout_hard_negative_mean': 0.0,
        'reversion_hard_negative_mean': 0.0,
        'breakout_gap': 0.0,
        'reversion_gap': 0.0,
        'reversion_event_count': 0,
        'composite_score': 0.0,
        'breakout_metrics': compute_event_metrics([], [], []),
        'reversion_metrics': compute_event_metrics([], [], []),
        'direction_metrics': _zero_direction_metrics(),
        'signal_frequency': {'breakout': 0.0, 'reversion': 0.0},
        'label_frequency': {'breakout': 0.0, 'reversion': 0.0},
        'oversignal': {'breakout': 0.0, 'reversion': 0.0},
        'breakout_signal_frequency': 0.0,
        'reversion_signal_frequency': 0.0,
        'breakout_label_frequency': 0.0,
        'reversion_label_frequency': 0.0,
        'breakout_oversignal': 0.0,
        'reversion_oversignal': 0.0,
        'breakout_signal_target_mean': 0.0,
        'reversion_signal_target_mean': 0.0,
        'breakout_signal_space_mean': 0.0,
        'reversion_signal_space_mean': 0.0,
        'breakout_signal_quality_mean': 0.0,
        'reversion_signal_quality_mean': 0.0,
        'breakout_all_quality_mean': 0.0,
        'reversion_all_quality_mean': 0.0,
        'direction_gate_mean': 0.0,
        'direction_gate_std': 0.0,
        'compression_gate_mean': 0.0,
        'compression_gate_std': 0.0,
        'directional_gate_mean': 0.0,
        'directional_gate_std': 0.0,
        'gate_floor_hit_rate': 0.0,
        'public_reversion_gate_mean': 0.0,
        'public_reversion_gate_std': 0.0,
        'breakout_residual_gate_mean': 0.0,
        'breakout_residual_gate_std': 0.0,
        'directional_floor_mean': 0.0,
        'directional_floor_reversion_event_mean': 0.0,
        'directional_floor_quality': 0.0,
        'directional_support_rate': 0.0,
        'public_feasibility_mean': 0.0,
        'structural_guard_quality': 0.0,
        'pred_breakout_std': 0.0,
        'pred_reversion_std': 0.0,
        'selected_horizon_breakout_mean': 0.0,
        'selected_horizon_reversion_mean': 0.0,
        'selected_horizon_breakout_value_mean': 0.0,
        'selected_horizon_reversion_value_mean': 0.0,
        'horizon_entropy_breakout_mean': 0.0,
        'horizon_entropy_reversion_mean': 0.0,
        'loss_main': 0.0,
        'loss_aux': 0.0,
        'loss_rank': 0.0,
        'loss_constraint_penalty': 0.0,
        'loss_horizon_event': 0.0,
        'loss_horizon_aux': 0.0,
        'loss_horizon_align': 0.0,
        'loss_horizon_hard_negative': 0.0,
        'loss_horizon_entropy': 0.0,
        'loss_signal_calibration': 0.0,
        'thresholds_frozen': False,
        'frozen_thresholds': None,
        **{
            key: 0.0
            for base_key in CONSTRAINT_STAT_BASE_KEYS
            for key in (base_key, f'{base_key}_rate')
        },
    }
    if not bucket['preds']:
        return zero_summary

    pred_np = np.vstack(bucket['preds'])
    aux_pred_np = np.vstack(bucket['aux_preds']) if bucket['aux_preds'] else np.zeros_like(pred_np)
    target_np = np.vstack(bucket['targets'])
    aux_np = np.vstack(bucket['aux_targets']) if bucket['aux_targets'] else np.zeros_like(target_np)
    flags_np = np.vstack(bucket['flags'])
    pred_rev_np = np.maximum(pred_np[:, 1], 0.0)
    aux_breakout_np = np.maximum(aux_pred_np[:, 0], 0.0)
    aux_reversion_np = np.maximum(aux_pred_np[:, 1], 0.0)
    breakout_metric_scores = compose_metric_scores(
        pred_np[:, 0],
        aux_breakout_np,
        score_profile=score_profile,
        event_type='breakout',
    )
    reversion_metric_scores = compose_metric_scores(
        pred_rev_np,
        aux_reversion_np,
        score_profile=score_profile,
        event_type='reversion',
    )
    reversion_event_mask = flags_np[:, 1] > 0.5
    breakout_corr = safe_corr(pred_np[:, 0], target_np[:, 0])
    reversion_corr = safe_corr(pred_rev_np, target_np[:, 1])
    breakout_event_mean = float(breakout_metric_scores[flags_np[:, 0] > 0.5].mean()) if np.any(flags_np[:, 0] > 0.5) else 0.0
    reversion_event_mean = float(reversion_metric_scores[reversion_event_mask].mean()) if np.any(reversion_event_mask) else 0.0
    breakout_hn_mean = float(breakout_metric_scores[flags_np[:, 2] > 0.5].mean()) if np.any(flags_np[:, 2] > 0.5) else 0.0
    reversion_hn_mean = float(reversion_metric_scores[flags_np[:, 3] > 0.5].mean()) if np.any(flags_np[:, 3] > 0.5) else 0.0
    event_selection_mode = resolve_event_selection_mode(score_profile)
    frozen_breakout_threshold = None
    frozen_reversion_threshold = None
    if isinstance(frozen_thresholds, dict):
        frozen_breakout_threshold = frozen_thresholds.get('breakout')
        frozen_reversion_threshold = frozen_thresholds.get('reversion')
    if frozen_breakout_threshold is None:
        breakout_metrics = compute_event_metrics(
            breakout_metric_scores,
            flags_np[:, 0] > 0.5,
            flags_np[:, 2] > 0.5,
            selection_mode=event_selection_mode,
            event_type='breakout',
            score_profile=score_profile,
            target_values=target_np[:, 0],
            aux_values=aux_np[:, 0],
        )
    else:
        breakout_metrics = compute_event_metrics_at_threshold(
            breakout_metric_scores,
            flags_np[:, 0] > 0.5,
            flags_np[:, 2] > 0.5,
            threshold=frozen_breakout_threshold,
            target_values=target_np[:, 0],
            aux_values=aux_np[:, 0],
            event_type='breakout',
            score_profile=score_profile,
        )
    if frozen_reversion_threshold is None:
        reversion_metrics = compute_event_metrics(
            reversion_metric_scores,
            reversion_event_mask,
            flags_np[:, 3] > 0.5,
            selection_mode=event_selection_mode,
            event_type='reversion',
            score_profile=score_profile,
            target_values=target_np[:, 1],
            aux_values=aux_np[:, 1],
        )
    else:
        reversion_metrics = compute_event_metrics_at_threshold(
            reversion_metric_scores,
            reversion_event_mask,
            flags_np[:, 3] > 0.5,
            threshold=frozen_reversion_threshold,
            target_values=target_np[:, 1],
            aux_values=aux_np[:, 1],
            event_type='reversion',
            score_profile=score_profile,
        )
    breakout_space_summary = compute_signal_space_summary(
        breakout_metric_scores,
        breakout_metrics['threshold'],
        target_np[:, 0],
        aux_np[:, 0],
    )
    reversion_space_summary = compute_signal_space_summary(
        reversion_metric_scores,
        reversion_metrics['threshold'],
        target_np[:, 1],
        aux_np[:, 1],
    )
    bear_scores = _flatten_per_sample_direction_batches(bucket['debug_batches']['bear_score'])
    bull_scores = _flatten_per_sample_direction_batches(bucket['debug_batches']['bull_score'])
    direction_metrics = compute_direction_metrics(bear_scores, bull_scores, flags_np) if (
        use_direction_metrics and bear_scores.size > 0 and bull_scores.size > 0
    ) else _zero_direction_metrics()
    direction_gate_values = _flatten_debug_batches(bucket['debug_batches']['direction_gate'])
    compression_gate_values = _flatten_debug_batches(bucket['debug_batches']['compression_gate'])
    directional_gate_values = _flatten_debug_batches(bucket['debug_batches']['directional_gate'])
    gate_floor_hit_values = _flatten_debug_batches(bucket['debug_batches']['gate_floor_hit'])
    public_gate_values = _flatten_debug_batches(bucket['debug_batches']['public_reversion_gate'])
    breakout_residual_gate_values = _flatten_debug_batches(bucket['debug_batches']['breakout_residual_gate'])
    directional_reversion_values = _flatten_per_sample_direction_batches(bucket['debug_batches']['directional_reversion'])
    directional_floor_values = _flatten_per_sample_direction_batches(bucket['debug_batches']['directional_floor'])
    directional_floor_mean = float(directional_floor_values.mean()) if directional_floor_values.size else 0.0
    directional_floor_reversion_event_mean = _masked_array_mean(directional_floor_values, reversion_event_mask)
    public_below_directional_violation_rate = _metric_bucket_mean(bucket['logs'], 'public_below_directional_violation_rate')
    directional_support_rate = 0.0
    if directional_floor_values.size and directional_reversion_values.size:
        aligned_size = min(
            directional_floor_values.size,
            directional_reversion_values.size,
            reversion_event_mask.shape[0],
        )
        if aligned_size > 0:
            support_mask = reversion_event_mask[:aligned_size]
            support_margin = directional_floor_values[:aligned_size] - directional_reversion_values[:aligned_size]
            if np.any(support_mask):
                directional_support_rate = float(np.mean(support_margin[support_mask] > 1e-6))
    structural_components = compute_iter14_structural_components(
        {
            'public_below_directional_violation_rate': public_below_directional_violation_rate,
            'directional_floor_mean': directional_floor_mean,
            'directional_floor_reversion_event_mean': directional_floor_reversion_event_mean,
            'directional_support_rate': directional_support_rate,
            'prior_label_frequency': reversion_metrics.get('label_frequency', 0.04) # 传入真实的标签频率作为先验校准
        }
    )
    selected_breakout = _flatten_debug_batches(bucket['debug_batches']['selected_horizon_breakout'])
    selected_reversion = _flatten_debug_batches(bucket['debug_batches']['selected_horizon_reversion'])
    selected_breakout_value = _flatten_debug_batches(bucket['debug_batches']['selected_horizon_breakout_value'])
    selected_reversion_value = _flatten_debug_batches(bucket['debug_batches']['selected_horizon_reversion_value'])
    entropy_breakout = _flatten_debug_batches(bucket['debug_batches']['horizon_entropy_breakout'])
    entropy_reversion = _flatten_debug_batches(bucket['debug_batches']['horizon_entropy_reversion'])
    breakout_gap = breakout_event_mean - breakout_hn_mean
    reversion_gap = reversion_event_mean - reversion_hn_mean
    composite_score = compute_checkpoint_score(
        breakout_metrics,
        reversion_metrics,
        breakout_corr,
        reversion_corr,
        direction_metrics['macro_f1'] if use_direction_metrics else None,
        breakout_space_summary=breakout_space_summary,
        reversion_space_summary=reversion_space_summary,
        structural_summary=structural_components,
        profile=score_profile,
    )
    summary = dict(zero_summary)
    summary.update(
        {
            'sample_count': int(pred_np.shape[0]),
            'avg_val_loss': _metric_bucket_mean(bucket['logs'], 'total_loss'),
            'breakout_corr': breakout_corr,
            'reversion_corr': reversion_corr,
            'pred_breakout_std': float(pred_np[:, 0].std()) if pred_np.size else 0.0,
            'pred_reversion_std': float(pred_rev_np.std()) if pred_rev_np.size else 0.0,
            'pred_rev_mean': float(pred_rev_np.mean()) if pred_rev_np.size else 0.0,
            'pred_rev_event_mean': reversion_event_mean,
            'breakout_event_mean': breakout_event_mean,
            'reversion_event_mean': reversion_event_mean,
            'breakout_hard_negative_mean': breakout_hn_mean,
            'reversion_hard_negative_mean': reversion_hn_mean,
            'breakout_gap': breakout_gap,
            'reversion_gap': reversion_gap,
            'reversion_event_count': int(np.sum(reversion_event_mask)),
            'composite_score': composite_score,
            'breakout_metrics': breakout_metrics,
            'reversion_metrics': reversion_metrics,
            'direction_metrics': direction_metrics,
            'signal_frequency': {
                'breakout': breakout_metrics['signal_frequency'],
                'reversion': reversion_metrics['signal_frequency'],
            },
            'label_frequency': {
                'breakout': breakout_metrics['label_frequency'],
                'reversion': reversion_metrics['label_frequency'],
            },
            'oversignal': {
                'breakout': breakout_metrics['oversignal'],
                'reversion': reversion_metrics['oversignal'],
            },
            'breakout_signal_frequency': breakout_metrics['signal_frequency'],
            'reversion_signal_frequency': reversion_metrics['signal_frequency'],
            'breakout_label_frequency': breakout_metrics['label_frequency'],
            'reversion_label_frequency': reversion_metrics['label_frequency'],
            'breakout_oversignal': breakout_metrics['oversignal'],
            'reversion_oversignal': reversion_metrics['oversignal'],
            'breakout_signal_target_mean': breakout_space_summary['signal_target_mean'],
            'reversion_signal_target_mean': reversion_space_summary['signal_target_mean'],
            'breakout_signal_space_mean': breakout_space_summary['signal_space_mean'],
            'reversion_signal_space_mean': reversion_space_summary['signal_space_mean'],
            'breakout_signal_quality_mean': breakout_space_summary['signal_quality_mean'],
            'reversion_signal_quality_mean': reversion_space_summary['signal_quality_mean'],
            'breakout_all_quality_mean': breakout_space_summary['all_quality_mean'],
            'reversion_all_quality_mean': reversion_space_summary['all_quality_mean'],
            'direction_gate_mean': float(direction_gate_values.mean()) if direction_gate_values.size else 0.0,
            'direction_gate_std': float(direction_gate_values.std()) if direction_gate_values.size else 0.0,
            'compression_gate_mean': float(compression_gate_values.mean()) if compression_gate_values.size else 0.0,
            'compression_gate_std': float(compression_gate_values.std()) if compression_gate_values.size else 0.0,
            'directional_gate_mean': float(directional_gate_values.mean()) if directional_gate_values.size else 0.0,
            'directional_gate_std': float(directional_gate_values.std()) if directional_gate_values.size else 0.0,
            'gate_floor_hit_rate': float(gate_floor_hit_values.mean()) if gate_floor_hit_values.size else 0.0,
            'public_reversion_gate_mean': float(public_gate_values.mean()) if public_gate_values.size else 0.0,
            'public_reversion_gate_std': float(public_gate_values.std()) if public_gate_values.size else 0.0,
            'breakout_residual_gate_mean': float(breakout_residual_gate_values.mean()) if breakout_residual_gate_values.size else 0.0,
            'breakout_residual_gate_std': float(breakout_residual_gate_values.std()) if breakout_residual_gate_values.size else 0.0,
            'directional_floor_mean': directional_floor_mean,
            'directional_floor_reversion_event_mean': directional_floor_reversion_event_mean,
            'directional_floor_quality': structural_components['directional_floor_quality'],
            'directional_support_rate': structural_components['directional_support_rate'],
            'public_feasibility_mean': structural_components['public_feasibility'],
            'structural_guard_quality': structural_components['structural_quality'],
            'selected_horizon_breakout_mean': float(selected_breakout.mean()) if selected_breakout.size else 0.0,
            'selected_horizon_reversion_mean': float(selected_reversion.mean()) if selected_reversion.size else 0.0,
            'selected_horizon_breakout_value_mean': float(selected_breakout_value.mean()) if selected_breakout_value.size else 0.0,
            'selected_horizon_reversion_value_mean': float(selected_reversion_value.mean()) if selected_reversion_value.size else 0.0,
            'horizon_entropy_breakout_mean': float(entropy_breakout.mean()) if entropy_breakout.size else 0.0,
            'horizon_entropy_reversion_mean': float(entropy_reversion.mean()) if entropy_reversion.size else 0.0,
            'loss_main': _metric_bucket_mean(bucket['logs'], 'main'),
            'loss_aux': _metric_bucket_mean(bucket['logs'], 'aux'),
            'loss_rank': _metric_bucket_mean(bucket['logs'], 'rank'),
            'loss_constraint_penalty': _metric_bucket_mean(bucket['logs'], 'constraint_penalty'),
            'loss_horizon_event': _metric_bucket_mean(bucket['logs'], 'horizon_event'),
            'loss_horizon_aux': _metric_bucket_mean(bucket['logs'], 'horizon_aux'),
            'loss_horizon_align': _metric_bucket_mean(bucket['logs'], 'horizon_align'),
            'loss_horizon_hard_negative': _metric_bucket_mean(bucket['logs'], 'horizon_hard_negative'),
            'loss_horizon_entropy': _metric_bucket_mean(bucket['logs'], 'horizon_entropy'),
            'loss_signal_calibration': _metric_bucket_mean(bucket['logs'], 'signal_calibration'),
            'thresholds_frozen': bool(frozen_breakout_threshold is not None or frozen_reversion_threshold is not None),
            'frozen_thresholds': {
                'breakout': float(frozen_breakout_threshold) if frozen_breakout_threshold is not None else None,
                'reversion': float(frozen_reversion_threshold) if frozen_reversion_threshold is not None else None,
            } if (frozen_breakout_threshold is not None or frozen_reversion_threshold is not None) else None,
        }
    )
    for base_key in CONSTRAINT_STAT_BASE_KEYS:
        summary[base_key] = _metric_bucket_mean(bucket['logs'], base_key)
        summary[f'{base_key}_rate'] = _metric_bucket_mean(bucket['logs'], f'{base_key}_rate')
    return summary


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def append_jsonl(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(make_json_safe(payload), ensure_ascii=False) + '\n')


def read_jsonl_records(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


# SECTION: DATA HELPERS


def dataset_cache_enabled(args):
    if getattr(args, 'disable_dataset_cache', False):
        return False
    market = getattr(args, 'market', 'legacy_multiasset')
    if market == 'ashare':
        return True
    return uses_recent_runtime_splits(args)


def uses_recent_runtime_splits(args):
    return (
        getattr(args, 'split_scheme', 'time') == 'rolling_recent_v1' and
        getattr(args, 'market', 'legacy_multiasset') in {'ashare', 'legacy_multiasset'}
    )


def resolve_dataset_cache_dir(args):
    cache_dir = getattr(args, 'dataset_cache_dir', None)
    if cache_dir:
        return cache_dir
    return os.path.join(args.save_dir, 'dataset_cache')


def build_runtime_dataset_cache_path(record, args, global_horizon_grid=None):
    source_path = os.path.abspath(record['path'])
    source_stat = os.stat(source_path)
    cache_identity = {
        'version': DATASET_CACHE_VERSION,
        'path': source_path,
        'source_size': int(source_stat.st_size),
        'source_mtime_ns': int(getattr(source_stat, 'st_mtime_ns', int(source_stat.st_mtime * 1e9))),
        'split_label': record.get('split_label') or getattr(args, 'split_label', None),
        'window_size': int(getattr(args, 'window_size', 20)),
        'forecast_horizon': int(getattr(args, 'horizon', 10)),
        'fast_full': bool(getattr(args, 'fast_full', False)),
        'dataset_profile': getattr(args, 'dataset_profile', 'iterA2'),
        'split_scheme': getattr(args, 'split_scheme', 'time'),
        'horizon_search_spec': getattr(args, 'horizon_search_spec', None),
        'horizon_family_mode': getattr(args, 'horizon_family_mode', 'legacy'),
        'global_horizon_grid': global_horizon_grid,
        'config_fingerprint': getattr(args, 'config_fingerprint', None),
    }
    digest = hashlib.sha256(
        json.dumps(make_json_safe(cache_identity), ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()[:24]
    file_stub = '_'.join(
        item for item in [
            record.get('asset_code'),
            normalize_timeframe_label(record.get('timeframe')) or 'unknown',
            record.get('split_label') or getattr(args, 'split_label', None) or 'default',
        ]
        if item
    )
    safe_stub = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in file_stub) or 'dataset'
    return os.path.join(resolve_dataset_cache_dir(args), f'{safe_stub}_{digest}.pt')


def try_load_runtime_dataset_cache(cache_path):
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f"[DATASET CACHE] Failed to load {cache_path}: {exc}. Rebuilding cache entry.")
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None
    if not isinstance(payload, dict):
        return None
    dataset_payloads = payload.get('datasets', {})
    if isinstance(dataset_payloads, dict):
        payload['datasets'] = {
            split_name: deserialize_cached_dataset(dataset_payload)
            for split_name, dataset_payload in dataset_payloads.items()
            if dataset_payload is not None
        }
    return payload


def serialize_cached_dataset(dataset):
    if dataset is None:
        return None
    state = dataset.__getstate__() if hasattr(dataset, '__getstate__') else dict(dataset.__dict__)
    return {
        'class_name': dataset.__class__.__name__,
        'state': state,
    }


def deserialize_cached_dataset(payload):
    if payload is None:
        return None
    class_name = payload.get('class_name')
    if class_name != 'AshareFinancialDataset':
        raise ValueError(f'Unsupported cached dataset class: {class_name}')
    dataset_cls = create_ashare_dataset_splits.__globals__.get('AshareFinancialDataset')
    if dataset_cls is None:
        raise RuntimeError('AshareFinancialDataset class is unavailable for cache restore.')
    dataset = dataset_cls.__new__(dataset_cls)
    state = payload.get('state', {})
    if hasattr(dataset, '__setstate__'):
        dataset.__setstate__(state)
    else:
        dataset.__dict__.update(state)
    return dataset


def write_runtime_dataset_cache(cache_path, datasets, metadata):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        'version': DATASET_CACHE_VERSION,
        'datasets': {
            split_name: serialize_cached_dataset(dataset)
            for split_name, dataset in (datasets or {}).items()
            if dataset is not None
        },
        'metadata': metadata,
    }
    temp_path = f"{cache_path}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, cache_path)


def parse_list_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    return [str(value).strip()]


def parse_timeframe_cap_config(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            normalize_timeframe_label(key): int(val)
            for key, val in value.items()
            if normalize_timeframe_label(key) is not None
        }
    caps = {}
    for item in parse_list_arg(value):
        if '=' not in item:
            continue
        timeframe, raw_cap = item.split('=', 1)
        normalized = normalize_timeframe_label(timeframe)
        if normalized is None:
            continue
        caps[normalized] = int(raw_cap)
    return caps


def parse_timeframe_threshold_config(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            normalize_timeframe_label(key): float(val)
            for key, val in value.items()
            if normalize_timeframe_label(key) is not None
        }
    thresholds = {}
    for item in parse_list_arg(value):
        if '=' not in item:
            continue
        timeframe, raw_threshold = item.split('=', 1)
        normalized = normalize_timeframe_label(timeframe)
        if normalized is None:
            continue
        thresholds[normalized] = float(raw_threshold)
    return thresholds


def resolve_normalized_timeframes(value):
    normalized = [normalize_timeframe_label(item) for item in parse_list_arg(value)]
    return [item for item in normalized if item]


def resolve_training_filters(args):
    market = getattr(args, 'market', 'legacy_multiasset')
    assets = parse_list_arg(getattr(args, 'assets', None))
    if not assets and market != 'ashare':
        assets = list(DEFAULT_MULTI_ASSETS)
    timeframes = [normalize_timeframe_label(item) for item in parse_list_arg(getattr(args, 'timeframes', None))]
    if not timeframes and market == 'ashare':
        timeframes = list(DEFAULT_ASHARE_TIMEFRAMES)
    return market, assets, [item for item in timeframes if item]


def discover_runtime_files(args):
    market, assets, timeframes = resolve_training_filters(args)
    records = discover_training_files(
        data_dir=args.data_dir,
        market=market,
        assets=assets,
        timeframes=timeframes,
        training_subdir=getattr(args, 'training_subdir', None),
    )
    max_files = getattr(args, 'max_files', None)
    if max_files:
        records = records[:max_files]
    return records


def resolve_split_labels(args, include_final_holdout=False):
    if getattr(args, 'split_scheme', 'time') != 'rolling_recent_v1':
        split_label = getattr(args, 'split_label', None)
        return [split_label] if split_label else [None]
    requested = parse_list_arg(getattr(args, 'split_labels', None))
    if requested:
        unknown = [item for item in requested if item not in ROLLING_RECENT_SPLITS]
        if unknown:
            raise ValueError(f'Unsupported rolling split labels: {unknown}')
        labels = requested
    else:
        labels = [label for label in ROLLING_RECENT_SPLITS if include_final_holdout or label != 'final_holdout']
    if include_final_holdout and 'final_holdout' not in labels:
        labels = list(labels) + ['final_holdout']
    if not include_final_holdout:
        labels = [label for label in labels if label != 'final_holdout']
    return labels


def expand_runtime_records(records, args, include_final_holdout=False):
    if not uses_recent_runtime_splits(args):
        return [dict(record) for record in records]
    expanded = []
    for split_label in resolve_split_labels(args, include_final_holdout=include_final_holdout):
        for record in records:
            expanded_record = dict(record)
            expanded_record['split_label'] = split_label
            expanded.append(expanded_record)
    return expanded


def resolve_global_horizon_grid(args):
    if getattr(args, 'market', 'legacy_multiasset') != 'ashare' and not uses_recent_runtime_splits(args):
        return None
    if not getattr(args, 'horizon_search_spec', None):
        return None
    return build_global_horizon_grid(getattr(args, 'horizon_search_spec', None))


def recommend_horizon_family(task_stats, h_mode_std=None):
    recommended_family = task_stats.get('recommended_family')
    if recommended_family in {'single_cycle', 'adaptive_resonance'}:
        return recommended_family
    mode_mass = float(task_stats.get('mode_mass', 0.0))
    iqr = float(task_stats.get('iqr', 0.0))
    h_mode = float(task_stats.get('h_mode', 0.0))
    stable_single_cycle = bool(
        mode_mass >= 0.68 and
        iqr <= 0.25 * max(h_mode, 1.0) and
        (h_mode_std is None or float(h_mode_std) <= 6.0)
    )
    return 'single_cycle' if stable_single_cycle else 'adaptive_resonance'


def create_market_datasets(record, args, global_horizon_grid=None):
    market = getattr(args, 'market', 'legacy_multiasset')
    use_recent_runtime = uses_recent_runtime_splits(args)
    if market == 'ashare' or use_recent_runtime:
        cache_path = None
        if dataset_cache_enabled(args):
            cache_path = build_runtime_dataset_cache_path(record, args, global_horizon_grid=global_horizon_grid)
            cached_payload = try_load_runtime_dataset_cache(cache_path)
            if cached_payload is not None:
                datasets = cached_payload.get('datasets', {})
                metadata = dict(cached_payload.get('metadata', {}) or {})
                metadata['split_label'] = record.get('split_label') or getattr(args, 'split_label', None)
                return datasets.get('train'), datasets.get('val'), datasets.get('test'), metadata
        datasets, metadata = create_ashare_dataset_splits(
            file_path=record['path'],
            window_size=args.window_size,
            horizon=getattr(args, 'horizon', 10),
            train_end=getattr(args, 'train_end', None),
            val_end=getattr(args, 'val_end', None),
            test_start=getattr(args, 'test_start', None),
            fast_full=args.fast_full,
            return_metadata=True,
            dataset_profile=getattr(args, 'dataset_profile', 'iterA2'),
            split_scheme=getattr(args, 'split_scheme', 'time'),
            split_label=record.get('split_label') or getattr(args, 'split_label', None),
            horizon_search_spec=getattr(args, 'horizon_search_spec', None),
            horizon_family_mode=getattr(args, 'horizon_family_mode', 'legacy'),
            global_horizon_grid=global_horizon_grid,
        )
        metadata['split_label'] = record.get('split_label') or getattr(args, 'split_label', None)
        if cache_path:
            try:
                write_runtime_dataset_cache(cache_path, datasets, metadata)
            except Exception as exc:
                print(f"[DATASET CACHE] Failed to write {cache_path}: {exc}")
        return datasets.get('train'), datasets.get('val'), datasets.get('test'), metadata

    train_ds, test_ds = create_rolling_datasets(
        record['path'],
        window_size=args.window_size,
        horizon=getattr(args, 'horizon', 10),
        fast_full=args.fast_full,
    )
    return train_ds, test_ds, None, {
        'asset_code': record.get('asset_code'),
        'timeframe': record.get('timeframe'),
        'split_label': record.get('split_label'),
    }


def prewarm_runtime_dataset_cache(runtime_records, args, global_horizon_grid=None):
    if not dataset_cache_enabled(args) or getattr(args, 'skip_dataset_cache_prewarm', False):
        return
    missing_records = []
    for record in runtime_records:
        cache_path = build_runtime_dataset_cache_path(record, args, global_horizon_grid=global_horizon_grid)
        if not os.path.exists(cache_path):
            missing_records.append((record, cache_path))
    if not missing_records:
        print('[DATASET CACHE] Runtime dataset cache already warm.')
        return
    print(f"[DATASET CACHE] Prewarming {len(missing_records)} runtime dataset artifacts...")
    for index, (record, _) in enumerate(missing_records, start=1):
        train_ds = val_ds = test_ds = dataset_meta = None
        try:
            train_ds, val_ds, test_ds, dataset_meta = create_market_datasets(
                record,
                args,
                global_horizon_grid=global_horizon_grid,
            )
        finally:
            del train_ds, val_ds, test_ds, dataset_meta
            gc.collect()
            torch.cuda.empty_cache()
        if index % 20 == 0 or index == len(missing_records):
            print(f"[DATASET CACHE] Prewarm progress {index}/{len(missing_records)}")


def build_capped_dataset(dataset, args, timeframe_label):
    cap_config = parse_timeframe_cap_config(getattr(args, 'per_timeframe_train_cap', None))
    sample_cap = cap_config.get(normalize_timeframe_label(timeframe_label))
    if sample_cap and len(dataset) > 0:
        if len(dataset) < sample_cap:
            indices = torch.randint(0, len(dataset), (sample_cap,)).tolist()
        else:
            indices = torch.randperm(len(dataset))[:sample_cap].tolist()
        return Subset(dataset, indices)
    return dataset


def resolve_train_sample_target(dataset, args, timeframe_label):
    dataset_size = int(len(dataset))
    if dataset_size <= 0:
        return 0
    cap_config = parse_timeframe_cap_config(getattr(args, 'per_timeframe_train_cap', None))
    sample_cap = cap_config.get(normalize_timeframe_label(timeframe_label))
    if sample_cap:
        return max(int(sample_cap), 0)
    return dataset_size


def resolve_dataloader_pin_memory():
    return bool(torch.cuda.is_available())


def resolve_dataloader_num_workers(args):
    return max(int(getattr(args, 'num_workers', 0) or 0), 0)


def resolve_dataloader_prefetch_factor(args):
    return max(int(getattr(args, 'prefetch_factor', 2) or 2), 2)


def build_dataloader_runtime_kwargs(args):
    kwargs = {
        'pin_memory': resolve_dataloader_pin_memory(),
    }
    num_workers = resolve_dataloader_num_workers(args)
    if num_workers > 0:
        kwargs['num_workers'] = num_workers
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = resolve_dataloader_prefetch_factor(args)
    return kwargs


class BalancedConcatSampler(Sampler):
    def __init__(self, dataset_lengths, target_samples, seed=42):
        self.dataset_lengths = [int(length) for length in dataset_lengths]
        self.target_samples = [int(samples) for samples in target_samples]
        if len(self.dataset_lengths) != len(self.target_samples):
            raise ValueError('dataset_lengths and target_samples must have the same length')
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(sum(self.target_samples))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled_indices = []
        offset = 0
        for dataset_length, target_samples in zip(self.dataset_lengths, self.target_samples):
            if dataset_length <= 0 or target_samples <= 0:
                offset += max(dataset_length, 0)
                continue
            if target_samples > dataset_length:
                local_indices = torch.randint(0, dataset_length, (target_samples,), generator=generator)
            elif target_samples == dataset_length:
                local_indices = torch.randperm(dataset_length, generator=generator)
            else:
                local_indices = torch.randperm(dataset_length, generator=generator)[:target_samples]
            sampled_indices.append(local_indices + offset)
            offset += dataset_length
        if not sampled_indices:
            return iter(())
        merged = torch.cat(sampled_indices)
        if merged.numel() > 1:
            merged = merged[torch.randperm(merged.numel(), generator=generator)]
        return iter(merged.tolist())


def build_train_loader(dataset, args, timeframe_label):
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kwargs = build_dataloader_runtime_kwargs(args)
    target_samples = resolve_train_sample_target(dataset, args, timeframe_label)
    if target_samples <= 0:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            generator=g,
            **loader_kwargs,
        )
    if target_samples != len(dataset):
        replacement = len(dataset) < target_samples
        sampler = RandomSampler(
            dataset,
            replacement=replacement,
            num_samples=target_samples,
            generator=g,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=False,
            generator=g,
            **loader_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        **loader_kwargs,
    )


def build_global_train_loader(train_dataset_specs, args):
    if not train_dataset_specs:
        raise RuntimeError('No train dataset specs supplied for global train loader.')
    datasets = []
    dataset_lengths = []
    target_samples = []
    samples_by_timeframe = defaultdict(int)
    plan = []
    for spec in train_dataset_specs:
        dataset = spec['dataset']
        timeframe_label = normalize_timeframe_label(spec.get('timeframe_label')) or 'unknown'
        dataset_size = int(len(dataset))
        effective_samples = resolve_train_sample_target(dataset, args, timeframe_label)
        if dataset_size <= 0 or effective_samples <= 0:
            continue
        datasets.append(dataset)
        dataset_lengths.append(dataset_size)
        target_samples.append(effective_samples)
        samples_by_timeframe[timeframe_label] += int(effective_samples)
        plan.append(
            {
                'asset_code': spec.get('asset_code'),
                'timeframe_label': timeframe_label,
                'split_label': spec.get('split_label'),
                'data_path': spec.get('data_path'),
                'dataset_size': dataset_size,
                'effective_samples': int(effective_samples),
            }
        )
    if not datasets:
        raise RuntimeError('No non-empty training datasets available after applying cap semantics.')
    concat_dataset = ConcatDataset(datasets)
    sampler = BalancedConcatSampler(dataset_lengths, target_samples, seed=args.seed)
    loader = DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        **build_dataloader_runtime_kwargs(args),
    )
    return loader, {
        'dataset_count': len(datasets),
        'sample_count': len(sampler),
        'samples_by_timeframe': dict(sorted(samples_by_timeframe.items())),
        'datasets': plan,
        'sampler': sampler,
    }


def build_eval_loader(dataset, args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
        **build_dataloader_runtime_kwargs(args),
    )


# SECTION: FORWARD


def _move_payload_to_device(payload, device):
    if payload is None:
        return None
    moved = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def unpack_batch(batch, device):
    horizon_payload = None
    if len(batch) == 7:
        features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags, horizon_payload = batch
    elif len(batch) == 6:
        features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags = batch
    else:
        raise ValueError(f'Unexpected batch structure: {len(batch)}')
    return (
        features_seq.to(device, non_blocking=True),
        batch_y.to(device, non_blocking=True),
        batch_aux.to(device, non_blocking=True),
        batch_sigma.to(device, non_blocking=True),
        batch_weights.to(device, non_blocking=True).unsqueeze(1),
        batch_flags.to(device, non_blocking=True),
        _move_payload_to_device(horizon_payload, device),
    )


def _selected_horizon_values(debug_info, horizon_payload):
    if debug_info is None or horizon_payload is None:
        return {}
    grid = horizon_payload.get('global_horizon_grid')
    if grid is None:
        return {}
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    grid = grid.to(dtype=torch.float32)
    values = {}
    for key in ('selected_horizon_breakout', 'selected_horizon_reversion'):
        selected = debug_info.get(key)
        if selected is None:
            continue
        selected = selected.detach().long()
        if selected.dim() == 0:
            selected = selected.unsqueeze(0)
        selected = selected.clamp(min=0, max=grid.size(1) - 1)
        values[f'{key}_value'] = torch.gather(grid, 1, selected.unsqueeze(1)).squeeze(1)
    return values


def append_debug_batches(bucket, debug_info, horizon_payload=None):
    if debug_info is None:
        return
    computed = _selected_horizon_values(debug_info, horizon_payload)
    for debug_key in DEBUG_BATCH_KEYS:
        debug_value = computed.get(debug_key, debug_info.get(debug_key))
        if debug_value is None:
            continue
        if isinstance(debug_value, torch.Tensor):
            debug_value = debug_value.detach().float().cpu().numpy()
        else:
            debug_value = np.asarray(debug_value, dtype=np.float32)
        bucket['debug_batches'][debug_key].append(debug_value)


def _tensor_is_finite(value):
    if value is None:
        return True
    if not torch.is_tensor(value):
        try:
            return bool(np.all(np.isfinite(np.asarray(value, dtype=np.float32))))
        except Exception:
            return True
    return bool(torch.isfinite(value).all().item())


def _mapping_is_finite(values):
    if not values:
        return True
    for value in values.values():
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all().item()):
                return False
            continue
        try:
            if not np.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            continue
    return True


def _count_non_finite_gradients(model):
    invalid = 0
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is not None and not bool(torch.isfinite(grad).all().item()):
            invalid += 1
    return invalid


def forward_model(kan, features_seq, args, horizon_payload=None, return_debug=False):
    horizon_prior = horizon_payload.get('horizon_prior') if horizon_payload else None
    valid_horizon_mask = horizon_payload.get('valid_horizon_mask') if horizon_payload else None
    family_mode = getattr(args, 'horizon_family_mode', None) if horizon_payload else None
    if return_debug:
        pred, aux_pred, debug_info = kan(
            features_seq,
            return_aux=True,
            return_debug=True,
            horizon_prior=horizon_prior,
            family_mode=family_mode,
            valid_horizon_mask=valid_horizon_mask,
        )
        return pred, aux_pred, debug_info
    pred, aux_pred = kan(
        features_seq,
        return_aux=True,
        horizon_prior=horizon_prior,
        family_mode=family_mode,
        valid_horizon_mask=valid_horizon_mask,
    )
    return pred, aux_pred, None


# SECTION: CHECKPOINT


def get_best_raw_path(args):
    return os.path.join(
        args.save_dir,
        getattr(args, 'best_raw_name', getattr(args, 'best_name', 'khaos_kan_best.pth')),
    )


def get_best_gate_path(args):
    return os.path.join(
        args.save_dir,
        getattr(args, 'best_gate_name', 'khaos_kan_best_gate.pth'),
    )


def get_resume_path(args):
    resume_path = getattr(args, 'resume_path', None)
    if resume_path:
        return resume_path
    return os.path.join(args.save_dir, getattr(args, 'resume_name', 'khaos_kan_resume.pth'))


def save_resume_checkpoint(
    resume_path,
    kan,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch,
    best_val_loss,
    best_score,
    best_gate_val_loss,
    best_gate_score,
    no_improve_epochs,
    latest_metrics,
    device,
    completed=False,
    best_checkpoint=None,
    best_score_components=None,
    fitted_thresholds=None,
    catastrophic_state=None,
):
    torch.save(
        {
            'model_state_dict': kan.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'args': vars(args),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_score': best_score,
            'best_raw_val_loss': best_val_loss,
            'best_raw_score': best_score,
            'best_gate_val_loss': best_gate_val_loss,
            'best_gate_score': best_gate_score,
            'no_improve_epochs': no_improve_epochs,
            'latest_metrics': latest_metrics,
            'best_checkpoint': best_checkpoint,
            'best_score_components': best_score_components,
            'fitted_thresholds': fitted_thresholds,
            'completed_epoch': epoch,
            'catastrophic_state': catastrophic_state or {},
            'feature_names': PHYSICS_FEATURE_NAMES,
            'completed': completed,
            'config_fingerprint': getattr(args, 'config_fingerprint', None),
            'dataset_cache_fingerprint': getattr(args, 'dataset_cache_fingerprint', None),
            'global_horizon_grid': getattr(args, 'global_horizon_grid', None),
            'env': {
                'torch': torch.__version__,
                'cuda': torch.version.cuda if torch.cuda.is_available() else None,
                'device': str(device),
            },
        },
        resume_path,
    )


def load_matching_model_weights(model, state_dict):
    current_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current_state or current_state[key].shape != value.shape:
            skipped.append(key)
            continue
        matched[key] = value
    current_state.update(matched)
    model.load_state_dict(current_state, strict=False)
    return {
        'matched_keys': len(matched),
        'skipped_keys': len(skipped),
        'sample_skipped': skipped[:8],
    }


def resolve_config_fingerprint(args, runtime_records, global_horizon_grid):
    payload = {
        'market': getattr(args, 'market', 'legacy_multiasset'),
        'training_subdir': getattr(args, 'training_subdir', None),
        'assets': parse_list_arg(getattr(args, 'assets', None)),
        'timeframes': resolve_normalized_timeframes(getattr(args, 'timeframes', None)),
        'window_size': getattr(args, 'window_size', None),
        'arch_version': getattr(args, 'arch_version', None),
        'dataset_profile': getattr(args, 'dataset_profile', None),
        'loss_profile': getattr(args, 'loss_profile', None),
        'constraint_profile': getattr(args, 'constraint_profile', None),
        'score_profile': getattr(args, 'score_profile', None),
        'split_scheme': getattr(args, 'split_scheme', None),
        'split_labels': [record.get('split_label') for record in runtime_records if record.get('split_label') is not None],
        'horizon_search_spec': getattr(args, 'horizon_search_spec', None),
        'horizon_family_mode': getattr(args, 'horizon_family_mode', None),
        'global_horizon_grid': global_horizon_grid,
        'records': [
            {
                'asset_code': record.get('asset_code'),
                'timeframe': record.get('timeframe'),
                'path': os.path.abspath(record['path']),
                'split_label': record.get('split_label'),
            }
            for record in runtime_records
        ],
    }
    return hashlib.sha1(
        json.dumps(make_json_safe(payload), ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()[:16]


def resolve_dataset_cache_fingerprint(args, runtime_records, global_horizon_grid):
    cache_paths = [
        build_runtime_dataset_cache_path(record, args, global_horizon_grid=global_horizon_grid)
        for record in runtime_records
    ]
    payload = {
        'dataset_cache_version': DATASET_CACHE_VERSION,
        'dataset_cache_dir': resolve_dataset_cache_dir(args),
        'runtime_cache_paths': cache_paths,
    }
    return hashlib.sha1(
        json.dumps(make_json_safe(payload), ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()[:16]


def load_baseline_reference(reference_path):
    if not reference_path:
        return None
    base_dir = reference_path
    if os.path.isfile(base_dir):
        base_dir = os.path.dirname(base_dir)
    epoch_path = os.path.join(base_dir, 'epoch_metrics.jsonl')
    per_tf_path = os.path.join(base_dir, 'per_timeframe_metrics.jsonl')
    epoch_records = read_jsonl_records(epoch_path)
    if not epoch_records:
        return None
    latest_epoch = int(epoch_records[-1].get('epoch', 0))
    per_tf_records = {
        item['timeframe']: item
        for item in read_jsonl_records(per_tf_path)
        if int(item.get('epoch', -1)) == latest_epoch and item.get('timeframe')
    }
    return {
        'overall': epoch_records[-1],
        'per_timeframe': per_tf_records,
        'epoch_metrics_path': epoch_path,
        'per_timeframe_metrics_path': per_tf_path,
    }


def resolve_baseline_signal_cap(baseline_reference, ratio, fallback=None):
    values = []
    if baseline_reference:
        overall = baseline_reference.get('overall', {})
        signal_frequency = overall.get('signal_frequency', {})
        values.extend(
            [
                overall.get('breakout_signal_frequency'),
                overall.get('reversion_signal_frequency'),
                signal_frequency.get('breakout') if isinstance(signal_frequency, dict) else None,
                signal_frequency.get('reversion') if isinstance(signal_frequency, dict) else None,
            ]
        )
    valid = [float(item) for item in values if item is not None]
    if valid:
        return max(valid) * float(ratio)
    if fallback is not None:
        return float(fallback)
    return None


def should_update_checkpoint(candidate_score, candidate_val_loss, best_score, best_val_loss, min_delta):
    return candidate_score > best_score + min_delta or (
        abs(candidate_score - best_score) <= min_delta and
        candidate_val_loss < best_val_loss - min_delta
    )


def try_resume_training(args, kan, optimizer, scheduler, scaler, device):
    start_epoch = 0
    best_raw_val_loss = float('inf')
    best_raw_score = float('-inf')
    best_gate_val_loss = float('inf')
    best_gate_score = float('-inf')
    no_improve_epochs = 0
    catastrophic_state = {'public_violation_streak': 0, 'collapse_streak': 0}
    best_checkpoint = None
    best_score_components = None
    fitted_thresholds = None
    resume_path = get_resume_path(args)
    current_fingerprint = getattr(args, 'config_fingerprint', None)
    current_cache_fingerprint = getattr(args, 'dataset_cache_fingerprint', None)
    if getattr(args, 'resume', False) and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            checkpoint_fingerprint = checkpoint.get('config_fingerprint')
            checkpoint_cache_fingerprint = checkpoint.get('dataset_cache_fingerprint')
            if (
                checkpoint_fingerprint and current_fingerprint and checkpoint_fingerprint != current_fingerprint
            ) or (
                checkpoint_cache_fingerprint and current_cache_fingerprint and
                checkpoint_cache_fingerprint != current_cache_fingerprint
            ):
                print(
                    f"[RESUME] Fingerprint mismatch. checkpoint={checkpoint_fingerprint}/{checkpoint_cache_fingerprint}, "
                    f"current={current_fingerprint}/{current_cache_fingerprint}. Skipping strict resume."
                )
            else:
                kan.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = int(checkpoint.get('epoch', 0))
                legacy_best_val_loss = checkpoint.get('best_val_loss', best_raw_val_loss)
                legacy_best_score = checkpoint.get('best_score', best_raw_score)
                best_raw_val_loss = float(checkpoint.get('best_raw_val_loss', legacy_best_val_loss))
                best_raw_score = float(checkpoint.get('best_raw_score', legacy_best_score))
                best_gate_val_loss = float(checkpoint.get('best_gate_val_loss', legacy_best_val_loss))
                best_gate_score = float(checkpoint.get('best_gate_score', legacy_best_score))
                no_improve_epochs = int(checkpoint.get('no_improve_epochs', 0))
                catastrophic_state = dict(checkpoint.get('catastrophic_state', catastrophic_state) or catastrophic_state)
                best_checkpoint = checkpoint.get('best_checkpoint')
                best_score_components = checkpoint.get('best_score_components')
                fitted_thresholds = checkpoint.get('fitted_thresholds')
                print(
                    f"[RESUME] Loaded checkpoint: epoch={start_epoch}, "
                    f"best_raw_score={best_raw_score:.4f}, best_gate_score={best_gate_score:.4f}, "
                    f"best_raw_val_loss={best_raw_val_loss:.4f}"
                )
                return {
                    'start_epoch': start_epoch,
                    'best_raw_val_loss': best_raw_val_loss,
                    'best_raw_score': best_raw_score,
                    'best_gate_val_loss': best_gate_val_loss,
                    'best_gate_score': best_gate_score,
                    'no_improve_epochs': no_improve_epochs,
                    'catastrophic_state': catastrophic_state,
                    'best_checkpoint': best_checkpoint,
                    'best_score_components': best_score_components,
                    'fitted_thresholds': fitted_thresholds,
                }
        except Exception as exc:
            print(f"[RESUME] Failed to load strict resume checkpoint: {resume_path} | {exc}")

    warm_start_enabled = bool(getattr(args, 'warm_start_weights_only', False) or getattr(args, 'warm_start_path', None))
    warm_start_path = getattr(args, 'warm_start_path', None)
    if warm_start_enabled and warm_start_path and os.path.exists(warm_start_path):
        try:
            checkpoint = torch.load(warm_start_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            report = load_matching_model_weights(kan, state_dict)
            print(
                f"[WARM START] Loaded weights only from {warm_start_path} | "
                f"matched={report['matched_keys']} skipped={report['skipped_keys']}"
            )
        except Exception as exc:
            print(f"[WARM START] Failed to load weights from {warm_start_path}: {exc}")
    return {
        'start_epoch': start_epoch,
        'best_raw_val_loss': best_raw_val_loss,
        'best_raw_score': best_raw_score,
        'best_gate_val_loss': best_gate_val_loss,
        'best_gate_score': best_gate_score,
        'no_improve_epochs': no_improve_epochs,
        'catastrophic_state': catastrophic_state,
        'best_checkpoint': best_checkpoint,
        'best_score_components': best_score_components,
        'fitted_thresholds': fitted_thresholds,
    }


def collect_horizon_guard_registry(runtime_records, args, global_horizon_grid):
    registry = {}
    for record in runtime_records:
        train_ds = val_ds = test_ds = None
        try:
            train_ds, val_ds, test_ds, dataset_meta = create_market_datasets(
                record,
                args,
                global_horizon_grid=global_horizon_grid,
            )
            update_horizon_registry(registry, dataset_meta)
        finally:
            del train_ds, val_ds, test_ds
            gc.collect()
            torch.cuda.empty_cache()
    return registry


def evaluate_single_cycle_family_guard(horizon_registry, score_timeframes=None):
    score_timeframe_set = set(score_timeframes or [])
    violations = []
    checked_records = 0
    checked_tasks = 0
    for (asset_code, timeframe, split_label), profile in sorted(horizon_registry.items()):
        if score_timeframe_set and timeframe not in score_timeframe_set:
            continue
        task_stats_map = profile.get('task_stats', {})
        if not task_stats_map:
            continue
        checked_records += 1
        for task_name, task_stats in sorted(task_stats_map.items()):
            checked_tasks += 1
            recommended_family = recommend_horizon_family(task_stats)
            if recommended_family != 'single_cycle':
                violations.append(
                    {
                        'asset_code': asset_code,
                        'timeframe': timeframe,
                        'split_label': split_label,
                        'task': task_name,
                        'recommended_family': recommended_family,
                        'mode_mass': float(task_stats.get('mode_mass', 0.0)),
                        'iqr': float(task_stats.get('iqr', 0.0)),
                        'h_mode': float(task_stats.get('h_mode', 0.0)),
                    }
                )
    return {
        'passed': bool(checked_tasks > 0 and not violations),
        'checked_records': checked_records,
        'checked_tasks': checked_tasks,
        'violations': violations,
    }


def write_horizon_summary_artifact(path, registry, global_horizon_grid, config_fingerprint):
    summary = summarize_horizon_registry(registry)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(
            make_json_safe(
                {
                    'config_fingerprint': config_fingerprint,
                    'global_horizon_grid': global_horizon_grid,
                    'records': registry,
                    'summary': summary,
                }
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return summary


def evaluate_kill_keep_review(epoch_score_summary, timeframe_summaries, args):
    review_epoch = int(getattr(args, 'kill_keep_review_epoch', 0) or 0)
    if review_epoch <= 0:
        return None
    signal_frequency = epoch_score_summary.get('signal_frequency', {})
    avg_signal_frequency = 0.5 * (
        float(signal_frequency.get('breakout', 0.0)) +
        float(signal_frequency.get('reversion', 0.0))
    )
    entropy_threshold = float(getattr(args, 'kill_keep_horizon_entropy_min', 0.0) or 0.0)
    entropy_timeframes = resolve_normalized_timeframes(
        getattr(args, 'kill_keep_horizon_entropy_timeframes', None)
    ) or ['15m', '60m']
    entropy_candidates = {}
    for timeframe in entropy_timeframes:
        summary = timeframe_summaries.get(timeframe)
        if summary is None:
            continue
        entropy_candidates[timeframe] = max(
            float(summary.get('horizon_entropy_breakout_mean', 0.0)),
            float(summary.get('horizon_entropy_reversion_mean', 0.0)),
        )
    checks = {
        'public_below_directional_violation_rate': {
            'actual': float(epoch_score_summary.get('public_below_directional_violation_rate', 0.0)),
            'threshold': float(getattr(args, 'kill_keep_public_violation_rate_max', 1.0)),
            'passed': float(epoch_score_summary.get('public_below_directional_violation_rate', 0.0)) <
            float(getattr(args, 'kill_keep_public_violation_rate_max', 1.0)),
        },
        'avg_signal_frequency': {
            'actual': avg_signal_frequency,
            'threshold': float(getattr(args, 'kill_keep_signal_frequency_max', 1.0)),
            'passed': avg_signal_frequency < float(getattr(args, 'kill_keep_signal_frequency_max', 1.0)),
        },
        'timeframe_60m_composite': {
            'actual': timeframe_summaries.get('60m', {}).get('composite_score'),
            'threshold': float(getattr(args, 'kill_keep_timeframe_60m_composite_min', 0.0)),
            'passed': timeframe_summaries.get('60m', {}).get('composite_score') is not None and
            float(timeframe_summaries.get('60m', {}).get('composite_score', 0.0)) >
            float(getattr(args, 'kill_keep_timeframe_60m_composite_min', 0.0)),
        },
        'horizon_entropy': {
            'actual': entropy_candidates,
            'threshold': entropy_threshold,
            'passed': any(value > entropy_threshold for value in entropy_candidates.values()),
        },
    }
    return {
        'review_epoch': review_epoch,
        'passed': all(item['passed'] for item in checks.values()),
        'checks': checks,
        'entropy_timeframes': entropy_timeframes,
    }


def compute_summary_signal_frequency(summary):
    signal_frequency = summary.get('signal_frequency', {})
    breakout_signal = summary.get('breakout_signal_frequency')
    reversion_signal = summary.get('reversion_signal_frequency')
    if breakout_signal is None and isinstance(signal_frequency, dict):
        breakout_signal = signal_frequency.get('breakout')
    if reversion_signal is None and isinstance(signal_frequency, dict):
        reversion_signal = signal_frequency.get('reversion')
    return 0.5 * (float(breakout_signal or 0.0) + float(reversion_signal or 0.0))


def compute_score_timeframe_avg_signal_frequency(timeframe_summaries, score_timeframes=None, fallback_summary=None):
    candidate_labels = [label for label in (score_timeframes or []) if label in timeframe_summaries]
    if not candidate_labels:
        candidate_labels = sorted(timeframe_summaries.keys())
    if candidate_labels:
        values = [compute_summary_signal_frequency(timeframe_summaries[label]) for label in candidate_labels]
        return float(np.mean(values)), candidate_labels
    if fallback_summary is not None:
        return compute_summary_signal_frequency(fallback_summary), []
    return 0.0, []


def compute_standard_score_veto(epoch_score_summary, timeframe_summaries, args, score_timeframes=None):
    avg_signal_frequency, used_timeframes = compute_score_timeframe_avg_signal_frequency(
        timeframe_summaries=timeframe_summaries,
        score_timeframes=score_timeframes,
        fallback_summary=epoch_score_summary,
    )
    diagnostic_only = is_modern_score_profile(getattr(args, 'score_profile', 'default'))
    veto_reasons = []
    checks = {}

    signal_threshold = getattr(args, 'kill_keep_signal_frequency_max', None)
    if signal_threshold is not None:
        signal_threshold = float(signal_threshold)
        signal_passed = avg_signal_frequency <= signal_threshold
        checks['avg_signal_frequency'] = {
            'actual': avg_signal_frequency,
            'threshold': signal_threshold,
            'passed': signal_passed,
            'timeframes': used_timeframes,
        }
        if not diagnostic_only and not signal_passed:
            veto_reasons.append(f'avg_signal_frequency({avg_signal_frequency:.3f}>{signal_threshold:.3f})')

    public_violation_threshold = getattr(args, 'kill_keep_public_violation_rate_max', None)
    if public_violation_threshold is not None and float(public_violation_threshold) < 1.0:
        public_violation_actual = float(epoch_score_summary.get('public_below_directional_violation_rate', 0.0))
        public_violation_threshold = float(public_violation_threshold)
        public_violation_passed = public_violation_actual <= public_violation_threshold
        checks['public_below_directional_violation_rate'] = {
            'actual': public_violation_actual,
            'threshold': public_violation_threshold,
            'passed': public_violation_passed,
        }
        if not diagnostic_only and not public_violation_passed:
            veto_reasons.append(
                f'public_below_directional_violation_rate({public_violation_actual:.3f}>{public_violation_threshold:.3f})'
            )

    timeframe_60m_threshold = getattr(args, 'kill_keep_timeframe_60m_composite_min', None)
    timeframe_60m_summary = timeframe_summaries.get('60m', {})
    timeframe_60m_breakout_precision = None
    timeframe_60m_reversion_precision = None
    timeframe_60m_hard_negative_rate = None
    if timeframe_60m_summary:
        breakout_metrics = timeframe_60m_summary.get('breakout_metrics', {}) or {}
        reversion_metrics = timeframe_60m_summary.get('reversion_metrics', {}) or {}
        if breakout_metrics and reversion_metrics:
            timeframe_60m_breakout_precision = float(breakout_metrics.get('precision', 0.0))
            timeframe_60m_reversion_precision = float(reversion_metrics.get('precision', 0.0))
            timeframe_60m_hard_negative_rate = 0.5 * (
                float(breakout_metrics.get('hard_negative_rate', 0.0)) +
                float(reversion_metrics.get('hard_negative_rate', 0.0))
            )
            checks['timeframe_60m_min_precision_diagnostic'] = {
                'actual': min(timeframe_60m_breakout_precision, timeframe_60m_reversion_precision),
                'breakout_precision': timeframe_60m_breakout_precision,
                'reversion_precision': timeframe_60m_reversion_precision,
                'threshold': None,
                'passed': True,
            }
            checks['timeframe_60m_hard_negative_rate_diagnostic'] = {
                'actual': timeframe_60m_hard_negative_rate,
                'threshold': None,
                'passed': True,
            }
    if timeframe_60m_threshold is not None and float(timeframe_60m_threshold) > 0.0:
        timeframe_60m_actual = timeframe_60m_summary.get('composite_score')
        timeframe_60m_threshold = float(timeframe_60m_threshold)
        timeframe_60m_passed = timeframe_60m_actual is not None and float(timeframe_60m_actual) >= timeframe_60m_threshold
        checks['timeframe_60m_composite'] = {
            'actual': float(timeframe_60m_actual) if timeframe_60m_actual is not None else None,
            'threshold': timeframe_60m_threshold,
            'passed': timeframe_60m_passed,
        }
        if not diagnostic_only and not timeframe_60m_passed:
            actual_repr = 'missing' if timeframe_60m_actual is None else f'{float(timeframe_60m_actual):.3f}'
            veto_reasons.append(f'timeframe_60m_composite({actual_repr}<{timeframe_60m_threshold:.3f})')

    breakout_space_threshold = getattr(args, 'kill_keep_breakout_signal_space_min', None)
    if breakout_space_threshold is not None and float(breakout_space_threshold) > 0.0:
        breakout_space_actual = float(epoch_score_summary.get('breakout_signal_space_mean', 0.0))
        breakout_space_threshold = float(breakout_space_threshold)
        breakout_space_passed = breakout_space_actual >= breakout_space_threshold
        checks['breakout_signal_space_mean'] = {
            'actual': breakout_space_actual,
            'threshold': breakout_space_threshold,
            'passed': breakout_space_passed,
        }
        if not diagnostic_only and not breakout_space_passed:
            veto_reasons.append(
                f'breakout_signal_space_mean({breakout_space_actual:.3f}<{breakout_space_threshold:.3f})'
            )

    reversion_space_threshold = getattr(args, 'kill_keep_reversion_signal_space_min', None)
    if reversion_space_threshold is not None and float(reversion_space_threshold) > 0.0:
        reversion_space_actual = float(epoch_score_summary.get('reversion_signal_space_mean', 0.0))
        reversion_space_threshold = float(reversion_space_threshold)
        reversion_space_passed = reversion_space_actual >= reversion_space_threshold
        checks['reversion_signal_space_mean'] = {
            'actual': reversion_space_actual,
            'threshold': reversion_space_threshold,
            'passed': reversion_space_passed,
        }
        if not diagnostic_only and not reversion_space_passed:
            veto_reasons.append(
                f'reversion_signal_space_mean({reversion_space_actual:.3f}<{reversion_space_threshold:.3f})'
            )

    return {
        'passed': len(veto_reasons) == 0,
        'reasons': veto_reasons,
        'checks': checks,
        'avg_signal_frequency': avg_signal_frequency,
        'score_timeframes': used_timeframes,
    }


def compute_recent_precision_score(
    fold_summaries,
    timeframe_summaries,
    baseline_reference,
    args,
):
    ordered_labels = [label for label in RECENT_FOLD_ORDER if label in fold_summaries]
    if not ordered_labels:
        ordered_labels = sorted(fold_summaries.keys())[-2:]
    if not ordered_labels:
        return 0.0, {'mode': 'empty'}, {'passed': True, 'reasons': [], 'checks': {}}

    recent_min_precision = []
    recent_avg_precision = []
    recent_direction = []
    recent_hn = []
    recent_signal = []
    recent_public_violation = []
    recent_public_feasibility = []
    recent_directional_support_rate = []
    recent_directional_floor_quality = []
    recent_structural_guard_quality = []
    public_violation_cap = float(getattr(args, 'public_violation_cap', 0.20))
    signal_cap = resolve_baseline_signal_cap(
        baseline_reference=baseline_reference,
        ratio=float(getattr(args, 'signal_frequency_cap_ratio', 0.70)),
        fallback=getattr(args, 'signal_frequency_cap', None),
    )
    diagnostics = {'folds': {}}
    for label in ordered_labels:
        summary = fold_summaries[label]
        breakout_precision = float(summary['breakout_metrics']['precision'])
        reversion_precision = float(summary['reversion_metrics']['precision'])
        recent_min_precision.append(min(breakout_precision, reversion_precision))
        recent_avg_precision.append(0.5 * (breakout_precision + reversion_precision))
        recent_direction.append(float(summary['direction_metrics']['macro_f1']))
        recent_hn.append(
            0.5 * (
                float(summary['breakout_metrics']['hard_negative_rate']) +
                float(summary['reversion_metrics']['hard_negative_rate'])
            )
        )
        recent_signal.append(
            0.5 * (
                float(summary['breakout_signal_frequency']) +
                float(summary['reversion_signal_frequency'])
            )
        )
        recent_public_violation.append(float(summary['public_below_directional_violation_rate']))
        structural_components = compute_iter14_structural_components(summary)
        recent_public_feasibility.append(structural_components['public_feasibility'])
        recent_directional_support_rate.append(structural_components['directional_support_rate'])
        recent_directional_floor_quality.append(structural_components['directional_floor_quality'])
        recent_structural_guard_quality.append(structural_components['structural_quality'])
        diagnostics['folds'][label] = {
            'breakout_precision': breakout_precision,
            'reversion_precision': reversion_precision,
            'public_below_directional_violation_rate': float(summary['public_below_directional_violation_rate']),
            'directional_support_rate': structural_components['directional_support_rate'],
        }

    current_hn = float(np.mean(recent_hn)) if recent_hn else 1.0
    baseline_hn = None
    if baseline_reference:
        overall = baseline_reference.get('overall', {})
        baseline_hn_values = [overall.get('breakout_hard_negative_rate'), overall.get('reversion_hard_negative_rate')]
        baseline_hn_values = [float(item) for item in baseline_hn_values if item is not None]
        if baseline_hn_values:
            baseline_hn = float(np.mean(baseline_hn_values))
    if baseline_hn is not None and baseline_hn > 1e-6:
        hard_negative_improvement = max((baseline_hn - current_hn) / baseline_hn, 0.0)
    else:
        hard_negative_improvement = max(1.0 - current_hn, 0.0)

    current_signal = float(np.mean(recent_signal)) if recent_signal else 0.0
    signal_health = compute_signal_health(current_signal, signal_cap)

    timeframe_60m = timeframe_summaries.get('60m')
    timeframe_60m_min_precision = 0.0
    timeframe_60m_breakout_precision = 0.0
    timeframe_60m_reversion_precision = 0.0
    timeframe_60m_hard_negative_rate = 1.0
    if timeframe_60m:
        timeframe_60m_breakout_precision = float(timeframe_60m['breakout_metrics']['precision'])
        timeframe_60m_reversion_precision = float(timeframe_60m['reversion_metrics']['precision'])
        timeframe_60m_min_precision = min(
            timeframe_60m_breakout_precision,
            timeframe_60m_reversion_precision,
        )
        timeframe_60m_hard_negative_rate = 0.5 * (
            float(timeframe_60m['breakout_metrics']['hard_negative_rate']) +
            float(timeframe_60m['reversion_metrics']['hard_negative_rate'])
        )
    timeframe_60m_hard_negative_quality = max(0.0, 1.0 - timeframe_60m_hard_negative_rate)
    components = {
        'recent_min_precision': float(np.mean(recent_min_precision)) if recent_min_precision else 0.0,
        'recent_avg_precision': float(np.mean(recent_avg_precision)) if recent_avg_precision else 0.0,
        'recent_public_feasibility': float(np.mean(recent_public_feasibility)) if recent_public_feasibility else 0.0,
        'recent_directional_support_rate': float(np.mean(recent_directional_support_rate)) if recent_directional_support_rate else 0.0,
        'recent_directional_floor_quality': float(np.mean(recent_directional_floor_quality)) if recent_directional_floor_quality else 0.0,
        'recent_structural_guard_quality': float(np.mean(recent_structural_guard_quality)) if recent_structural_guard_quality else 0.0,
        'timeframe_60m_breakout_precision': timeframe_60m_breakout_precision,
        'timeframe_60m_reversion_precision': timeframe_60m_reversion_precision,
        'timeframe_60m_min_precision': timeframe_60m_min_precision,
        'timeframe_60m_hard_negative_rate': timeframe_60m_hard_negative_rate,
        'timeframe_60m_hard_negative_quality': timeframe_60m_hard_negative_quality,
        'hard_negative_improvement': hard_negative_improvement,
        'signal_health': signal_health,
        'direction_consistency': float(np.mean(recent_direction)) if recent_direction else 0.0,
        'signal_frequency': current_signal,
        'signal_cap': signal_cap,
        'public_violation_recent_mean': float(np.mean(recent_public_violation)) if recent_public_violation else 0.0,
    }
    if getattr(args, 'score_profile', 'default') == 'iter12_guarded_precision_first':
        score = (
            0.24 * components['recent_min_precision'] +
            0.18 * components['recent_avg_precision'] +
            0.14 * components['timeframe_60m_min_precision'] +
            0.10 * components['timeframe_60m_hard_negative_quality'] +
            0.12 * components['recent_public_feasibility'] +
            0.12 * components['recent_directional_floor_quality'] +
            0.06 * components['hard_negative_improvement'] +
            0.02 * components['signal_health'] +
            0.02 * components['direction_consistency']
        )
    elif is_modern_score_profile(getattr(args, 'score_profile', 'default')):
        score = (
            0.26 * components['recent_min_precision'] +
            0.16 * components['recent_avg_precision'] +
            0.12 * components['timeframe_60m_min_precision'] +
            0.10 * components['timeframe_60m_hard_negative_quality'] +
            0.16 * components['recent_public_feasibility'] +
            0.08 * components['recent_directional_support_rate'] +
            0.06 * components['direction_consistency'] +
            0.06 * components['signal_health']
        )
    else:
        score = (
            0.30 * components['recent_min_precision'] +
            0.20 * components['recent_avg_precision'] +
            0.20 * components['timeframe_60m_min_precision'] +
            0.15 * components['hard_negative_improvement'] +
            0.10 * components['signal_health'] +
            0.05 * components['direction_consistency']
        )
    diagnostics['public_violation_cap'] = {
        'actual': components['public_violation_recent_mean'],
        'threshold': public_violation_cap,
        'passed': components['public_violation_recent_mean'] <= public_violation_cap,
    }
    diagnostics['signal_health'] = {
        'actual': components['signal_health'],
        'signal_frequency': current_signal,
        'signal_cap': signal_cap,
        'passed': components['signal_health'] > 0.0,
    }
    diagnostics['timeframe_60m_min_precision_diagnostic'] = {
        'actual': components['timeframe_60m_min_precision'],
        'breakout_precision': timeframe_60m_breakout_precision,
        'reversion_precision': timeframe_60m_reversion_precision,
        'passed': True,
    }
    diagnostics['timeframe_60m_hard_negative_rate_diagnostic'] = {
        'actual': timeframe_60m_hard_negative_rate,
        'passed': True,
    }
    if is_modern_score_profile(getattr(args, 'score_profile', 'default')):
        return score, components, {'passed': True, 'reasons': [], 'checks': diagnostics, 'mode': 'diagnostic_only'}
    return score, components, {'passed': True, 'reasons': [], 'checks': diagnostics}


def compute_catastrophic_guard(
    epoch_score_summary,
    fold_summaries,
    args,
    state=None,
    non_finite_detected=False,
):
    state = dict(state or {})
    public_streak = int(state.get('public_violation_streak', 0))
    collapse_streak = int(state.get('collapse_streak', 0))
    reasons = []
    checks = {}

    recent_labels = [label for label in RECENT_FOLD_ORDER if label in fold_summaries]
    if not recent_labels:
        recent_labels = sorted(fold_summaries.keys())[-2:]
    recent_public_values = [
        float(fold_summaries[label].get('public_below_directional_violation_rate', 0.0))
        for label in recent_labels
    ]
    recent_public_mean = float(np.mean(recent_public_values)) if recent_public_values else 0.0
    fold_public_cap = 0.65
    recent_public_cap = float(getattr(args, 'catastrophic_public_violation_cap', 0.50))
    fold_public_breach = [value for value in recent_public_values if value > fold_public_cap]
    public_condition = recent_public_mean > recent_public_cap
    public_streak = public_streak + 1 if public_condition else 0
    checks['catastrophic_public_violation'] = {
        'recent_labels': recent_labels,
        'recent_public_violation_mean': recent_public_mean,
        'mean_threshold': recent_public_cap,
        'fold_threshold': fold_public_cap,
        'consecutive_failures': public_streak,
        'passed': not fold_public_breach and public_streak < 2,
    }
    if fold_public_breach:
        reasons.append('recent_fold_public_violation_exploded')
    elif public_streak >= 2:
        reasons.append('recent_public_violation_mean_persisted')

    pred_breakout_std = float(epoch_score_summary.get('pred_breakout_std', 0.0))
    pred_reversion_std = float(epoch_score_summary.get('pred_reversion_std', 0.0))
    catastrophic_signal_floor = float(getattr(args, 'catastrophic_signal_floor', 0.003))
    breakout_signal_frequency = float(epoch_score_summary.get('breakout_signal_frequency', 0.0))
    reversion_signal_frequency = float(epoch_score_summary.get('reversion_signal_frequency', 0.0))
    collapsed_now = (
        pred_breakout_std < 1e-3 and
        pred_reversion_std < 1e-3 and
        breakout_signal_frequency < catastrophic_signal_floor and
        reversion_signal_frequency < catastrophic_signal_floor
    )
    collapse_streak = collapse_streak + 1 if collapsed_now else 0
    checks['catastrophic_output_collapse'] = {
        'pred_breakout_std': pred_breakout_std,
        'pred_reversion_std': pred_reversion_std,
        'breakout_signal_frequency': breakout_signal_frequency,
        'reversion_signal_frequency': reversion_signal_frequency,
        'signal_floor': catastrophic_signal_floor,
        'consecutive_failures': collapse_streak,
        'passed': collapse_streak < 2,
    }
    if collapse_streak >= 2:
        reasons.append('output_collapse_persisted')

    checks['catastrophic_non_finite'] = {
        'detected': bool(non_finite_detected),
        'passed': not bool(non_finite_detected),
    }
    if non_finite_detected:
        reasons.append('non_finite_detected')

    return {
        'passed': len(reasons) == 0,
        'reasons': reasons,
        'checks': checks,
        'state': {
            'public_violation_streak': public_streak,
            'collapse_streak': collapse_streak,
        },
        'mode': 'catastrophic_only',
    }


def evaluate_dataset_loader(kan, eval_loader, criterion, args, device, use_debug_metrics):
    file_bucket = build_metric_bucket()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        for batch in eval_loader:
            features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags, horizon_payload = unpack_batch(batch, device)
            batch_size_val = int(batch_y.size(0))
            psi_t = features_seq[:, -1, :]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                pred, aux_pred, debug_info = forward_model(
                    kan,
                    features_seq,
                    args,
                    horizon_payload=horizon_payload,
                    return_debug=use_debug_metrics,
                )
                loss_unweighted, rank_loss, l_dict = criterion(
                    pred,
                    aux_pred,
                    batch_y,
                    batch_aux,
                    psi_t,
                    batch_flags,
                    batch_sigma,
                    debug_info=debug_info,
                    horizon_payload=horizon_payload,
                )
                reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
            if not _tensor_is_finite(pred) or not _tensor_is_finite(aux_pred):
                file_bucket['skip_counters']['pred_non_finite'] += 1
                continue
            if not (
                _tensor_is_finite(loss_unweighted) and
                _tensor_is_finite(rank_loss) and
                _tensor_is_finite(reg_loss) and
                _tensor_is_finite(loss) and
                _mapping_is_finite(l_dict)
            ):
                file_bucket['skip_counters']['loss_non_finite'] += 1
                continue
            batch_log = dict(l_dict)
            batch_log['total_loss'] = float(loss.item())
            batch_log['batch_size'] = batch_size_val
            file_bucket['logs'].append(batch_log)
            # In EDL, compute Expected Probability (Belief) from evidence
            # Expected Probability = (evidence_1 + 1) / (evidence_0 + evidence_1 + 2)
            evidence_0 = pred[..., 0]
            evidence_1 = pred[..., 1]
            expected_prob = (evidence_1 + 1.0) / (evidence_0 + evidence_1 + 2.0)
            
            file_bucket['preds'].append(expected_prob.detach().float().cpu().numpy())
            file_bucket['aux_preds'].append(aux_pred.detach().float().cpu().numpy())
            file_bucket['targets'].append(batch_y.detach().cpu().numpy())
            file_bucket['aux_targets'].append(batch_aux.detach().cpu().numpy())
            file_bucket['flags'].append(batch_flags.detach().cpu().numpy())
            append_debug_batches(file_bucket, debug_info=debug_info, horizon_payload=horizon_payload)
    return file_bucket


def evaluate_final_holdout(
    args,
    base_records,
    kan,
    criterion,
    device,
    global_horizon_grid,
    use_debug_metrics,
    use_direction_metrics,
):
    if not uses_recent_runtime_splits(args):
        return None
    overall_bucket = build_metric_bucket()
    timeframe_buckets = defaultdict(build_metric_bucket)
    processed = 0
    for record in base_records:
        holdout_record = dict(record)
        holdout_record['split_label'] = 'final_holdout'
        try:
            _, _, test_ds, dataset_meta = create_market_datasets(holdout_record, args, global_horizon_grid=global_horizon_grid)
            if test_ds is None or len(test_ds) == 0:
                continue
            eval_loader = build_eval_loader(test_ds, args)
            bucket = evaluate_dataset_loader(kan, eval_loader, criterion, args, device, use_debug_metrics)
            merge_metric_bucket(overall_bucket, bucket)
            timeframe = normalize_timeframe_label(dataset_meta.get('timeframe') or record.get('timeframe')) or 'unknown'
            merge_metric_bucket(timeframe_buckets[timeframe], bucket)
            processed += 1
            del test_ds, eval_loader
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[FINAL HOLDOUT] Failed on {record['path']}: {exc}")
    if processed == 0:
        return None
    return {
        'processed_files': processed,
        'overall': summarize_metric_bucket(
            overall_bucket,
            score_profile=getattr(args, 'score_profile', 'default'),
            use_direction_metrics=use_direction_metrics,
        ),
        'by_timeframe': {
            timeframe: summarize_metric_bucket(
                bucket,
                score_profile=getattr(args, 'score_profile', 'default'),
                use_direction_metrics=use_direction_metrics,
            )
            for timeframe, bucket in sorted(timeframe_buckets.items())
        },
    }


# SECTION: HORIZON SUMMARY


def update_horizon_registry(registry, dataset_meta):
    horizon_profile = dataset_meta.get('horizon_profile')
    split_label = dataset_meta.get('split_label')
    timeframe = normalize_timeframe_label(dataset_meta.get('timeframe'))
    asset_code = dataset_meta.get('asset_code')
    if not horizon_profile or not split_label or not timeframe or not asset_code:
        return
    key = (asset_code, timeframe, split_label)
    if key not in registry:
        registry[key] = make_json_safe(horizon_profile)


def summarize_horizon_registry(registry):
    grouped = {}
    for (_, timeframe, split_label), profile in registry.items():
        for task_name, task_stats in profile.get('task_stats', {}).items():
            grouped.setdefault((timeframe, task_name), {'distributions': [], 'mode_mass': [], 'iqr': [], 'h_mode': []})
            distribution = {
                int(item['horizon']): float(item['prob'])
                for item in task_stats.get('distribution', [])
            }
            grouped[(timeframe, task_name)]['distributions'].append(distribution)
            grouped[(timeframe, task_name)]['mode_mass'].append(float(task_stats.get('mode_mass', 0.0)))
            grouped[(timeframe, task_name)]['iqr'].append(float(task_stats.get('iqr', 0.0)))
            grouped[(timeframe, task_name)]['h_mode'].append(float(task_stats.get('h_mode', 0.0)))
    summary = {}
    for (timeframe, task_name), values in grouped.items():
        all_horizons = sorted({h for dist in values['distributions'] for h in dist.keys()})
        distribution_mean = [
            {
                'horizon': int(horizon),
                'prob': float(np.mean([dist.get(horizon, 0.0) for dist in values['distributions']])),
            }
            for horizon in all_horizons
        ]
        h_mode_mean = float(np.mean(values['h_mode'])) if values['h_mode'] else 0.0
        h_mode_std = float(np.std(values['h_mode'])) if values['h_mode'] else 0.0
        mode_mass_mean = float(np.mean(values['mode_mass'])) if values['mode_mass'] else 0.0
        iqr_mean = float(np.mean(values['iqr'])) if values['iqr'] else 0.0
        summary.setdefault(timeframe, {})[task_name] = {
            'distribution_mean': distribution_mean,
            'mode_mass_mean': mode_mass_mean,
            'iqr_mean': iqr_mean,
            'h_mode_mean': h_mode_mean,
            'h_mode_std': h_mode_std,
            'recommended_family': recommend_horizon_family(
                {
                    'mode_mass': mode_mass_mean,
                    'iqr': iqr_mean,
                    'h_mode': h_mode_mean,
                },
                h_mode_std=h_mode_std,
            ),
            'fold_count': len(values['distributions']),
        }
    return summary


# SECTION: TRAIN


def train(args):
    set_seed(args.seed, args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    market = getattr(args, 'market', 'legacy_multiasset')
    training_subdir = getattr(args, 'training_subdir', None)
    train_data_dir = resolve_training_ready_dir(args.data_dir, market=market, training_subdir=training_subdir)
    os.makedirs(train_data_dir, exist_ok=True)
    ready_files = glob.glob(os.path.join(train_data_dir, '*.csv'))
    if len(ready_files) == 0 and market != 'ashare':
        processed_files = []
        raw_files = glob.glob(os.path.join(args.data_dir, '**', '*.csv'), recursive=True)
        for file_path in raw_files:
            if 'training_ready' in file_path or os.path.getsize(file_path) < 1024:
                continue
            processed_files.extend(process_multi_timeframe(file_path, train_data_dir))
        print(f"Generated {len(processed_files)} processed files into {train_data_dir}.")

    base_records = discover_runtime_files(args)
    if args.test_mode:
        base_records = base_records[:1]
        args.epochs = min(args.epochs, 1)
        print("Running in test mode.")
    if not base_records:
        raise RuntimeError(f'No training files found under {train_data_dir} for market={market}.')

    runtime_records = expand_runtime_records(base_records, args, include_final_holdout=False)
    if args.test_mode and getattr(args, 'split_scheme', 'time') == 'rolling_recent_v1':
        runtime_records = runtime_records[:min(len(runtime_records), 2)]
    print(f"Loading {len(runtime_records)} runtime jobs from {len(base_records)} base files.")

    global_horizon_grid = resolve_global_horizon_grid(args)
    setattr(args, 'global_horizon_grid', global_horizon_grid)
    if not getattr(args, 'config_fingerprint', None):
        setattr(args, 'config_fingerprint', resolve_config_fingerprint(args, runtime_records, global_horizon_grid))
    print(f"Config fingerprint: {args.config_fingerprint}")
    if global_horizon_grid:
        print(f"Global horizon grid size: {len(global_horizon_grid)}")
    prewarm_runtime_dataset_cache(runtime_records, args, global_horizon_grid=global_horizon_grid)
    if getattr(args, 'prewarm_dataset_cache_only', False):
        print('[DATASET CACHE] Prewarm-only mode enabled; skipping model training.')
        return

    kan = KHAOS_KAN(
        input_dim=len(PHYSICS_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        output_dim=2,
        layers=args.layers,
        grid_size=args.grid_size,
        arch_version=getattr(args, 'arch_version', 'iterA2_base'),
        horizon_count=len(global_horizon_grid) if global_horizon_grid else 1,
        horizon_family_mode=getattr(args, 'horizon_family_mode', 'legacy'),
    ).to(device)
    optimizer = optim.AdamW(
        kan.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = PhysicsLoss(
        profile=getattr(args, 'loss_profile', 'default'),
        constraint_profile=getattr(args, 'constraint_profile', 'default'),
        family_mode=getattr(args, 'horizon_family_mode', 'legacy'),
    )
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    use_direction_metrics = getattr(args, 'arch_version', 'iterA2_base') in {
        'iterA3_multiscale',
        'iterA4_multiscale',
        'iterA5_multiscale',
    }
    use_debug_metrics = use_direction_metrics or getattr(args, 'constraint_profile', 'default') != 'default' or bool(global_horizon_grid)
    score_profile = getattr(args, 'score_profile', 'default')
    score_timeframes = resolve_normalized_timeframes(getattr(args, 'score_timeframes', None))
    score_timeframe_set = set(score_timeframes)
    aux_timeframes = resolve_normalized_timeframes(getattr(args, 'aux_timeframes', None))
    aux_timeframe_set = set(aux_timeframes)
    promotion_overall_threshold = getattr(args, 'promotion_overall_composite_threshold', None)
    promotion_timeframe_thresholds = parse_timeframe_threshold_config(
        getattr(args, 'promotion_timeframe_composite_thresholds', None)
    )
    baseline_reference = load_baseline_reference(getattr(args, 'baseline_reference_dir', None))
    resume_path = get_resume_path(args)
    best_raw_path = get_best_raw_path(args)
    best_gate_path = get_best_gate_path(args)
    epoch_metrics_path = os.path.join(args.save_dir, getattr(args, 'epoch_metrics_name', 'epoch_metrics.jsonl'))
    per_timeframe_metrics_path = os.path.join(args.save_dir, getattr(args, 'per_timeframe_metrics_name', 'per_timeframe_metrics.jsonl'))
    per_fold_metrics_path = os.path.join(args.save_dir, getattr(args, 'per_fold_metrics_name', 'per_fold_metrics.jsonl'))
    per_asset_metrics_path = os.path.join(args.save_dir, getattr(args, 'per_asset_metrics_name', 'per_asset_metrics.jsonl'))
    final_holdout_metrics_path = os.path.join(args.save_dir, getattr(args, 'final_holdout_metrics_name', 'final_holdout_metrics.json'))
    horizon_summary_path = os.path.join(args.save_dir, getattr(args, 'horizon_summary_name', 'horizon_discovery_summary.json'))
    horizon_family_guard_passed = True
    if bool(getattr(args, 'enforce_horizon_family_guard', False)) and getattr(args, 'horizon_family_mode', 'legacy') == 'single_cycle':
        print('[HORIZON GUARD] Validating single_cycle against discovered horizon family recommendations...')
        guard_registry = collect_horizon_guard_registry(runtime_records, args, global_horizon_grid)
        guard_summary = write_horizon_summary_artifact(
            horizon_summary_path,
            guard_registry,
            global_horizon_grid,
            getattr(args, 'config_fingerprint', None),
        )
        guard_result = evaluate_single_cycle_family_guard(guard_registry, score_timeframes=score_timeframes)
        horizon_family_guard_passed = guard_result['passed']
        if not horizon_family_guard_passed:
            violation_preview = ', '.join(
                f"{item['asset_code']}/{item['timeframe']}/{item['split_label']}/{item['task']}->{item['recommended_family']}"
                for item in guard_result['violations'][:6]
            )
            raise RuntimeError(
                'single_cycle horizon family guard failed; use adaptive_resonance instead. '
                f"checked_tasks={guard_result['checked_tasks']} violations={len(guard_result['violations'])} "
                f"summary_path={horizon_summary_path} preview=[{violation_preview}]"
            )
        print(
            f"[HORIZON GUARD] Passed. checked_records={guard_result['checked_records']} "
            f"checked_tasks={guard_result['checked_tasks']}"
        )
        del guard_summary
        gc.collect()
        torch.cuda.empty_cache()

    resume_state = try_resume_training(args, kan, optimizer, scheduler, scaler, device)
    start_epoch = resume_state['start_epoch']
    best_raw_val_loss = resume_state['best_raw_val_loss']
    best_raw_score = resume_state['best_raw_score']
    best_gate_val_loss = resume_state['best_gate_val_loss']
    best_gate_score = resume_state['best_gate_score']
    no_improve_epochs = resume_state['no_improve_epochs']
    if start_epoch == 0:
        for metric_path in (epoch_metrics_path, per_timeframe_metrics_path, per_fold_metrics_path, per_asset_metrics_path):
            if metric_path and os.path.exists(metric_path):
                os.remove(metric_path)


    latest_metrics = None
    horizon_registry = {}

    print("Loading all datasets for global mixed training...")
    train_dataset_specs = []
    eval_jobs = []
    threshold_fit_jobs = []
    for job_idx, record in enumerate(runtime_records):
        data_path = record['path']
        split_label = record.get('split_label')
        try:
            train_ds, val_ds, test_ds, dataset_meta = create_market_datasets(
                record,
                args,
                global_horizon_grid=global_horizon_grid,
            )
            update_horizon_registry(horizon_registry, dataset_meta)
            eval_ds = val_ds if val_ds is not None else test_ds
            timeframe_label = normalize_timeframe_label(dataset_meta.get('timeframe') or record.get('timeframe')) or 'unknown'
            
            if train_ds is not None and len(train_ds) >= max(1, min(args.batch_size, 8)):
                train_dataset_specs.append({
                    'dataset': train_ds,
                    'timeframe_label': timeframe_label,
                    'split_label': split_label,
                    'data_path': data_path,
                    'asset_code': dataset_meta.get('asset_code') or record.get('asset_code'),
                })
                if not getattr(args, 'disable_threshold_fit', False):
                    threshold_fit_jobs.append({
                        'eval_ds': build_capped_dataset(train_ds, args, timeframe_label),
                        'timeframe_label': timeframe_label,
                        'split_label': split_label,
                        'data_path': data_path,
                    })
            
            if eval_ds is not None and len(eval_ds) > 0:
                eval_jobs.append({
                    'eval_ds': eval_ds,
                    'timeframe_label': timeframe_label,
                    'split_label': split_label,
                    'data_path': data_path,
                    'asset_code': dataset_meta.get('asset_code') or record.get('asset_code'),
                })
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Error processing {data_path} split={split_label}: {exc}")
            continue

    if not train_dataset_specs:
        raise RuntimeError("No valid training datasets found.")

    global_train_loader, global_train_plan = build_global_train_loader(train_dataset_specs, args)
    global_train_sampler = global_train_plan['sampler']
    print(
        f"Global training loader created: {global_train_plan['sample_count']} sampled items "
        f"across {global_train_plan['dataset_count']} datasets."
    )
    print(f"Global training balance by timeframe: {global_train_plan['samples_by_timeframe']}")

    for epoch in range(start_epoch, args.epochs):

        print(f"\\n========== EPOCH {epoch + 1}/{args.epochs} ==========")
        epoch_train_logs = []
        epoch_all_bucket = build_metric_bucket()
        epoch_timeframe_buckets = defaultdict(build_metric_bucket)
        epoch_fold_buckets = defaultdict(build_metric_bucket)
        train_skip_counters = defaultdict(int)
        
        kan.train()
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch, args.epochs)
        if hasattr(kan, 'set_epoch_progress'):
            kan.set_epoch_progress(epoch, args.epochs)
        global_train_sampler.set_epoch(epoch)
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        for batch_idx, batch in enumerate(global_train_loader):
            features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags, horizon_payload = unpack_batch(batch, device)
            batch_size_val = int(batch_y.size(0))
            psi_t = features_seq[:, -1, :]
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                pred, aux_pred, debug_info = forward_model(
                    kan,
                    features_seq,
                    args,
                    horizon_payload=horizon_payload,
                    return_debug=use_debug_metrics,
                )
                if not _tensor_is_finite(pred) or not _tensor_is_finite(aux_pred):
                    train_skip_counters['pred_non_finite'] += 1
                    continue

                loss_unweighted, rank_loss, l_dict = criterion(
                    pred,
                    aux_pred,
                    batch_y,
                    batch_aux,
                    psi_t,
                    batch_flags,
                    batch_sigma,
                    debug_info=debug_info,
                    horizon_payload=horizon_payload,
                )
                reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
            if not (
                _tensor_is_finite(loss_unweighted) and
                _tensor_is_finite(rank_loss) and
                _tensor_is_finite(reg_loss) and
                _tensor_is_finite(loss) and
                _mapping_is_finite(l_dict)
            ):
                train_skip_counters['loss_non_finite'] += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            invalid_grad_count = _count_non_finite_gradients(kan)
            if invalid_grad_count:
                train_skip_counters['grad_non_finite'] += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(kan.parameters(), max_norm=getattr(args, 'grad_clip', 1.0))
            grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            if not np.isfinite(grad_norm_value):
                train_skip_counters['grad_clip_non_finite'] += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            l_dict_log = dict(l_dict)
            l_dict_log['total_loss'] = float(loss.item())
            l_dict_log['batch_size'] = batch_size_val
            epoch_train_logs.append(l_dict_log)
            
            if batch_idx % 100 == 0:
                print(
                    f"  [EPOCH {epoch + 1}] batch {batch_idx}/{len(global_train_loader)} loss={loss.item():.6f}"
                )
        epoch_effective_train_samples_by_timeframe = defaultdict(int, global_train_plan['samples_by_timeframe'])

        kan.eval()
        frozen_thresholds_by_timeframe = {}
        if threshold_fit_jobs:
            epoch_threshold_fit_buckets = defaultdict(build_metric_bucket)
            for threshold_job in threshold_fit_jobs:
                timeframe_label = threshold_job['timeframe_label']
                try:
                    fit_loader = build_eval_loader(threshold_job['eval_ds'], args)
                    fit_bucket = evaluate_dataset_loader(
                        kan=kan,
                        eval_loader=fit_loader,
                        criterion=criterion,
                        args=args,
                        device=device,
                        use_debug_metrics=use_debug_metrics,
                    )
                    merge_metric_bucket(epoch_threshold_fit_buckets[timeframe_label], fit_bucket)
                except Exception as exc:
                    print(f"Error threshold-fitting {os.path.basename(threshold_job['data_path'])} timeframe={timeframe_label}: {exc}")
                    continue
            for timeframe, bucket in epoch_threshold_fit_buckets.items():
                if not bucket['preds']:
                    continue
                fit_summary = summarize_metric_bucket(
                    bucket,
                    score_profile=score_profile,
                    use_direction_metrics=use_direction_metrics,
                )
                frozen_thresholds_by_timeframe[timeframe] = {
                    'breakout': float(fit_summary['breakout_metrics']['threshold']),
                    'reversion': float(fit_summary['reversion_metrics']['threshold']),
                }
        processed_jobs = 0
        for eval_job in eval_jobs:
            data_path = eval_job['data_path']
            split_label = eval_job['split_label']
            timeframe_label = eval_job['timeframe_label']
            
            try:
                eval_loader = build_eval_loader(eval_job['eval_ds'], args)
                file_bucket = evaluate_dataset_loader(
                    kan=kan,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    args=args,
                    device=device,
                    use_debug_metrics=use_debug_metrics,
                )
                file_summary = summarize_metric_bucket(
                    file_bucket,
                    score_profile=score_profile,
                    use_direction_metrics=use_direction_metrics,
                    frozen_thresholds=frozen_thresholds_by_timeframe.get(timeframe_label),
                )
                
                print(
                    f"  -> {os.path.basename(data_path)} [{split_label or 'default'}|{timeframe_label}] "
                    f"val={file_summary['avg_val_loss']:.6f} "
                    f"precision={file_summary['breakout_metrics']['precision']:.4f}/{file_summary['reversion_metrics']['precision']:.4f} "
                    f"composite={file_summary['composite_score']:.4f}"
                )
                
                merge_metric_bucket(epoch_all_bucket, file_bucket)
                merge_metric_bucket(epoch_timeframe_buckets[timeframe_label], file_bucket)
                if split_label:
                    merge_metric_bucket(epoch_fold_buckets[split_label], file_bucket)
                append_jsonl(
                    per_asset_metrics_path,
                    {
                        'epoch': epoch + 1,
                        'asset': eval_job.get('asset_code') or 'unknown',
                        'timeframe': timeframe_label,
                        'split_label': split_label or 'default',
                        'file': os.path.basename(data_path),
                        'constraint_profile': getattr(args, 'constraint_profile', 'default'),
                        'arch_version': getattr(args, 'arch_version', 'iterA2_base'),
                        'dataset_profile': getattr(args, 'dataset_profile', 'iterA2'),
                        'loss_profile': getattr(args, 'loss_profile', 'default'),
                        'score_profile': score_profile,
                        'thresholds_frozen': bool(file_summary.get('thresholds_frozen', False)),
                        'frozen_thresholds': file_summary.get('frozen_thresholds'),
                        'sample_count': file_summary['sample_count'],
                        'breakout_f1': file_summary['breakout_metrics']['f1'],
                        'reversion_f1': file_summary['reversion_metrics']['f1'],
                        'breakout_precision': file_summary['breakout_metrics']['precision'],
                        'reversion_precision': file_summary['reversion_metrics']['precision'],
                        'breakout_hard_negative_rate': file_summary['breakout_metrics']['hard_negative_rate'],
                        'reversion_hard_negative_rate': file_summary['reversion_metrics']['hard_negative_rate'],
                        'breakout_oversignal': file_summary['breakout_oversignal'],
                        'reversion_oversignal': file_summary['reversion_oversignal'],
                        'breakout_signal_space_mean': file_summary['breakout_signal_space_mean'],
                        'reversion_signal_space_mean': file_summary['reversion_signal_space_mean'],
                        'breakout_signal_quality_mean': file_summary['breakout_signal_quality_mean'],
                        'reversion_signal_quality_mean': file_summary['reversion_signal_quality_mean'],
                        'direction_macro_f1': file_summary['direction_metrics']['macro_f1'],
                        'signal_frequency': file_summary['signal_frequency'],
                        'pred_rev_mean': file_summary['pred_rev_mean'],
                        'pred_rev_event_mean': file_summary['pred_rev_event_mean'],
                        'directional_floor_mean': file_summary['directional_floor_mean'],
                        'directional_floor_reversion_event_mean': file_summary['directional_floor_reversion_event_mean'],
                        'reversion_event_count': file_summary['reversion_event_count'],
                        'selected_horizon_breakout_value_mean': file_summary['selected_horizon_breakout_value_mean'],
                        'selected_horizon_reversion_value_mean': file_summary['selected_horizon_reversion_value_mean'],
                        'horizon_entropy_breakout_mean': file_summary['horizon_entropy_breakout_mean'],
                        'horizon_entropy_reversion_mean': file_summary['horizon_entropy_reversion_mean'],
                        'public_below_directional_violation_rate': file_summary['public_below_directional_violation_rate'],
                        'composite_score': file_summary['composite_score'],
                    },
                )
                processed_jobs += 1
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"Error evaluating {data_path} split={split_label}: {exc}")
                continue

        if processed_jobs == 0 or not epoch_all_bucket['preds']:

            print("No valid runtime jobs processed in this epoch.")
            continue

        timeframe_summaries = {
            timeframe: summarize_metric_bucket(
                bucket,
                score_profile=score_profile,
                use_direction_metrics=use_direction_metrics,
                frozen_thresholds=frozen_thresholds_by_timeframe.get(timeframe),
            )
            for timeframe, bucket in sorted(epoch_timeframe_buckets.items())
        }
        fold_summaries = {
            fold: summarize_metric_bucket(
                bucket,
                score_profile=score_profile,
                use_direction_metrics=use_direction_metrics,
            )
            for fold, bucket in sorted(epoch_fold_buckets.items())
        }
        overall_summary = summarize_metric_bucket(
            epoch_all_bucket,
            score_profile=score_profile,
            use_direction_metrics=use_direction_metrics,
        )
        if score_timeframe_set:
            score_bucket = build_metric_bucket()
            for timeframe, bucket in epoch_timeframe_buckets.items():
                if timeframe in score_timeframe_set:
                    merge_metric_bucket(score_bucket, bucket)
            if score_bucket['preds']:
                epoch_score_summary = summarize_metric_bucket(
                    score_bucket,
                    score_profile=score_profile,
                    use_direction_metrics=use_direction_metrics,
                )
            else:
                epoch_score_summary = overall_summary
        else:
            epoch_score_summary = overall_summary

        if getattr(args, 'split_scheme', 'time') == 'rolling_recent_v1':
            composite_score, recent_score_components, score_veto = compute_recent_precision_score(
                fold_summaries=fold_summaries,
                timeframe_summaries=timeframe_summaries,
                baseline_reference=baseline_reference,
                args=args,
            )
        else:
            composite_score = epoch_score_summary['composite_score']
            score_veto = compute_standard_score_veto(
                epoch_score_summary=epoch_score_summary,
                timeframe_summaries=timeframe_summaries,
                args=args,
                score_timeframes=score_timeframes,
            )
            recent_score_components = {
                'mode': 'standard',
                'avg_signal_frequency': score_veto['avg_signal_frequency'],
                'score_timeframes': score_veto['score_timeframes'],
            }

        epoch_avg_loss = _metric_bucket_mean(epoch_train_logs, 'total_loss') if epoch_train_logs else 0.0
        epoch_avg_val_loss = epoch_score_summary['avg_val_loss']
        epoch_train_main = _metric_bucket_mean(epoch_train_logs, 'main') if epoch_train_logs else 0.0
        epoch_train_aux = _metric_bucket_mean(epoch_train_logs, 'aux') if epoch_train_logs else 0.0
        skipped_train_steps = int(sum(train_skip_counters.values()))
        eval_skip_counters = dict(sorted(epoch_all_bucket['skip_counters'].items()))
        print(
            f"[EPOCH {epoch + 1}] train_loss={epoch_avg_loss:.6f} val_loss={epoch_avg_val_loss:.6f} "
            f"score={composite_score:.4f} veto={score_veto['passed']}"
        )
        print(
            f"[EPOCH {epoch + 1}] effective_train_samples_by_timeframe="
            f"{dict(sorted(epoch_effective_train_samples_by_timeframe.items()))}"
        )
        if skipped_train_steps:
            print(
                f"[EPOCH {epoch + 1}] skipped_train_steps={skipped_train_steps} "
                f"detail={dict(sorted(train_skip_counters.items()))}"
            )
        if eval_skip_counters:
            print(f"[EPOCH {epoch + 1}] eval_skip_counters={eval_skip_counters}")
        if score_veto['reasons']:
            print(f"[EPOCH {epoch + 1}] veto_reasons={score_veto['reasons']}")

        promotion_scoreboard = {}
        if promotion_overall_threshold is not None:
            delta = composite_score - float(promotion_overall_threshold)
            promotion_scoreboard['overall_model_composite'] = {
                'actual': float(composite_score),
                'threshold': float(promotion_overall_threshold),
                'delta': float(delta),
                'passed': bool(delta >= 0.0),
            }
        for timeframe, threshold in promotion_timeframe_thresholds.items():
            actual = timeframe_summaries.get(timeframe, {}).get('composite_score')
            promotion_scoreboard[f'timeframe_{timeframe}_composite'] = {
                'actual': float(actual) if actual is not None else None,
                'threshold': float(threshold),
                'delta': float(actual - threshold) if actual is not None else None,
                'passed': bool(actual is not None and actual >= threshold),
            }

        for timeframe, summary in timeframe_summaries.items():
            append_jsonl(
                per_timeframe_metrics_path,
                {
                    'epoch': epoch + 1,
                    'timeframe': timeframe,
                    'score_included': timeframe in score_timeframe_set if score_timeframe_set else True,
                    'aux_timeframe': timeframe in aux_timeframe_set,
                    'constraint_profile': getattr(args, 'constraint_profile', 'default'),
                    'arch_version': getattr(args, 'arch_version', 'iterA2_base'),
                    'dataset_profile': getattr(args, 'dataset_profile', 'iterA2'),
                    'loss_profile': getattr(args, 'loss_profile', 'default'),
                    'score_profile': score_profile,
                    'thresholds_frozen': bool(summary.get('thresholds_frozen', False)),
                    'frozen_thresholds': summary.get('frozen_thresholds'),
                    'sample_count': summary['sample_count'],
                    'breakout_f1': summary['breakout_metrics']['f1'],
                    'reversion_f1': summary['reversion_metrics']['f1'],
                    'breakout_precision': summary['breakout_metrics']['precision'],
                    'reversion_precision': summary['reversion_metrics']['precision'],
                    'breakout_hard_negative_rate': summary['breakout_metrics']['hard_negative_rate'],
                    'reversion_hard_negative_rate': summary['reversion_metrics']['hard_negative_rate'],
                    'breakout_oversignal': summary['breakout_oversignal'],
                    'reversion_oversignal': summary['reversion_oversignal'],
                    'breakout_signal_space_mean': summary['breakout_signal_space_mean'],
                    'reversion_signal_space_mean': summary['reversion_signal_space_mean'],
                    'breakout_signal_quality_mean': summary['breakout_signal_quality_mean'],
                    'reversion_signal_quality_mean': summary['reversion_signal_quality_mean'],
                    'direction_macro_f1': summary['direction_metrics']['macro_f1'],
                    'signal_frequency': summary['signal_frequency'],
                    'pred_rev_mean': summary['pred_rev_mean'],
                    'pred_rev_event_mean': summary['pred_rev_event_mean'],
                    'directional_floor_mean': summary['directional_floor_mean'],
                    'directional_floor_reversion_event_mean': summary['directional_floor_reversion_event_mean'],
                    'directional_floor_quality': summary['directional_floor_quality'],
                    'public_feasibility_mean': summary['public_feasibility_mean'],
                    'structural_guard_quality': summary['structural_guard_quality'],
                    'reversion_event_count': summary['reversion_event_count'],
                    'selected_horizon_breakout_value_mean': summary['selected_horizon_breakout_value_mean'],
                    'selected_horizon_reversion_value_mean': summary['selected_horizon_reversion_value_mean'],
                    'horizon_entropy_breakout_mean': summary['horizon_entropy_breakout_mean'],
                    'horizon_entropy_reversion_mean': summary['horizon_entropy_reversion_mean'],
                    'public_below_directional_violation_rate': summary['public_below_directional_violation_rate'],
                    'composite_score': summary['composite_score'],
                },
            )
        for fold, summary in fold_summaries.items():
            append_jsonl(
                per_fold_metrics_path,
                {
                    'epoch': epoch + 1,
                    'split_label': fold,
                    'breakout_f1': summary['breakout_metrics']['f1'],
                    'reversion_f1': summary['reversion_metrics']['f1'],
                    'breakout_precision': summary['breakout_metrics']['precision'],
                    'reversion_precision': summary['reversion_metrics']['precision'],
                    'breakout_hard_negative_rate': summary['breakout_metrics']['hard_negative_rate'],
                    'reversion_hard_negative_rate': summary['reversion_metrics']['hard_negative_rate'],
                    'breakout_oversignal': summary['breakout_oversignal'],
                    'reversion_oversignal': summary['reversion_oversignal'],
                    'breakout_signal_space_mean': summary['breakout_signal_space_mean'],
                    'reversion_signal_space_mean': summary['reversion_signal_space_mean'],
                    'breakout_signal_quality_mean': summary['breakout_signal_quality_mean'],
                    'reversion_signal_quality_mean': summary['reversion_signal_quality_mean'],
                    'direction_macro_f1': summary['direction_metrics']['macro_f1'],
                    'signal_frequency': summary['signal_frequency'],
                    'pred_rev_mean': summary['pred_rev_mean'],
                    'pred_rev_event_mean': summary['pred_rev_event_mean'],
                    'directional_floor_mean': summary['directional_floor_mean'],
                    'directional_floor_reversion_event_mean': summary['directional_floor_reversion_event_mean'],
                    'directional_floor_quality': summary['directional_floor_quality'],
                    'public_feasibility_mean': summary['public_feasibility_mean'],
                    'structural_guard_quality': summary['structural_guard_quality'],
                    'reversion_event_count': summary['reversion_event_count'],
                    'selected_horizon_breakout_value_mean': summary['selected_horizon_breakout_value_mean'],
                    'selected_horizon_reversion_value_mean': summary['selected_horizon_reversion_value_mean'],
                    'public_below_directional_violation_rate': summary['public_below_directional_violation_rate'],
                    'composite_score': summary['composite_score'],
                },
            )

        horizon_summary = write_horizon_summary_artifact(
            horizon_summary_path,
            horizon_registry,
            global_horizon_grid,
            getattr(args, 'config_fingerprint', None),
        )
        best_raw_updated = False
        best_gate_updated = False
        review_result = None
        if int(getattr(args, 'kill_keep_review_epoch', 0) or 0) == epoch + 1:
            review_result = evaluate_kill_keep_review(
                epoch_score_summary=epoch_score_summary,
                timeframe_summaries=timeframe_summaries,
                args=args,
            )
            if review_result is not None:
                print(f"[EPOCH {epoch + 1}] kill_keep_review={review_result}")

        scheduler.step(epoch_avg_val_loss)
        raw_improved = should_update_checkpoint(
            composite_score,
            epoch_avg_val_loss,
            best_raw_score,
            best_raw_val_loss,
            args.early_stop_min_delta,
        )
        gate_improved = score_veto['passed'] and should_update_checkpoint(
            composite_score,
            epoch_avg_val_loss,
            best_gate_score,
            best_gate_val_loss,
            args.early_stop_min_delta,
        )
        if raw_improved:
            best_raw_score = composite_score
            best_raw_val_loss = epoch_avg_val_loss
            no_improve_epochs = 0
            best_raw_updated = True
            raw_checkpoint_payload = {
                'model_state_dict': kan.state_dict(),
                'args': vars(args),
                'dataset_manifest': base_records,
                'runtime_manifest': runtime_records,
                'val_loss': best_raw_val_loss,
                'best_score': best_raw_score,
                'selection_mode': 'best_raw',
                'config_fingerprint': getattr(args, 'config_fingerprint', None),
                'global_horizon_grid': global_horizon_grid,
                'metrics': {
                    'epoch': epoch + 1,
                    'overall_summary': epoch_score_summary,
                    'timeframe_summaries': timeframe_summaries,
                    'fold_summaries': fold_summaries,
                    'recent_score_components': recent_score_components,
                    'score_veto': score_veto,
                    'horizon_summary': horizon_summary,
                },
                'feature_names': PHYSICS_FEATURE_NAMES,
                'env': {
                    'torch': torch.__version__,
                    'cuda': torch.version.cuda if torch.cuda.is_available() else None,
                    'device': str(device),
                },
            }
            torch.save(raw_checkpoint_payload, best_raw_path)
            torch.save(raw_checkpoint_payload, os.path.join(args.save_dir, getattr(args, 'best_name', 'khaos_kan_best.pth')))
        else:
            no_improve_epochs += 1
            print(f"[EPOCH {epoch + 1}] no improvement {no_improve_epochs}/{args.early_stop_patience}")

        if gate_improved:
            best_gate_score = composite_score
            best_gate_val_loss = epoch_avg_val_loss
            best_gate_updated = True
            torch.save(
                {
                    'model_state_dict': kan.state_dict(),
                    'args': vars(args),
                    'dataset_manifest': base_records,
                    'runtime_manifest': runtime_records,
                    'val_loss': best_gate_val_loss,
                    'best_score': best_gate_score,
                    'selection_mode': 'best_gate',
                    'config_fingerprint': getattr(args, 'config_fingerprint', None),
                    'global_horizon_grid': global_horizon_grid,
                    'metrics': {
                        'epoch': epoch + 1,
                        'overall_summary': epoch_score_summary,
                        'timeframe_summaries': timeframe_summaries,
                        'fold_summaries': fold_summaries,
                        'recent_score_components': recent_score_components,
                        'score_veto': score_veto,
                        'horizon_summary': horizon_summary,
                    },
                    'feature_names': PHYSICS_FEATURE_NAMES,
                    'env': {
                        'torch': torch.__version__,
                        'cuda': torch.version.cuda if torch.cuda.is_available() else None,
                        'device': str(device),
                    },
                },
                best_gate_path,
            )

        epoch_metric_payload = {
            'epoch': epoch + 1,
            'processed_jobs': processed_jobs,
            'constraint_profile': getattr(args, 'constraint_profile', 'default'),
            'arch_version': getattr(args, 'arch_version', 'iterA2_base'),
            'dataset_profile': getattr(args, 'dataset_profile', 'iterA2'),
            'loss_profile': getattr(args, 'loss_profile', 'default'),
            'score_profile': score_profile,
            'split_scheme': getattr(args, 'split_scheme', 'time'),
            'score_timeframes': score_timeframes or sorted(timeframe_summaries.keys()),
            'aux_timeframes': aux_timeframes,
            'train_loss': epoch_avg_loss,
            'val_loss': epoch_avg_val_loss,
            'train_main': epoch_train_main,
            'train_aux': epoch_train_aux,
            'effective_train_samples_by_timeframe': dict(sorted(epoch_effective_train_samples_by_timeframe.items())),
            'skipped_train_steps': skipped_train_steps,
            'train_skip_counters': dict(sorted(train_skip_counters.items())),
            'eval_skip_counters': eval_skip_counters,
            'horizon_family_guard_passed': horizon_family_guard_passed,
            'best_raw_updated': best_raw_updated,
            'best_gate_updated': best_gate_updated,
            'sample_count': epoch_score_summary['sample_count'],
            'breakout_f1': epoch_score_summary['breakout_metrics']['f1'],
            'reversion_f1': epoch_score_summary['reversion_metrics']['f1'],
            'breakout_precision': epoch_score_summary['breakout_metrics']['precision'],
            'reversion_precision': epoch_score_summary['reversion_metrics']['precision'],
            'breakout_hard_negative_rate': epoch_score_summary['breakout_metrics']['hard_negative_rate'],
            'reversion_hard_negative_rate': epoch_score_summary['reversion_metrics']['hard_negative_rate'],
            'breakout_oversignal': epoch_score_summary['breakout_oversignal'],
            'reversion_oversignal': epoch_score_summary['reversion_oversignal'],
            'breakout_signal_space_mean': epoch_score_summary['breakout_signal_space_mean'],
            'reversion_signal_space_mean': epoch_score_summary['reversion_signal_space_mean'],
            'breakout_signal_quality_mean': epoch_score_summary['breakout_signal_quality_mean'],
            'reversion_signal_quality_mean': epoch_score_summary['reversion_signal_quality_mean'],
            'direction_macro_f1': epoch_score_summary['direction_metrics']['macro_f1'],
            'signal_frequency': epoch_score_summary['signal_frequency'],
            'pred_rev_mean': epoch_score_summary['pred_rev_mean'],
            'pred_rev_event_mean': epoch_score_summary['pred_rev_event_mean'],
            'directional_floor_mean': epoch_score_summary['directional_floor_mean'],
            'directional_floor_reversion_event_mean': epoch_score_summary['directional_floor_reversion_event_mean'],
            'directional_floor_quality': epoch_score_summary['directional_floor_quality'],
            'public_feasibility_mean': epoch_score_summary['public_feasibility_mean'],
            'structural_guard_quality': epoch_score_summary['structural_guard_quality'],
            'reversion_event_count': epoch_score_summary['reversion_event_count'],
            'public_below_directional_violation_rate': epoch_score_summary['public_below_directional_violation_rate'],
            'selected_horizon_breakout_value_mean': epoch_score_summary['selected_horizon_breakout_value_mean'],
            'selected_horizon_reversion_value_mean': epoch_score_summary['selected_horizon_reversion_value_mean'],
            'horizon_entropy_breakout_mean': epoch_score_summary['horizon_entropy_breakout_mean'],
            'horizon_entropy_reversion_mean': epoch_score_summary['horizon_entropy_reversion_mean'],
            'composite_score': composite_score,
            'all_timeframes_composite_score': overall_summary['composite_score'],
            'recent_score_components': recent_score_components,
            'score_veto': score_veto,
            'promotion_scoreboard': promotion_scoreboard,
        }
        if review_result is not None:
            epoch_metric_payload['kill_keep_review'] = review_result
        append_jsonl(epoch_metrics_path, epoch_metric_payload)

        latest_metrics = {
            'epoch': epoch + 1,
            'composite_score': composite_score,
            'val_loss': epoch_avg_val_loss,
            'overall_summary': epoch_score_summary,
            'timeframe_summaries': timeframe_summaries,
            'fold_summaries': fold_summaries,
            'recent_score_components': recent_score_components,
            'score_veto': score_veto,
            'horizon_summary': horizon_summary,
            'effective_train_samples_by_timeframe': dict(sorted(epoch_effective_train_samples_by_timeframe.items())),
            'skipped_train_steps': skipped_train_steps,
            'train_skip_counters': dict(sorted(train_skip_counters.items())),
            'eval_skip_counters': eval_skip_counters,
            'horizon_family_guard_passed': horizon_family_guard_passed,
            'best_raw_updated': best_raw_updated,
            'best_gate_updated': best_gate_updated,
        }
        if review_result is not None:
            latest_metrics['kill_keep_review'] = review_result
        save_resume_checkpoint(
            resume_path=resume_path,
            kan=kan,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch + 1,
            best_val_loss=best_raw_val_loss,
            best_score=best_raw_score,
            best_gate_val_loss=best_gate_val_loss,
            best_gate_score=best_gate_score,
            no_improve_epochs=no_improve_epochs,
            latest_metrics=latest_metrics,
            device=device,
            completed=False,
        )
        if review_result is not None and not review_result['passed']:
            print(f"[KILL/KEEP] Review failed at epoch {epoch + 1}; stopping run for label/constraint rebuild.")
            break
        if no_improve_epochs >= args.early_stop_patience:
            print(
                f"[EARLY STOP] No improvement for {args.early_stop_patience} epochs "
                f"(delta={args.early_stop_min_delta:.4f})."
            )
            break

    best_path = best_raw_path if os.path.exists(best_raw_path) else os.path.join(
        args.save_dir,
        getattr(args, 'best_name', 'khaos_kan_best.pth'),
    )
    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            kan.load_state_dict(checkpoint['model_state_dict'])
        except Exception as exc:
            print(f"[FINAL HOLDOUT] Failed to load best checkpoint: {exc}")

    final_holdout_metrics = evaluate_final_holdout(
        args=args,
        base_records=base_records,
        kan=kan,
        criterion=criterion,
        device=device,
        global_horizon_grid=global_horizon_grid,
        use_debug_metrics=use_debug_metrics,
        use_direction_metrics=use_direction_metrics,
    )
    if final_holdout_metrics is not None:
        with open(final_holdout_metrics_path, 'w', encoding='utf-8') as handle:
            json.dump(make_json_safe(final_holdout_metrics), handle, ensure_ascii=False, indent=2)
        if latest_metrics is None:
            latest_metrics = {}
        latest_metrics['final_holdout'] = final_holdout_metrics

    torch.save(
        {
            'model_state_dict': kan.state_dict(),
            'args': vars(args),
            'dataset_manifest': base_records,
            'runtime_manifest': runtime_records,
            'feature_names': PHYSICS_FEATURE_NAMES,
            'latest_metrics': latest_metrics,
            'config_fingerprint': getattr(args, 'config_fingerprint', None),
            'global_horizon_grid': global_horizon_grid,
            'env': {
                'torch': torch.__version__,
                'cuda': torch.version.cuda if torch.cuda.is_available() else None,
                'device': str(device),
            },
        },
        os.path.join(args.save_dir, getattr(args, 'final_name', 'khaos_kan_model_final.pth')),
    )
    save_resume_checkpoint(
        resume_path=resume_path,
        kan=kan,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        epoch=args.epochs,
        best_val_loss=best_raw_val_loss,
        best_score=best_raw_score,
        best_gate_val_loss=best_gate_val_loss,
        best_gate_score=best_gate_score,
        no_improve_epochs=no_improve_epochs,
        latest_metrics=latest_metrics,
        device=device,
        completed=True,
    )
    print(f"Training complete. Final model saved to {os.path.join(args.save_dir, getattr(args, 'final_name', 'khaos_kan_model_final.pth'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--market', type=str, default='legacy_multiasset')
    parser.add_argument('--training_subdir', type=str, default=None)
    parser.add_argument('--assets', type=str, default=None)
    parser.add_argument('--timeframes', type=str, default=None)
    parser.add_argument('--split_mode', type=str, default='ratio')
    parser.add_argument('--split_scheme', type=str, default='time')
    parser.add_argument('--split_label', type=str, default=None)
    parser.add_argument('--split_labels', type=str, default=None)
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--val_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--per_timeframe_train_cap', type=str, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--best_name', type=str, default='khaos_kan_best.pth')
    parser.add_argument('--best_raw_name', type=str, default='khaos_kan_best_raw.pth')
    parser.add_argument('--best_gate_name', type=str, default='khaos_kan_best_gate.pth')
    parser.add_argument('--final_name', type=str, default='khaos_kan_model_final.pth')
    parser.add_argument('--resume_name', type=str, default='khaos_kan_resume.pth')
    parser.add_argument('--epoch_metrics_name', type=str, default='epoch_metrics.jsonl')
    parser.add_argument('--per_timeframe_metrics_name', type=str, default='per_timeframe_metrics.jsonl')
    parser.add_argument('--per_fold_metrics_name', type=str, default='per_fold_metrics.jsonl')
    parser.add_argument('--per_asset_metrics_name', type=str, default='per_asset_metrics.jsonl')
    parser.add_argument('--final_holdout_metrics_name', type=str, default='final_holdout_metrics.json')
    parser.add_argument('--horizon_summary_name', type=str, default='horizon_discovery_summary.json')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--horizon_search_spec', type=str, default=None)
    parser.add_argument('--horizon_family_mode', type=str, default='legacy')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--arch_version', type=str, default='iterA2_base')
    parser.add_argument('--dataset_profile', type=str, default='iterA2')
    parser.add_argument('--loss_profile', type=str, default='default')
    parser.add_argument('--constraint_profile', type=str, default='default')
    parser.add_argument('--score_profile', type=str, default='default')
    parser.add_argument('--score_timeframes', type=str, default=None)
    parser.add_argument('--aux_timeframes', type=str, default=None)
    parser.add_argument('--promotion_overall_composite_threshold', type=float, default=None)
    parser.add_argument('--promotion_timeframe_composite_thresholds', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', default=True)
    parser.add_argument('--non_deterministic', dest='deterministic', action='store_false')
    parser.add_argument('--test_mode', action='store_true', default=False)
    parser.add_argument('--fast_full', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=2)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.002)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--warm_start_weights_only', action='store_true', default=False)
    parser.add_argument('--warm_start_path', type=str, default=None)
    parser.add_argument('--dataset_cache_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--disable_dataset_cache', action='store_true', default=False)
    parser.add_argument('--skip_dataset_cache_prewarm', action='store_true', default=False)
    parser.add_argument('--prewarm_dataset_cache_only', action='store_true', default=False)
    parser.add_argument('--disable_threshold_fit', action='store_true', default=False)
    parser.add_argument('--config_fingerprint', type=str, default=None)
    parser.add_argument('--baseline_reference_dir', type=str, default=None)
    parser.add_argument('--enforce_horizon_family_guard', action='store_true', default=False)
    parser.add_argument('--signal_frequency_cap_ratio', type=float, default=0.70)
    parser.add_argument('--signal_frequency_cap', type=float, default=None)
    parser.add_argument('--public_violation_cap', type=float, default=0.20)
    parser.add_argument('--breakout_precision_floor', type=float, default=0.0)
    parser.add_argument('--reversion_precision_floor', type=float, default=0.0)
    parser.add_argument('--gate_mode', type=str, default='disabled', choices=['soft_annealed', 'legacy_hard', 'disabled'])
    parser.add_argument('--gate_floor_breakout', type=float, default=0.0)
    parser.add_argument('--gate_floor_reversion', type=float, default=0.0)
    parser.add_argument('--gate_anneal_fraction', type=float, default=0.60)
    parser.add_argument('--horizon_search_spec', type=str, default=None)
    parser.add_argument('--kill_keep_review_epoch', type=int, default=0)
    parser.add_argument('--kill_keep_public_violation_rate_max', type=float, default=0.25)
    parser.add_argument('--kill_keep_signal_frequency_max', type=float, default=1.0)
    parser.add_argument('--kill_keep_timeframe_60m_composite_min', type=float, default=0.0)
    parser.add_argument('--kill_keep_breakout_signal_space_min', type=float, default=0.0)
    parser.add_argument('--kill_keep_reversion_signal_space_min', type=float, default=0.0)
    parser.add_argument('--kill_keep_horizon_entropy_min', type=float, default=0.0)
    parser.add_argument('--kill_keep_horizon_entropy_timeframes', type=str, default=None)
    train(parser.parse_args())
