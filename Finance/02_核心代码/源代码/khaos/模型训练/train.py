import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.amp import autocast, GradScaler
import argparse
import glob
import json
import os
import numpy as np
import random
from collections import defaultdict

# Adjust import to run as script
import sys
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from khaos.数据处理.ashare_dataset import EVENT_FLAG_INDEX, create_ashare_dataset_splits
from khaos.数据处理.ashare_support import (
    DEFAULT_ASHARE_TIMEFRAMES,
    LEGACY_ITER9_ASSETS,
    discover_training_files,
    normalize_timeframe_label,
    resolve_training_ready_dir,
)
from khaos.数据处理.data_processor import process_multi_timeframe
from khaos.数据处理.data_loader import create_rolling_datasets
from khaos.模型定义.kan import KHAOS_KAN
from khaos.模型训练.loss import PhysicsLoss
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES

DEBUG_BATCH_KEYS = (
    'blue_score',
    'purple_score',
    'direction_gate',
    'public_reversion_gate',
    'breakout_residual_gate',
    'directional_floor',
)

CONSTRAINT_STAT_BASE_KEYS = (
    'blue_over_purple_violation',
    'purple_over_blue_violation',
    'public_below_directional_violation',
    'continuation_public_violation',
)


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def compute_event_metrics(scores, event_flags, hard_negative_flags):
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
            'label_frequency': 0.0
        }
    label_frequency = float(np.mean(event_flags)) if len(event_flags) > 0 else 0.0
    thresholds = np.unique(np.quantile(scores, np.linspace(0.55, 0.95, 9)))
    best = None
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
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'event_rate': float(event_rate),
            'hard_negative_rate': float(hn_rate),
            'signal_frequency': float(np.mean(pred)),
            'label_frequency': label_frequency
        }
        if best is None or candidate['f1'] > best['f1'] or (
            candidate['f1'] == best['f1'] and candidate['hard_negative_rate'] < best['hard_negative_rate']
        ):
            best = candidate
    return best


def compute_event_quality(metrics):
    signal_gap = abs(metrics['signal_frequency'] - metrics['label_frequency'])
    return (
        metrics['f1']
        - 0.20 * metrics['hard_negative_rate']
        - 0.05 * signal_gap
    )


def compute_direction_metrics(blue_scores, purple_scores, flags):
    blue_scores = np.asarray(blue_scores, dtype=np.float64).reshape(-1)
    purple_scores = np.asarray(purple_scores, dtype=np.float64).reshape(-1)
    flags = np.asarray(flags, dtype=np.float64)
    blue_idx = EVENT_FLAG_INDEX['reversion_down_context']
    purple_idx = EVENT_FLAG_INDEX['reversion_up_context']
    if flags.ndim != 2 or flags.shape[1] <= max(blue_idx, purple_idx):
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'blue_f1': 0.0,
            'purple_f1': 0.0,
            'support': 0,
        }
    blue_mask = flags[:, blue_idx] > 0.5
    purple_mask = flags[:, purple_idx] > 0.5
    valid_mask = blue_mask | purple_mask
    if not np.any(valid_mask):
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'blue_f1': 0.0,
            'purple_f1': 0.0,
            'support': 0,
        }
    truth = np.where(blue_mask[valid_mask], 1, 0)
    pred = np.where(blue_scores[valid_mask] >= purple_scores[valid_mask], 1, 0)

    def _binary_f1(target_label):
        truth_mask = truth == target_label
        pred_mask = pred == target_label
        tp = np.sum(truth_mask & pred_mask)
        fp = np.sum(~truth_mask & pred_mask)
        fn = np.sum(truth_mask & ~pred_mask)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return 2 * precision * recall / max(precision + recall, 1e-8)

    blue_f1 = float(_binary_f1(1))
    purple_f1 = float(_binary_f1(0))
    return {
        'accuracy': float(np.mean(truth == pred)),
        'macro_f1': float((blue_f1 + purple_f1) / 2.0),
        'blue_f1': blue_f1,
        'purple_f1': purple_f1,
        'support': int(np.sum(valid_mask)),
    }


def compute_checkpoint_score(
    breakout_metrics,
    reversion_metrics,
    breakout_corr,
    reversion_corr,
    direction_macro_f1=None,
    profile='default',
):
    breakout_quality = compute_event_quality(breakout_metrics)
    reversion_quality = compute_event_quality(reversion_metrics)
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
        'targets': [],
        'flags': [],
        'logs': [],
        'debug_batches': {key: [] for key in DEBUG_BATCH_KEYS},
    }


def merge_metric_bucket(destination, source):
    destination['preds'].extend(source['preds'])
    destination['targets'].extend(source['targets'])
    destination['flags'].extend(source['flags'])
    destination['logs'].extend(source['logs'])
    for key in DEBUG_BATCH_KEYS:
        destination['debug_batches'][key].extend(source['debug_batches'][key])


def _flatten_debug_batches(batches):
    if not batches:
        return np.array([], dtype=np.float32)
    return np.concatenate([np.asarray(batch, dtype=np.float32).reshape(-1) for batch in batches]).astype(np.float32)


def _metric_bucket_mean(logs, key):
    values = [float(item[key]) for item in logs if key in item]
    if not values:
        return 0.0
    return float(np.mean(values))


def _zero_direction_metrics():
    return {
        'accuracy': 0.0,
        'macro_f1': 0.0,
        'blue_f1': 0.0,
        'purple_f1': 0.0,
        'support': 0,
    }


def summarize_metric_bucket(bucket, score_profile='default', use_direction_metrics=False):
    if not bucket['preds']:
        breakout_metrics = compute_event_metrics([], [], [])
        reversion_metrics = compute_event_metrics([], [], [])
        return {
            'sample_count': 0,
            'avg_val_loss': 0.0,
            'breakout_corr': 0.0,
            'reversion_corr': 0.0,
            'breakout_event_mean': 0.0,
            'reversion_event_mean': 0.0,
            'breakout_hard_negative_mean': 0.0,
            'reversion_hard_negative_mean': 0.0,
            'breakout_gap': 0.0,
            'reversion_gap': 0.0,
            'composite_score': 0.0,
            'breakout_metrics': breakout_metrics,
            'reversion_metrics': reversion_metrics,
            'direction_metrics': _zero_direction_metrics(),
            'signal_frequency': {'breakout': 0.0, 'reversion': 0.0},
            'label_frequency': {'breakout': 0.0, 'reversion': 0.0},
            'breakout_signal_frequency': 0.0,
            'reversion_signal_frequency': 0.0,
            'breakout_label_frequency': 0.0,
            'reversion_label_frequency': 0.0,
            'direction_gate_mean': 0.0,
            'direction_gate_std': 0.0,
            'public_reversion_gate_mean': 0.0,
            'public_reversion_gate_std': 0.0,
            'breakout_residual_gate_mean': 0.0,
            'breakout_residual_gate_std': 0.0,
            'directional_floor_mean': 0.0,
            'loss_main': 0.0,
            'loss_aux': 0.0,
            'loss_rank': 0.0,
            'loss_constraint_penalty': 0.0,
            **{
                key: 0.0
                for base_key in CONSTRAINT_STAT_BASE_KEYS
                for key in (base_key, f'{base_key}_rate')
            },
        }

    pred_np = np.vstack(bucket['preds'])
    target_np = np.vstack(bucket['targets'])
    flags_np = np.vstack(bucket['flags'])
    pred_rev_np = np.maximum(pred_np[:, 1], 0.0)
    breakout_corr = safe_corr(pred_np[:, 0], target_np[:, 0])
    reversion_corr = safe_corr(pred_rev_np, target_np[:, 1])
    breakout_event_mean = float(pred_np[flags_np[:, 0] > 0.5, 0].mean()) if np.any(flags_np[:, 0] > 0.5) else 0.0
    reversion_event_mean = float(pred_rev_np[flags_np[:, 1] > 0.5].mean()) if np.any(flags_np[:, 1] > 0.5) else 0.0
    breakout_hn_mean = float(pred_np[flags_np[:, 2] > 0.5, 0].mean()) if np.any(flags_np[:, 2] > 0.5) else 0.0
    reversion_hn_mean = float(pred_rev_np[flags_np[:, 3] > 0.5].mean()) if np.any(flags_np[:, 3] > 0.5) else 0.0
    breakout_metrics = compute_event_metrics(pred_np[:, 0], flags_np[:, 0] > 0.5, flags_np[:, 2] > 0.5)
    reversion_metrics = compute_event_metrics(pred_rev_np, flags_np[:, 1] > 0.5, flags_np[:, 3] > 0.5)

    blue_scores = _flatten_debug_batches(bucket['debug_batches']['blue_score'])
    purple_scores = _flatten_debug_batches(bucket['debug_batches']['purple_score'])
    if use_direction_metrics and blue_scores.size > 0 and purple_scores.size > 0:
        direction_metrics = compute_direction_metrics(blue_scores, purple_scores, flags_np)
    else:
        direction_metrics = _zero_direction_metrics()

    direction_gate_values = _flatten_debug_batches(bucket['debug_batches']['direction_gate'])
    public_gate_values = _flatten_debug_batches(bucket['debug_batches']['public_reversion_gate'])
    breakout_residual_gate_values = _flatten_debug_batches(bucket['debug_batches']['breakout_residual_gate'])
    directional_floor_values = _flatten_debug_batches(bucket['debug_batches']['directional_floor'])

    breakout_gap = breakout_event_mean - breakout_hn_mean
    reversion_gap = reversion_event_mean - reversion_hn_mean
    composite_score = compute_checkpoint_score(
        breakout_metrics,
        reversion_metrics,
        breakout_corr,
        reversion_corr,
        direction_metrics['macro_f1'] if use_direction_metrics else None,
        profile=score_profile,
    )

    summary = {
        'sample_count': int(pred_np.shape[0]),
        'avg_val_loss': _metric_bucket_mean(bucket['logs'], 'total_loss'),
        'breakout_corr': breakout_corr,
        'reversion_corr': reversion_corr,
        'breakout_event_mean': breakout_event_mean,
        'reversion_event_mean': reversion_event_mean,
        'breakout_hard_negative_mean': breakout_hn_mean,
        'reversion_hard_negative_mean': reversion_hn_mean,
        'breakout_gap': breakout_gap,
        'reversion_gap': reversion_gap,
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
        'breakout_signal_frequency': breakout_metrics['signal_frequency'],
        'reversion_signal_frequency': reversion_metrics['signal_frequency'],
        'breakout_label_frequency': breakout_metrics['label_frequency'],
        'reversion_label_frequency': reversion_metrics['label_frequency'],
        'direction_gate_mean': float(direction_gate_values.mean()) if direction_gate_values.size else 0.0,
        'direction_gate_std': float(direction_gate_values.std()) if direction_gate_values.size else 0.0,
        'public_reversion_gate_mean': float(public_gate_values.mean()) if public_gate_values.size else 0.0,
        'public_reversion_gate_std': float(public_gate_values.std()) if public_gate_values.size else 0.0,
        'breakout_residual_gate_mean': float(breakout_residual_gate_values.mean()) if breakout_residual_gate_values.size else 0.0,
        'breakout_residual_gate_std': float(breakout_residual_gate_values.std()) if breakout_residual_gate_values.size else 0.0,
        'directional_floor_mean': float(directional_floor_values.mean()) if directional_floor_values.size else 0.0,
        'loss_main': _metric_bucket_mean(bucket['logs'], 'main'),
        'loss_aux': _metric_bucket_mean(bucket['logs'], 'aux'),
        'loss_rank': _metric_bucket_mean(bucket['logs'], 'rank'),
        'loss_constraint_penalty': _metric_bucket_mean(bucket['logs'], 'constraint_penalty'),
    }
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
    return value


def append_jsonl(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(make_json_safe(payload), ensure_ascii=False) + '\n')


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


def resolve_normalized_timeframes(value):
    normalized = [normalize_timeframe_label(item) for item in parse_list_arg(value)]
    return [item for item in normalized if item]


def resolve_training_filters(args):
    market = getattr(args, 'market', 'legacy_multiasset')
    assets = parse_list_arg(getattr(args, 'assets', None))
    if not assets and market != 'ashare':
        assets = list(LEGACY_ITER9_ASSETS)
    timeframes = [normalize_timeframe_label(item) for item in parse_list_arg(getattr(args, 'timeframes', None))]
    if not timeframes and market == 'ashare':
        timeframes = list(DEFAULT_ASHARE_TIMEFRAMES)
    return market, assets, [item for item in timeframes if item]


def discover_runtime_files(args):
    market, assets, timeframes = resolve_training_filters(args)
    training_subdir = getattr(args, 'training_subdir', None)
    records = discover_training_files(
        data_dir=args.data_dir,
        market=market,
        assets=assets,
        timeframes=timeframes,
        training_subdir=training_subdir,
    )
    max_files = getattr(args, 'max_files', None)
    if max_files:
        records = records[:max_files]
    return records


def create_market_datasets(record, args):
    market = getattr(args, 'market', 'legacy_multiasset')
    if market == 'ashare':
        datasets, metadata = create_ashare_dataset_splits(
            file_path=record['path'],
            window_size=args.window_size,
            horizon=args.horizon,
            train_end=getattr(args, 'train_end', None),
            val_end=getattr(args, 'val_end', None),
            test_start=getattr(args, 'test_start', None),
            fast_full=args.fast_full,
            return_metadata=True,
            dataset_profile=getattr(args, 'dataset_profile', 'iterA2'),
        )
        return datasets.get('train'), datasets.get('val'), datasets.get('test'), metadata

    train_ds, test_ds = create_rolling_datasets(
        record['path'],
        window_size=args.window_size,
        horizon=args.horizon,
        fast_full=args.fast_full,
    )
    metadata = {
        'asset_code': record.get('asset_code'),
        'timeframe': record.get('timeframe'),
    }
    return train_ds, test_ds, None, metadata


def build_train_loader(dataset, args, timeframe_label):
    g = torch.Generator()
    g.manual_seed(args.seed)
    cap_config = parse_timeframe_cap_config(getattr(args, 'per_timeframe_train_cap', None))
    sample_cap = cap_config.get(normalize_timeframe_label(timeframe_label))
    if sample_cap and len(dataset) > 0:
        replacement = len(dataset) < sample_cap
        sampler = RandomSampler(
            dataset,
            replacement=replacement,
            num_samples=sample_cap,
            generator=g,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
            generator=g,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )


def build_eval_loader(dataset, args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
    )


def forward_model(kan, features_seq, return_debug=False):
    if return_debug:
        pred, aux_pred, debug_info = kan(features_seq, return_aux=True, return_debug=True)
        return pred, aux_pred, debug_info
    pred, aux_pred = kan(features_seq, return_aux=True)
    return pred, aux_pred, None

def get_resume_path(args):
    resume_path = getattr(args, 'resume_path', None)
    if resume_path:
        return resume_path
    resume_name = getattr(args, 'resume_name', 'khaos_kan_resume.pth')
    return os.path.join(args.save_dir, resume_name)

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
    no_improve_epochs,
    latest_metrics,
    device,
    completed=False
):
    torch.save({
        'model_state_dict': kan.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'args': vars(args),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_score': best_score,
        'no_improve_epochs': no_improve_epochs,
        'latest_metrics': latest_metrics,
        'feature_names': PHYSICS_FEATURE_NAMES,
        'completed': completed,
        'env': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else None,
            'device': str(device)
        }
    }, resume_path)

def try_resume_training(args, kan, optimizer, scheduler, scaler, device):
    start_epoch = 0
    best_val_loss = float('inf')
    best_score = float('-inf')
    no_improve_epochs = 0
    resume_path = get_resume_path(args)
    if not getattr(args, 'resume', False):
        return start_epoch, best_val_loss, best_score, no_improve_epochs

    if os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            kan.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
            best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
            best_score = float(checkpoint.get('best_score', best_score))
            no_improve_epochs = int(checkpoint.get('no_improve_epochs', 0))
            print(
                f"[RESUME] 已从断点恢复训练：epoch={start_epoch}, "
                f"best_score={best_score:.4f}, best_val_loss={best_val_loss:.4f}"
            )
            return start_epoch, best_val_loss, best_score, no_improve_epochs
        except Exception as exc:
            print(f"[RESUME] 断点文件不可用，忽略并重新开始：{resume_path} | {exc}")

    best_path = os.path.join(args.save_dir, getattr(args, 'best_name', 'khaos_kan_best.pth'))
    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            kan.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = int(checkpoint.get('metrics', {}).get('epoch', 0))
            best_val_loss = float(checkpoint.get('val_loss', best_val_loss))
            best_score = float(checkpoint.get('best_score', best_score))
            print(
                f"[RESUME] 未找到断点文件，已从 best checkpoint 热启动：epoch={start_epoch}, "
                f"best_score={best_score:.4f}, best_val_loss={best_val_loss:.4f}"
            )
        except Exception as exc:
            print(f"[RESUME] best checkpoint 不可热启动，改为从头训练：{best_path} | {exc}")
    else:
        print("[RESUME] 未找到可恢复的断点，将从头开始训练。")
    return start_epoch, best_val_loss, best_score, no_improve_epochs

def train(args):
    # Setup
    set_seed(args.seed, args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Unified local physics window: {args.window_size}")
    print(f"Short smoothing window reserved for visualization/stability: 2-3")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 1. Prepare Data (Multi-Timeframe)
    print("Preparing Multi-Timeframe Datasets...")
    market = getattr(args, 'market', 'legacy_multiasset')
    training_subdir = getattr(args, 'training_subdir', None)
    train_data_dir = resolve_training_ready_dir(args.data_dir, market=market, training_subdir=training_subdir)
    os.makedirs(train_data_dir, exist_ok=True)

    ready_files = glob.glob(os.path.join(train_data_dir, '*.csv'))
    if len(ready_files) == 0 and market != 'ashare':
        processed_files = []
        search_pattern = os.path.join(args.data_dir, '**', '*.csv')
        raw_files = glob.glob(search_pattern, recursive=True)
        print(f"Found {len(raw_files)} raw files.")
        for file_path in raw_files:
            if 'training_ready' in file_path:
                continue
            if os.path.getsize(file_path) < 1024:
                continue
            processed_files.extend(process_multi_timeframe(file_path, train_data_dir))
        print(f"Generated {len(processed_files)} processed files into {train_data_dir}.")

    final_records = discover_runtime_files(args)
    if args.test_mode:
        final_records = final_records[:1]
        args.epochs = 1
        print("!!! RUNNING IN FAST TEST MODE !!!")

    print(f"Loading {len(final_records)} files into Global Memory...")
    print(f"Assets included: {[os.path.basename(item['path']) for item in final_records]}")
    if not final_records:
        raise RuntimeError(f'No training files found under {train_data_dir} for market={market}.')
    
    # =========================================================================
    # WARNING: Windows has severe memory fragmentation issues with CUDA 
    # when holding large tensors in a Dataset list across multiple files.
    # To ensure stable training, we fall back to sequential file processing,
    # but still utilize the massive speedup of Offline GPU Pre-computation.
    # =========================================================================
    
    # FOR DEBUGGING WINDOWS MEMORY HANGS, ONLY PROCESS FIRST FILE
    # In order to make it run full loop, we must clean memory aggressively 
    # but to show you it works, we run on all files but clear GPU cache
    
    # We run on all filtered files as per user requirement (no truncation)
    print(f"Loading {len(final_records)} files into Global Memory...")
    print(f"Assets included: {[os.path.basename(item['path']) for item in final_records]}")
    
    # 3. Model Init
    input_dim = len(PHYSICS_FEATURE_NAMES)
    kan = KHAOS_KAN(
        input_dim=input_dim, 
        hidden_dim=args.hidden_dim,
        output_dim=2,
        layers=args.layers, 
        grid_size=args.grid_size,
        arch_version=getattr(args, 'arch_version', 'iterA2_base'),
    ).to(device)
    
    optimizer = optim.AdamW(
        kan.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    constraint_profile = getattr(args, 'constraint_profile', 'default')
    criterion = PhysicsLoss(
        profile=getattr(args, 'loss_profile', 'default'),
        constraint_profile=constraint_profile,
    )
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    use_direction_metrics = getattr(args, 'arch_version', 'iterA2_base') in {'iterA3_multiscale', 'iterA4_multiscale', 'iterA5_multiscale'}
    use_debug_metrics = use_direction_metrics or constraint_profile != 'default'
    score_profile = getattr(args, 'score_profile', 'default')
    score_timeframes = resolve_normalized_timeframes(getattr(args, 'score_timeframes', None))
    score_timeframe_set = set(score_timeframes)
    aux_timeframes = resolve_normalized_timeframes(getattr(args, 'aux_timeframes', None))
    aux_timeframe_set = set(aux_timeframes)
    
    start_epoch, best_val_loss, best_score, no_improve_epochs = try_resume_training(
        args, kan, optimizer, scheduler, scaler, device
    )
    resume_path = get_resume_path(args)
    latest_metrics = None
    epoch_metrics_path = os.path.join(args.save_dir, getattr(args, 'epoch_metrics_name', 'epoch_metrics.jsonl'))
    per_timeframe_metrics_path = os.path.join(
        args.save_dir,
        getattr(args, 'per_timeframe_metrics_name', 'per_timeframe_metrics.jsonl'),
    )
    if start_epoch == 0:
        for metric_path in (epoch_metrics_path, per_timeframe_metrics_path):
            if os.path.exists(metric_path):
                os.remove(metric_path)
    
    print("\nStarting Accelerated Sequential Training...")
    
    if start_epoch >= args.epochs:
        print(f"[RESUME] 当前断点 epoch={start_epoch} 已达到目标 epochs={args.epochs}，无需继续训练。")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== EPOCH {epoch+1}/{args.epochs} ==========")
        epoch_train_loss_total = 0.0
        epoch_train_batches = 0
        epoch_train_logs = []
        epoch_all_bucket = build_metric_bucket()
        epoch_timeframe_buckets = defaultdict(build_metric_bucket)
        processed_files = 0
        for file_idx, record in enumerate(final_records):
            data_path = record['path']
            print(f"\n[{file_idx+1}/{len(final_records)}] Processing: {os.path.basename(data_path)}")
            
            try:
                train_ds, eval_ds, _, dataset_meta = create_market_datasets(record, args)
                
                if train_ds is None or eval_ds is None:
                    print("  -> Split generation returned empty train/validation dataset, skipping.")
                    continue

                if len(train_ds) < args.batch_size or len(eval_ds) == 0:
                    print("  -> Dataset too small for this configuration, skipping.")
                    continue

                timeframe_label = normalize_timeframe_label(dataset_meta.get('timeframe') or record.get('timeframe')) or 'unknown'
                train_loader = build_train_loader(train_ds, args, timeframe_label)
                test_loader = build_eval_loader(eval_ds, args)
                
                kan.train()
                total_loss = 0
                loss_logs = []
                
                for batch_idx, (features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags) in enumerate(train_loader):
                    features_seq = features_seq.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    batch_aux = batch_aux.to(device, non_blocking=True)
                    batch_sigma = batch_sigma.to(device, non_blocking=True)
                    batch_weights = batch_weights.to(device, non_blocking=True).unsqueeze(1)
                    batch_flags = batch_flags.to(device, non_blocking=True)
                    
                    psi_t = features_seq[:, -1, :]
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Use bfloat16 if supported for better numerical stability
                    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                        pred, aux_pred, debug_info = forward_model(kan, features_seq, return_debug=use_debug_metrics)
                        
                        if torch.isnan(pred).any():
                            print("NaN in pred! Skipping batch.")
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
                        )
                        reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                        loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN/Inf in loss! Skipping batch.")
                        continue
                        
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping to prevent explosion
                    scaler.unscale_(optimizer)
                    
                    # Check for NaN/Inf in gradients before clipping
                    has_nan_inf = False
                    for param in kan.parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                has_nan_inf = True
                                break
                                
                    if not has_nan_inf:
                        torch.nn.utils.clip_grad_norm_(kan.parameters(), max_norm=getattr(args, 'grad_clip', 1.0))
                        scaler.step(optimizer)
                    else:
                        print("NaN/Inf in gradients! Skipping optimizer step.")
                        
                    scaler.update()
                    
                    total_loss += loss.item()
                    loss_logs.append(l_dict)
                    
                    if batch_idx % 100 == 0:
                        print(
                            f"    [Batch {batch_idx}/{len(train_loader)}] "
                            f"Loss: {loss.item():.6f} | Main: {l_dict['main']:.4f} | Aux: {l_dict['aux']:.4f} | Rank: {l_dict['rank']:.4f}"
                        )
                        
                avg_loss = total_loss / max(len(train_loader), 1)
                
                # Validation
                kan.eval()
                file_bucket = build_metric_bucket()
                with torch.no_grad():
                    for features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags in test_loader:
                        features_seq = features_seq.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        batch_aux = batch_aux.to(device, non_blocking=True)
                        batch_sigma = batch_sigma.to(device, non_blocking=True)
                        batch_weights = batch_weights.to(device, non_blocking=True).unsqueeze(1)
                        batch_flags = batch_flags.to(device, non_blocking=True)
                        
                        psi_t = features_seq[:, -1, :]
                        
                        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                            pred, aux_pred, debug_info = forward_model(kan, features_seq, return_debug=use_debug_metrics)
                            loss_unweighted, rank_loss, l_dict = criterion(
                                pred,
                                aux_pred,
                                batch_y,
                                batch_aux,
                                psi_t,
                                batch_flags,
                                batch_sigma,
                                debug_info=debug_info,
                            )
                            reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                            loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
                            
                        batch_log = dict(l_dict)
                        batch_log['total_loss'] = float(loss.item())
                        file_bucket['logs'].append(batch_log)
                        file_bucket['preds'].append(pred.detach().float().cpu().numpy())
                        file_bucket['targets'].append(batch_y.detach().cpu().numpy())
                        file_bucket['flags'].append(batch_flags.detach().cpu().numpy())
                        if debug_info is not None:
                            for debug_key in DEBUG_BATCH_KEYS:
                                debug_value = debug_info.get(debug_key)
                                if debug_value is None:
                                    continue
                                if isinstance(debug_value, torch.Tensor):
                                    debug_value = debug_value.detach().float().cpu().numpy()
                                else:
                                    debug_value = np.asarray(debug_value, dtype=np.float32)
                                file_bucket['debug_batches'][debug_key].append(debug_value)

                file_summary = summarize_metric_bucket(
                    file_bucket,
                    score_profile=score_profile,
                    use_direction_metrics=use_direction_metrics,
                )
                train_main = float(np.mean([x['main'] for x in loss_logs])) if loss_logs else 0.0
                train_aux = float(np.mean([x['aux'] for x in loss_logs])) if loss_logs else 0.0
                print(
                    f"  -> Train Loss: {avg_loss:.6f} | Val Loss: {file_summary['avg_val_loss']:.6f} | "
                    f"Train Main/Aux: {train_main:.4f}/{train_aux:.4f} | "
                    f"Val Main/Aux: {file_summary['loss_main']:.4f}/{file_summary['loss_aux']:.4f}"
                )
                print(
                    f"  -> Val Corr Breakout/Reversion: {file_summary['breakout_corr']:.4f}/{file_summary['reversion_corr']:.4f} | "
                    f"Event Mean: {file_summary['breakout_event_mean']:.4f}/{file_summary['reversion_event_mean']:.4f} | "
                    f"HardNeg Mean: {file_summary['breakout_hard_negative_mean']:.4f}/{file_summary['reversion_hard_negative_mean']:.4f} | "
                    f"Gap: {file_summary['breakout_gap']:.4f}/{file_summary['reversion_gap']:.4f} | "
                    f"Composite: {file_summary['composite_score']:.4f}"
                )
                print(
                    f"  -> Event F1 Breakout/Reversion: {file_summary['breakout_metrics']['f1']:.4f}/{file_summary['reversion_metrics']['f1']:.4f} | "
                    f"HardNeg: {file_summary['breakout_metrics']['hard_negative_rate']:.4f}/{file_summary['reversion_metrics']['hard_negative_rate']:.4f} | "
                    f"Direction MacroF1: {file_summary['direction_metrics']['macro_f1']:.4f}"
                )
                print(
                    f"  -> Gate Mean Dir/Public/Break: "
                    f"{file_summary['direction_gate_mean']:.4f}/{file_summary['public_reversion_gate_mean']:.4f}/{file_summary['breakout_residual_gate_mean']:.4f} | "
                    f"Floor: {file_summary['directional_floor_mean']:.4f} | "
                    f"Constraint B/P/Pub/Cont: "
                    f"{file_summary['blue_over_purple_violation']:.4f}/"
                    f"{file_summary['purple_over_blue_violation']:.4f}/"
                    f"{file_summary['public_below_directional_violation']:.4f}/"
                    f"{file_summary['continuation_public_violation']:.4f}"
                )
                epoch_train_loss_total += total_loss
                epoch_train_batches += len(train_loader)
                epoch_train_logs.extend(loss_logs)
                merge_metric_bucket(epoch_all_bucket, file_bucket)
                merge_metric_bucket(epoch_timeframe_buckets[timeframe_label], file_bucket)
                processed_files += 1
                    
                # Cleanup to avoid memory leaks
                del train_ds, eval_ds, train_loader, test_loader
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing {data_path}: {e}")
                continue

        timeframe_summaries = {
            timeframe: summarize_metric_bucket(
                bucket,
                score_profile=score_profile,
                use_direction_metrics=use_direction_metrics,
            )
            for timeframe, bucket in sorted(epoch_timeframe_buckets.items())
        }
        overall_all_summary = summarize_metric_bucket(
            epoch_all_bucket,
            score_profile=score_profile,
            use_direction_metrics=use_direction_metrics,
        )
        if score_timeframe_set:
            score_bucket = build_metric_bucket()
            for timeframe, bucket in epoch_timeframe_buckets.items():
                if timeframe in score_timeframe_set:
                    merge_metric_bucket(score_bucket, bucket)
            if not score_bucket['preds']:
                score_bucket = epoch_all_bucket
        else:
            score_bucket = epoch_all_bucket
        epoch_score_summary = summarize_metric_bucket(
            score_bucket,
            score_profile=score_profile,
            use_direction_metrics=use_direction_metrics,
        )
        epoch_val_batches = len(score_bucket['logs'])
        epoch_val_loss_total = float(sum(item.get('total_loss', 0.0) for item in score_bucket['logs']))
        epoch_val_logs = list(score_bucket['logs'])
        epoch_val_preds = list(score_bucket['preds'])
        epoch_val_targets = list(score_bucket['targets'])
        epoch_val_flags = list(score_bucket['flags'])
        epoch_val_blue = list(score_bucket['debug_batches']['blue_score'])
        epoch_val_purple = list(score_bucket['debug_batches']['purple_score'])

        if processed_files == 0 or epoch_val_batches == 0:
            print("No valid files processed in this epoch.")
            continue

        epoch_avg_loss = epoch_train_loss_total / max(epoch_train_batches, 1)
        epoch_avg_val_loss = epoch_val_loss_total / max(epoch_val_batches, 1)
        epoch_train_main = float(np.mean([x['main'] for x in epoch_train_logs])) if epoch_train_logs else 0.0
        epoch_train_aux = float(np.mean([x['aux'] for x in epoch_train_logs])) if epoch_train_logs else 0.0
        epoch_val_main = float(np.mean([x['main'] for x in epoch_val_logs])) if epoch_val_logs else 0.0
        epoch_val_aux = float(np.mean([x['aux'] for x in epoch_val_logs])) if epoch_val_logs else 0.0
        pred_np = np.vstack(epoch_val_preds)
        target_np = np.vstack(epoch_val_targets)
        flags_np = np.vstack(epoch_val_flags)
        pred_rev_np = np.maximum(pred_np[:, 1], 0.0)
        breakout_corr = safe_corr(pred_np[:, 0], target_np[:, 0])
        reversion_corr = safe_corr(pred_rev_np, target_np[:, 1])
        breakout_event_mean = float(pred_np[flags_np[:, 0] > 0.5, 0].mean()) if np.any(flags_np[:, 0] > 0.5) else 0.0
        reversion_event_mean = float(pred_rev_np[flags_np[:, 1] > 0.5].mean()) if np.any(flags_np[:, 1] > 0.5) else 0.0
        breakout_hn_mean = float(pred_np[flags_np[:, 2] > 0.5, 0].mean()) if np.any(flags_np[:, 2] > 0.5) else 0.0
        reversion_hn_mean = float(pred_rev_np[flags_np[:, 3] > 0.5].mean()) if np.any(flags_np[:, 3] > 0.5) else 0.0
        breakout_metrics = compute_event_metrics(pred_np[:, 0], flags_np[:, 0] > 0.5, flags_np[:, 2] > 0.5)
        reversion_metrics = compute_event_metrics(pred_rev_np, flags_np[:, 1] > 0.5, flags_np[:, 3] > 0.5)
        direction_metrics = compute_direction_metrics(
            np.vstack(epoch_val_blue).reshape(-1),
            np.vstack(epoch_val_purple).reshape(-1),
            flags_np,
        ) if epoch_val_blue and epoch_val_purple else {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'blue_f1': 0.0,
            'purple_f1': 0.0,
            'support': 0,
        }
        breakout_gap = breakout_event_mean - breakout_hn_mean
        reversion_gap = reversion_event_mean - reversion_hn_mean
        composite_score = compute_checkpoint_score(
            breakout_metrics,
            reversion_metrics,
            breakout_corr,
            reversion_corr,
            direction_metrics['macro_f1'] if use_direction_metrics else None,
            profile=score_profile,
        )
        print(
            f"\n[EPOCH {epoch+1} SUMMARY] Train Loss: {epoch_avg_loss:.6f} | Val Loss: {epoch_avg_val_loss:.6f} | "
            f"Train Main/Aux: {epoch_train_main:.4f}/{epoch_train_aux:.4f} | "
            f"Val Main/Aux: {epoch_val_main:.4f}/{epoch_val_aux:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Val Corr Breakout/Reversion: {breakout_corr:.4f}/{reversion_corr:.4f} | "
            f"Event Mean: {breakout_event_mean:.4f}/{reversion_event_mean:.4f} | "
            f"HardNeg Mean: {breakout_hn_mean:.4f}/{reversion_hn_mean:.4f} | "
            f"Gap: {breakout_gap:.4f}/{reversion_gap:.4f} | "
            f"Composite: {composite_score:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Breakout Acc/P/R/F1: "
            f"{breakout_metrics['accuracy']:.4f}/{breakout_metrics['precision']:.4f}/"
            f"{breakout_metrics['recall']:.4f}/{breakout_metrics['f1']:.4f} | "
            f"阈值: {breakout_metrics['threshold']:.4f} | "
            f"事件命中率/伪信号率: {breakout_metrics['event_rate']:.4f}/{breakout_metrics['hard_negative_rate']:.4f} | "
            f"信号频次/标签频次: {breakout_metrics['signal_frequency']:.4f}/{breakout_metrics['label_frequency']:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Reversion Acc/P/R/F1: "
            f"{reversion_metrics['accuracy']:.4f}/{reversion_metrics['precision']:.4f}/"
            f"{reversion_metrics['recall']:.4f}/{reversion_metrics['f1']:.4f} | "
            f"阈值: {reversion_metrics['threshold']:.4f} | "
            f"事件命中率/伪信号率: {reversion_metrics['event_rate']:.4f}/{reversion_metrics['hard_negative_rate']:.4f} | "
            f"信号频次/标签频次: {reversion_metrics['signal_frequency']:.4f}/{reversion_metrics['label_frequency']:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Direction Acc/MacroF1/BlueF1/PurpleF1: "
            f"{direction_metrics['accuracy']:.4f}/{direction_metrics['macro_f1']:.4f}/"
            f"{direction_metrics['blue_f1']:.4f}/{direction_metrics['purple_f1']:.4f} | "
            f"Support: {direction_metrics['support']}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Gate Mean Dir/Public/Break: "
            f"{epoch_score_summary['direction_gate_mean']:.4f}/"
            f"{epoch_score_summary['public_reversion_gate_mean']:.4f}/"
            f"{epoch_score_summary['breakout_residual_gate_mean']:.4f} | "
            f"Floor: {epoch_score_summary['directional_floor_mean']:.4f} | "
            f"Constraint B/P/Pub/Cont: "
            f"{epoch_score_summary['blue_over_purple_violation']:.4f}/"
            f"{epoch_score_summary['purple_over_blue_violation']:.4f}/"
            f"{epoch_score_summary['public_below_directional_violation']:.4f}/"
            f"{epoch_score_summary['continuation_public_violation']:.4f}"
        )
        if score_timeframe_set:
            print(
                f"[EPOCH {epoch+1} SUMMARY] Score Timeframes: {sorted(score_timeframe_set)} | "
                f"All-Timeframe Composite: {overall_all_summary['composite_score']:.4f}"
            )
        for timeframe, summary in timeframe_summaries.items():
            score_included = timeframe in score_timeframe_set if score_timeframe_set else True
            print(
                f"[EPOCH {epoch+1}][{timeframe}] Composite: {summary['composite_score']:.4f} | "
                f"Breakout/Reversion F1: {summary['breakout_metrics']['f1']:.4f}/{summary['reversion_metrics']['f1']:.4f} | "
                f"DirectionF1: {summary['direction_metrics']['macro_f1']:.4f} | "
                f"ScoreIncluded: {score_included}"
            )
            append_jsonl(
                per_timeframe_metrics_path,
                {
                    'epoch': epoch + 1,
                    'timeframe': timeframe,
                    'score_included': score_included,
                    'aux_timeframe': timeframe in aux_timeframe_set,
                    'constraint_profile': constraint_profile,
                    'arch_version': getattr(args, 'arch_version', 'iterA2_base'),
                    'dataset_profile': getattr(args, 'dataset_profile', 'iterA2'),
                    'loss_profile': getattr(args, 'loss_profile', 'default'),
                    'score_profile': score_profile,
                    'sample_count': summary['sample_count'],
                    'breakout_f1': summary['breakout_metrics']['f1'],
                    'reversion_f1': summary['reversion_metrics']['f1'],
                    'breakout_hard_negative_rate': summary['breakout_metrics']['hard_negative_rate'],
                    'reversion_hard_negative_rate': summary['reversion_metrics']['hard_negative_rate'],
                    'breakout_gap': summary['breakout_gap'],
                    'reversion_gap': summary['reversion_gap'],
                    'composite_score': summary['composite_score'],
                    'direction_macro_f1': summary['direction_metrics']['macro_f1'],
                    'signal_frequency': summary['signal_frequency'],
                    'label_frequency': summary['label_frequency'],
                    'breakout_signal_frequency': summary['breakout_signal_frequency'],
                    'reversion_signal_frequency': summary['reversion_signal_frequency'],
                    'breakout_label_frequency': summary['breakout_label_frequency'],
                    'reversion_label_frequency': summary['reversion_label_frequency'],
                    'direction_gate_mean': summary['direction_gate_mean'],
                    'direction_gate_std': summary['direction_gate_std'],
                    'public_reversion_gate_mean': summary['public_reversion_gate_mean'],
                    'public_reversion_gate_std': summary['public_reversion_gate_std'],
                    'breakout_residual_gate_mean': summary['breakout_residual_gate_mean'],
                    'breakout_residual_gate_std': summary['breakout_residual_gate_std'],
                    'directional_floor_mean': summary['directional_floor_mean'],
                    'blue_over_purple_violation': summary['blue_over_purple_violation'],
                    'blue_over_purple_violation_rate': summary['blue_over_purple_violation_rate'],
                    'purple_over_blue_violation': summary['purple_over_blue_violation'],
                    'purple_over_blue_violation_rate': summary['purple_over_blue_violation_rate'],
                    'public_below_directional_violation': summary['public_below_directional_violation'],
                    'public_below_directional_violation_rate': summary['public_below_directional_violation_rate'],
                    'continuation_public_violation': summary['continuation_public_violation'],
                    'continuation_public_violation_rate': summary['continuation_public_violation_rate'],
                },
            )
        append_jsonl(
            epoch_metrics_path,
            {
                'epoch': epoch + 1,
                'processed_files': processed_files,
                'constraint_profile': constraint_profile,
                'arch_version': getattr(args, 'arch_version', 'iterA2_base'),
                'dataset_profile': getattr(args, 'dataset_profile', 'iterA2'),
                'loss_profile': getattr(args, 'loss_profile', 'default'),
                'score_profile': score_profile,
                'score_timeframes': score_timeframes or sorted(timeframe_summaries.keys()),
                'aux_timeframes': aux_timeframes,
                'train_loss': epoch_avg_loss,
                'val_loss': epoch_avg_val_loss,
                'train_main': epoch_train_main,
                'train_aux': epoch_train_aux,
                'val_main': epoch_val_main,
                'val_aux': epoch_val_aux,
                'sample_count': epoch_score_summary['sample_count'],
                'breakout_corr': epoch_score_summary['breakout_corr'],
                'reversion_corr': epoch_score_summary['reversion_corr'],
                'breakout_gap': epoch_score_summary['breakout_gap'],
                'reversion_gap': epoch_score_summary['reversion_gap'],
                'composite_score': composite_score,
                'direction_macro_f1': epoch_score_summary['direction_metrics']['macro_f1'],
                'breakout_f1': epoch_score_summary['breakout_metrics']['f1'],
                'reversion_f1': epoch_score_summary['reversion_metrics']['f1'],
                'breakout_hard_negative_rate': epoch_score_summary['breakout_metrics']['hard_negative_rate'],
                'reversion_hard_negative_rate': epoch_score_summary['reversion_metrics']['hard_negative_rate'],
                'signal_frequency': epoch_score_summary['signal_frequency'],
                'label_frequency': epoch_score_summary['label_frequency'],
                'direction_gate_mean': epoch_score_summary['direction_gate_mean'],
                'direction_gate_std': epoch_score_summary['direction_gate_std'],
                'public_reversion_gate_mean': epoch_score_summary['public_reversion_gate_mean'],
                'public_reversion_gate_std': epoch_score_summary['public_reversion_gate_std'],
                'breakout_residual_gate_mean': epoch_score_summary['breakout_residual_gate_mean'],
                'breakout_residual_gate_std': epoch_score_summary['breakout_residual_gate_std'],
                'directional_floor_mean': epoch_score_summary['directional_floor_mean'],
                'blue_over_purple_violation': epoch_score_summary['blue_over_purple_violation'],
                'blue_over_purple_violation_rate': epoch_score_summary['blue_over_purple_violation_rate'],
                'purple_over_blue_violation': epoch_score_summary['purple_over_blue_violation'],
                'purple_over_blue_violation_rate': epoch_score_summary['purple_over_blue_violation_rate'],
                'public_below_directional_violation': epoch_score_summary['public_below_directional_violation'],
                'public_below_directional_violation_rate': epoch_score_summary['public_below_directional_violation_rate'],
                'continuation_public_violation': epoch_score_summary['continuation_public_violation'],
                'continuation_public_violation_rate': epoch_score_summary['continuation_public_violation_rate'],
                'all_timeframes_composite_score': overall_all_summary['composite_score'],
            },
        )

        scheduler.step(epoch_avg_val_loss)

        improved = (
            composite_score > best_score + args.early_stop_min_delta or
            (
                abs(composite_score - best_score) <= args.early_stop_min_delta and
                epoch_avg_val_loss < best_val_loss - args.early_stop_min_delta
            )
        )

        if improved:
            best_score = composite_score
            best_val_loss = epoch_avg_val_loss
            no_improve_epochs = 0
            save_path = os.path.join(args.save_dir, getattr(args, 'best_name', 'khaos_kan_best.pth'))
            torch.save({
                'model_state_dict': kan.state_dict(),
                'args': vars(args),
                'dataset_manifest': final_records,
                'val_loss': best_val_loss,
                'best_score': best_score,
                'metrics': {
                    'breakout_corr': breakout_corr,
                    'reversion_corr': reversion_corr,
                    'direction_macro_f1': direction_metrics['macro_f1'],
                    'breakout_event_mean': breakout_event_mean,
                    'reversion_event_mean': reversion_event_mean,
                    'breakout_hard_negative_mean': breakout_hn_mean,
                    'reversion_hard_negative_mean': reversion_hn_mean,
                    'breakout_gap': breakout_gap,
                    'reversion_gap': reversion_gap,
                    'composite_score': composite_score,
                    'processed_files': processed_files,
                    'epoch': epoch + 1,
                    'breakout_eval': breakout_metrics,
                    'reversion_eval': reversion_metrics,
                    'direction_eval': epoch_score_summary['direction_metrics'],
                    'gate_stats': {
                        'direction_gate_mean': epoch_score_summary['direction_gate_mean'],
                        'direction_gate_std': epoch_score_summary['direction_gate_std'],
                        'public_reversion_gate_mean': epoch_score_summary['public_reversion_gate_mean'],
                        'public_reversion_gate_std': epoch_score_summary['public_reversion_gate_std'],
                        'breakout_residual_gate_mean': epoch_score_summary['breakout_residual_gate_mean'],
                        'breakout_residual_gate_std': epoch_score_summary['breakout_residual_gate_std'],
                        'directional_floor_mean': epoch_score_summary['directional_floor_mean'],
                    },
                    'constraint_stats': {
                        key: epoch_score_summary[key]
                        for key in CONSTRAINT_STAT_BASE_KEYS
                    },
                    'constraint_rates': {
                        f'{key}_rate': epoch_score_summary[f'{key}_rate']
                        for key in CONSTRAINT_STAT_BASE_KEYS
                    },
                    'score_timeframes': score_timeframes or sorted(timeframe_summaries.keys()),
                    'aux_timeframes': aux_timeframes,
                    'per_timeframe_metrics': {
                        timeframe: {
                            'composite_score': summary['composite_score'],
                            'breakout_f1': summary['breakout_metrics']['f1'],
                            'reversion_f1': summary['reversion_metrics']['f1'],
                            'direction_macro_f1': summary['direction_metrics']['macro_f1'],
                        }
                        for timeframe, summary in timeframe_summaries.items()
                    },
                    'all_timeframes_composite_score': overall_all_summary['composite_score'],
                },
                'feature_names': PHYSICS_FEATURE_NAMES,
                'env': {
                    'torch': torch.__version__,
                    'cuda': torch.version.cuda if torch.cuda.is_available() else None,
                    'device': str(device)
                }
            }, save_path)
        else:
            no_improve_epochs += 1
            print(
                f"[EPOCH {epoch+1} SUMMARY] 未超过最优，连续未改进轮数: "
                f"{no_improve_epochs}/{args.early_stop_patience}"
            )

        latest_metrics = {
            'breakout_corr': breakout_corr,
            'reversion_corr': reversion_corr,
            'direction_macro_f1': direction_metrics['macro_f1'],
            'breakout_gap': breakout_gap,
            'reversion_gap': reversion_gap,
            'composite_score': composite_score,
            'val_loss': epoch_avg_val_loss,
            'processed_files': processed_files,
            'epoch': epoch + 1,
            'breakout_eval': breakout_metrics,
            'reversion_eval': reversion_metrics,
            'direction_eval': epoch_score_summary['direction_metrics'],
            'gate_stats': {
                'direction_gate_mean': epoch_score_summary['direction_gate_mean'],
                'direction_gate_std': epoch_score_summary['direction_gate_std'],
                'public_reversion_gate_mean': epoch_score_summary['public_reversion_gate_mean'],
                'public_reversion_gate_std': epoch_score_summary['public_reversion_gate_std'],
                'breakout_residual_gate_mean': epoch_score_summary['breakout_residual_gate_mean'],
                'breakout_residual_gate_std': epoch_score_summary['breakout_residual_gate_std'],
                'directional_floor_mean': epoch_score_summary['directional_floor_mean'],
            },
            'constraint_stats': {
                key: epoch_score_summary[key]
                for key in CONSTRAINT_STAT_BASE_KEYS
            },
            'constraint_rates': {
                f'{key}_rate': epoch_score_summary[f'{key}_rate']
                for key in CONSTRAINT_STAT_BASE_KEYS
            },
            'score_timeframes': score_timeframes or sorted(timeframe_summaries.keys()),
            'aux_timeframes': aux_timeframes,
            'per_timeframe_metrics': {
                timeframe: {
                    'composite_score': summary['composite_score'],
                    'breakout_f1': summary['breakout_metrics']['f1'],
                    'reversion_f1': summary['reversion_metrics']['f1'],
                    'direction_macro_f1': summary['direction_metrics']['macro_f1'],
                }
                for timeframe, summary in timeframe_summaries.items()
            },
            'all_timeframes_composite_score': overall_all_summary['composite_score'],
        }
        save_resume_checkpoint(
            resume_path=resume_path,
            kan=kan,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            best_score=best_score,
            no_improve_epochs=no_improve_epochs,
            latest_metrics=latest_metrics,
            device=device,
            completed=False
        )

        if no_improve_epochs >= args.early_stop_patience:
            print(
                f"[EARLY STOP] 连续 {args.early_stop_patience} 个 epoch 未达到最小改进 "
                f"{args.early_stop_min_delta:.4f}，提前停止。"
            )
            break

    # Save Final Model
    save_path = os.path.join(args.save_dir, getattr(args, 'final_name', 'khaos_kan_model_final.pth'))
    torch.save({
        'model_state_dict': kan.state_dict(),
        'args': vars(args),
        'dataset_manifest': final_records,
        'feature_names': PHYSICS_FEATURE_NAMES,
        'latest_metrics': latest_metrics,
        'env': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else None,
            'device': str(device)
        }
    }, save_path)
    save_resume_checkpoint(
        resume_path=resume_path,
        kan=kan,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        epoch=args.epochs,
        best_val_loss=best_val_loss,
        best_score=best_score,
        no_improve_epochs=no_improve_epochs,
        latest_metrics=latest_metrics,
        device=device,
        completed=True
    )
    print(f"\nTraining Complete. Final model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed')
    parser.add_argument('--save_dir', type=str, default=r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份')
    parser.add_argument('--market', type=str, default='legacy_multiasset')
    parser.add_argument('--training_subdir', type=str, default=None)
    parser.add_argument('--assets', type=str, default=None)
    parser.add_argument('--timeframes', type=str, default=None)
    parser.add_argument('--split_mode', type=str, default='ratio')
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--val_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--per_timeframe_train_cap', type=str, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--best_name', type=str, default='khaos_kan_best.pth')
    parser.add_argument('--final_name', type=str, default='khaos_kan_model_final.pth')
    parser.add_argument('--resume_name', type=str, default='khaos_kan_resume.pth')
    parser.add_argument('--epochs', type=int, default=3) 
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=4)
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
    parser.add_argument('--epoch_metrics_name', type=str, default='epoch_metrics.jsonl')
    parser.add_argument('--per_timeframe_metrics_name', type=str, default='per_timeframe_metrics.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--test_mode', action='store_true', default=False)
    parser.add_argument('--fast_full', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=2)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.002)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', type=str, default=None)
    
    args = parser.parse_args()
    train(args)
