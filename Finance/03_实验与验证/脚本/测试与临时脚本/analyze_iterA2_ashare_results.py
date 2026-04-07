import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_ROOT = r'D:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》'
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'Finance', '02_核心代码', '源代码')
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

from khaos.数据处理.ashare_dataset import EVENT_FLAG_INDEX, create_ashare_dataset_splits
from khaos.数据处理.ashare_support import (
    ASHARE_FALLBACK_ASSETS,
    ASHARE_PRIMARY_ASSETS,
    DEFAULT_ASHARE_TIMEFRAMES,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    build_market_coverage_report,
    discover_training_files,
    normalize_timeframe_label,
)
from khaos.模型定义.kan import KHAOS_KAN
from khaos.模型训练.train import compute_checkpoint_score, compute_direction_metrics
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES
from khaos.同花顺公式.ths_core_proxy import (
    DEFAULT_THS_CORE_PARAMS,
    TUNABLE_THS_FIELDS,
    compute_ths_core_frame,
)


DATA_DIR = os.path.join(PROJECT_ROOT, 'Finance', '01_数据中心', '03_研究数据', 'research_processed')
WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, 'Finance', '02_核心代码', '模型权重备份')
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')
FORMULA_PATH = os.path.join(PROJECT_ROOT, 'Finance', '02_核心代码', '源代码', 'khaos', '同花顺公式', 'KHAOS_THS_CORE.txt')

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

DATA_DIR = os.path.join(str(PROJECT_ROOT), 'Finance', '01_数据中心', '03_研究数据', 'research_processed')
WEIGHTS_ROOT = os.path.join(str(PROJECT_ROOT), 'Finance', '02_核心代码', '模型权重备份')
LOG_ROOT = os.path.join(str(PROJECT_ROOT), 'logs')
FORMULA_PATH = os.path.join(str(PROJECT_ROOT), 'Finance', '02_核心代码', '源代码', 'khaos', '同花顺公式', 'KHAOS_THS_CORE.txt')

PHASE_LABELS = [1, 2, -2, 0]
PHASE_NAMES = {
    1: 'breakout',
    2: 'blue_reversion',
    -2: 'purple_reversion',
    0: 'neutral',
}


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
            'label_frequency': 0.0,
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
        candidate = {
            'threshold': float(threshold),
            'accuracy': float((tp + tn) / max(len(scores), 1)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'event_rate': float(np.mean(pred[event_flags])) if np.any(event_flags) else 0.0,
            'hard_negative_rate': float(np.mean(pred[hard_negative_flags])) if np.any(hard_negative_flags) else 0.0,
            'signal_frequency': float(np.mean(pred)),
            'label_frequency': label_frequency,
        }
        if best is None or candidate['f1'] > best['f1'] or (
            candidate['f1'] == best['f1'] and candidate['hard_negative_rate'] < best['hard_negative_rate']
        ):
            best = candidate
    return best


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def resolve_ths_metric_source(result, source_name='default'):
    if source_name == 'calibrated':
        return result['calibrated_ths']
    return result['default_ths']


def build_model(checkpoint, device):
    args = checkpoint.get('args', {})
    model = KHAOS_KAN(
        input_dim=len(checkpoint.get('feature_names', PHYSICS_FEATURE_NAMES)),
        hidden_dim=args.get('hidden_dim', 64),
        output_dim=2,
        layers=args.get('layers', 3),
        grid_size=args.get('grid_size', 10),
        num_heads=4,
        arch_version=args.get('arch_version', 'iterA2_base'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, args


def dataset_to_frame(dataset):
    return pd.DataFrame({
        'time': pd.to_datetime(dataset.time.values),
        'open': dataset.open,
        'high': dataset.high,
        'low': dataset.low,
        'close': dataset.close,
        'volume': dataset.volume,
    })


def evaluate_dataset_split(model, dataset, device, arch_version='iterA2_base'):
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    preds = []
    targets = []
    flags = []
    features = []
    blue_scores = []
    purple_scores = []
    use_debug = arch_version in {'iterA3_multiscale', 'iterA4_multiscale', 'iterA5_multiscale'}
    with torch.no_grad():
        for batch_x, batch_y, _, _, _, batch_flags in loader:
            batch_x = batch_x.to(device)
            if use_debug:
                pred, debug_info = model(batch_x, return_debug=True)
                blue_scores.append(torch.relu(debug_info['blue_score']).detach().cpu().numpy())
                purple_scores.append(torch.relu(debug_info['purple_score']).detach().cpu().numpy())
            else:
                pred = model(batch_x)
            preds.append(pred.detach().cpu().numpy())
            targets.append(batch_y.numpy())
            flags.append(batch_flags.numpy())
            features.append(batch_x[:, -1, :].detach().cpu().numpy())
    if not preds:
        return None
    result = {
        'frame': dataset_to_frame(dataset),
        'sample_bar_indices': np.asarray(getattr(dataset, 'sample_bar_indices', []), dtype=np.int64),
        'predictions': np.vstack(preds),
        'targets': np.vstack(targets),
        'flags': np.vstack(flags),
        'features': np.vstack(features),
    }
    if blue_scores and purple_scores:
        result['blue_scores'] = np.vstack(blue_scores).reshape(-1)
        result['purple_scores'] = np.vstack(purple_scores).reshape(-1)
    return result


def resolve_direction_masks(flags, features):
    flags = np.asarray(flags)
    features = np.asarray(features)
    down_idx = EVENT_FLAG_INDEX['reversion_down_context']
    up_idx = EVENT_FLAG_INDEX['reversion_up_context']
    down_mask = flags[:, down_idx] > 0.5 if flags.shape[1] > down_idx else features[:, 7] > 0.0
    up_mask = flags[:, up_idx] > 0.5 if flags.shape[1] > up_idx else features[:, 7] <= 0.0
    if flags.shape[1] > up_idx:
        unresolved = ~(down_mask | up_mask)
        fallback = (features[:, 7] + features[:, 3]) > 0.0
        down_mask = down_mask | (unresolved & fallback)
        up_mask = up_mask | (unresolved & ~fallback)
    return down_mask.astype(bool), up_mask.astype(bool)


def build_label_phase(targets, flags, features):
    targets = np.asarray(targets)
    flags = np.asarray(flags)
    features = np.asarray(features)
    breakout_mask = flags[:, EVENT_FLAG_INDEX['breakout_event']] > 0.5
    reversion_mask = flags[:, EVENT_FLAG_INDEX['reversion_event']] > 0.5
    down_mask, up_mask = resolve_direction_masks(flags, features)
    direction_score = features[:, 7] + features[:, 3]
    phase = np.zeros(len(flags), dtype=np.int32)
    blue_sel = reversion_mask & (down_mask | ((~down_mask & ~up_mask) & (direction_score > 0.0)))
    purple_sel = reversion_mask & ~blue_sel
    phase[blue_sel] = 2
    phase[purple_sel] = -2
    breakout_priority = breakout_mask & ((targets[:, 0] >= targets[:, 1]) | (phase == 0))
    phase[breakout_priority] = 1
    return phase


def build_teacher_phase(predictions, flags, features, breakout_threshold, reversion_threshold, dir_gap, blue_scores=None, purple_scores=None):
    predictions = np.asarray(predictions)
    features = np.asarray(features)
    breakout_scores = predictions[:, 0]
    if blue_scores is not None and purple_scores is not None:
        blue_dom = np.asarray(blue_scores, dtype=np.float64).reshape(-1)
        purple_dom = np.asarray(purple_scores, dtype=np.float64).reshape(-1)
    else:
        reversion_scores = np.maximum(predictions[:, 1], 0.0)
        down_mask, up_mask = resolve_direction_masks(flags, features)
        direction_score = features[:, 7] + features[:, 3]
        blue_dom = np.where(down_mask, reversion_scores, 0.0)
        purple_dom = np.where(up_mask, reversion_scores, 0.0)
        dual_mask = down_mask & up_mask
        blue_dom = np.where(dual_mask & (direction_score <= 0.0), 0.0, blue_dom)
        purple_dom = np.where(dual_mask & (direction_score > 0.0), 0.0, purple_dom)
    bk_on = (breakout_scores >= breakout_threshold) & (breakout_scores >= blue_dom) & (breakout_scores >= purple_dom)
    blue_on = (blue_dom >= reversion_threshold) & ((blue_dom - purple_dom) >= dir_gap) & (blue_dom > breakout_scores)
    purple_on = (purple_dom >= reversion_threshold) & ((purple_dom - blue_dom) >= dir_gap) & (purple_dom > breakout_scores)
    return np.where(bk_on, 1, np.where(blue_on, 2, np.where(purple_on, -2, 0))).astype(np.int32)


def compute_phase_metrics(truth, pred):
    truth = np.asarray(truth, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        pred,
        labels=PHASE_LABELS,
        zero_division=0,
    )
    metrics = {
        'accuracy': float(np.mean(truth == pred)) if len(truth) else 0.0,
        'macro_f1': float(np.mean(f1)) if len(f1) else 0.0,
        'phase_frequency': {
            PHASE_NAMES[label]: float(np.mean(pred == label)) if len(pred) else 0.0
            for label in PHASE_LABELS
        },
        'per_phase': {},
        'confusion_matrix': {},
    }
    for idx, label in enumerate(PHASE_LABELS):
        metrics['per_phase'][PHASE_NAMES[label]] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx]),
        }
    matrix = confusion_matrix(truth, pred, labels=PHASE_LABELS)
    for row_idx, row_label in enumerate(PHASE_LABELS):
        metrics['confusion_matrix'][PHASE_NAMES[row_label]] = {
            PHASE_NAMES[col_label]: int(matrix[row_idx, col_idx])
            for col_idx, col_label in enumerate(PHASE_LABELS)
        }
    return metrics


def aggregate_model_records(records, score_profile='default'):
    pred_np = np.vstack([item['predictions'] for item in records])
    target_np = np.vstack([item['targets'] for item in records])
    flags_np = np.vstack([item['flags'] for item in records])
    breakout_scores = pred_np[:, 0]
    reversion_scores = np.maximum(pred_np[:, 1], 0.0)
    breakout_eval = compute_event_metrics(
        breakout_scores,
        flags_np[:, EVENT_FLAG_INDEX['breakout_event']] > 0.5,
        flags_np[:, EVENT_FLAG_INDEX['breakout_hard_negative']] > 0.5,
    )
    reversion_eval = compute_event_metrics(
        reversion_scores,
        flags_np[:, EVENT_FLAG_INDEX['reversion_event']] > 0.5,
        flags_np[:, EVENT_FLAG_INDEX['reversion_hard_negative']] > 0.5,
    )
    breakout_corr = safe_corr(breakout_scores, target_np[:, 0])
    reversion_corr = safe_corr(reversion_scores, target_np[:, 1])
    direction_eval = {
        'accuracy': 0.0,
        'macro_f1': 0.0,
        'blue_f1': 0.0,
        'purple_f1': 0.0,
        'support': 0,
    }
    if any('blue_scores' in item for item in records):
        blue_np = np.concatenate([item['blue_scores'] for item in records if 'blue_scores' in item])
        purple_np = np.concatenate([item['purple_scores'] for item in records if 'purple_scores' in item])
        direction_eval = compute_direction_metrics(blue_np, purple_np, flags_np)
    return {
        'breakout_corr': breakout_corr,
        'reversion_corr': reversion_corr,
        'breakout_eval': breakout_eval,
        'reversion_eval': reversion_eval,
        'direction_eval': direction_eval,
        'composite_score': compute_checkpoint_score(
            breakout_eval,
            reversion_eval,
            breakout_corr,
            reversion_corr,
            direction_eval['macro_f1'] if direction_eval['support'] > 0 else None,
            profile=score_profile,
        ),
    }


def summarize_group(records, key_name, score_profile='default'):
    grouped = {}
    keys = sorted({item[key_name] for item in records})
    for key in keys:
        grouped[key] = aggregate_model_records(
            [item for item in records if item[key_name] == key],
            score_profile=score_profile,
        )
    return grouped


def kernel_rules(kernel_scores, features, feature_names):
    threshold = np.percentile(kernel_scores, 90)
    labels = (kernel_scores >= threshold).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree.fit(features, labels)
    return float(threshold), export_text(tree, feature_names=feature_names)


def attach_phase_views(record, ths_params, breakout_threshold, reversion_threshold, dir_gap):
    label_phase = build_label_phase(record['targets'], record['flags'], record['features'])
    teacher_phase = build_teacher_phase(
        record['predictions'],
        record['flags'],
        record['features'],
        breakout_threshold,
        reversion_threshold,
        dir_gap,
        blue_scores=record.get('blue_scores'),
        purple_scores=record.get('purple_scores'),
    )
    proxy_frame = compute_ths_core_frame(record['frame'], params=ths_params, normalize_input=False)
    proxy_phase = proxy_frame['PHASE'].to_numpy()[record['sample_bar_indices']]
    enriched = dict(record)
    enriched['label_phase'] = label_phase
    enriched['teacher_phase'] = teacher_phase
    enriched['proxy_phase'] = proxy_phase
    return enriched


def evaluate_ths_records(records, params, breakout_threshold, reversion_threshold, dir_gap):
    enriched = [
        attach_phase_views(record, params, breakout_threshold, reversion_threshold, dir_gap)
        for record in records
    ]
    label_truth = np.concatenate([item['label_phase'] for item in enriched])
    teacher_truth = np.concatenate([item['teacher_phase'] for item in enriched])
    proxy_phase = np.concatenate([item['proxy_phase'] for item in enriched])
    label_metrics = compute_phase_metrics(label_truth, proxy_phase)
    teacher_metrics = compute_phase_metrics(teacher_truth, proxy_phase)
    objective = 0.60 * label_metrics['macro_f1'] + 0.40 * teacher_metrics['macro_f1']
    return objective, {
        'objective': float(objective),
        'label_metrics': label_metrics,
        'teacher_metrics': teacher_metrics,
    }, enriched


def build_calibration_grid(params):
    return {
        'bk_evt_th': [params.bk_evt_th - 0.20, params.bk_evt_th - 0.10, params.bk_evt_th, params.bk_evt_th + 0.10, params.bk_evt_th + 0.20],
        'rv_evt_th': [max(params.rv_evt_th - 0.18, 0.40), max(params.rv_evt_th - 0.09, 0.40), params.rv_evt_th, params.rv_evt_th + 0.09, params.rv_evt_th + 0.18],
        'res_node': [-0.015, -0.0125, params.res_node, -0.0075, -0.005],
        'ent_bull': [0.50, 0.54, 0.58],
        'ent_bear': [0.56, 0.60, 0.64],
        'mle_gate': [0.05, 0.08, 0.11],
        'dir_gap': [0.10, 0.14, params.dir_gap, 0.22, 0.26],
        'bk_comp_weight': [params.bk_comp_weight * 0.85, params.bk_comp_weight, params.bk_comp_weight * 1.15],
        'bk_mle_weight': [params.bk_mle_weight * 0.85, params.bk_mle_weight, params.bk_mle_weight * 1.15],
        'bk_press_weight': [params.bk_press_weight * 0.85, params.bk_press_weight, params.bk_press_weight * 1.15],
        'bk_ent_turn_weight': [params.bk_ent_turn_weight * 0.85, params.bk_ent_turn_weight, params.bk_ent_turn_weight * 1.15],
        'bk_trend_weight': [params.bk_trend_weight * 0.85, params.bk_trend_weight, params.bk_trend_weight * 1.15],
        'dnrev_rev_setup_weight': [params.dnrev_rev_setup_weight * 0.85, params.dnrev_rev_setup_weight, params.dnrev_rev_setup_weight * 1.15],
        'dnrev_ent_weight': [params.dnrev_ent_weight * 0.85, params.dnrev_ent_weight, params.dnrev_ent_weight * 1.15],
        'dnrev_mom_weight': [params.dnrev_mom_weight * 0.85, params.dnrev_mom_weight, params.dnrev_mom_weight * 1.15],
        'uprev_rev_setup_weight': [params.uprev_rev_setup_weight * 0.85, params.uprev_rev_setup_weight, params.uprev_rev_setup_weight * 1.15],
        'uprev_ent_weight': [params.uprev_ent_weight * 0.85, params.uprev_ent_weight, params.uprev_ent_weight * 1.15],
        'uprev_ent_rise_weight': [params.uprev_ent_rise_weight * 0.85, params.uprev_ent_rise_weight, params.uprev_ent_rise_weight * 1.15],
        'uprev_bk_penalty': [params.uprev_bk_penalty * 0.85, params.uprev_bk_penalty, params.uprev_bk_penalty * 1.15],
        'dnrev_bk_penalty': [params.dnrev_bk_penalty * 0.85, params.dnrev_bk_penalty, params.dnrev_bk_penalty * 1.15],
    }


def calibrate_ths_params(val_records, breakout_threshold, reversion_threshold, base_params=DEFAULT_THS_CORE_PARAMS, rounds=2):
    best_params = base_params
    best_score, best_metrics, _ = evaluate_ths_records(
        val_records,
        best_params,
        breakout_threshold,
        reversion_threshold,
        best_params.dir_gap,
    )
    grid = build_calibration_grid(base_params)
    for _ in range(rounds):
        improved = False
        for field, candidates in grid.items():
            local_best_score = best_score
            local_best_params = best_params
            local_best_metrics = best_metrics
            for candidate in candidates:
                params = best_params.updated(**{field: float(candidate)})
                score, metrics, _ = evaluate_ths_records(
                    val_records,
                    params,
                    breakout_threshold,
                    reversion_threshold,
                    params.dir_gap,
                )
                if score > local_best_score + 1e-6:
                    local_best_score = score
                    local_best_params = params
                    local_best_metrics = metrics
            if local_best_params != best_params:
                best_params = local_best_params
                best_score = local_best_score
                best_metrics = local_best_metrics
                improved = True
        if not improved:
            break
    return best_params, best_score, best_metrics


def evaluate_version(version_name, device):
    checkpoint_dir = os.path.join(WEIGHTS_ROOT, version_name)
    best_path = os.path.join(checkpoint_dir, f'{version_name}_best.pth')
    final_path = os.path.join(checkpoint_dir, f'{version_name}_final.pth')
    if not os.path.exists(best_path):
        raise FileNotFoundError(best_path)
    if not os.path.exists(final_path):
        raise FileNotFoundError(final_path)

    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    final_ckpt = torch.load(final_path, map_location=device, weights_only=False)
    model, args = build_model(best_ckpt, device)
    dataset_profile = args.get('dataset_profile', 'iterA2')
    arch_version = args.get('arch_version', 'iterA2_base')
    score_profile = args.get('score_profile', 'default')
    timeframes = [
        normalize_timeframe_label(item)
        for item in args.get('timeframes', DEFAULT_ASHARE_TIMEFRAMES)
    ]
    timeframes = [item for item in timeframes if item] or list(DEFAULT_ASHARE_TIMEFRAMES)

    coverage_report = build_market_coverage_report(
        data_dir=DATA_DIR,
        market='ashare',
        primary_assets=ASHARE_PRIMARY_ASSETS,
        fallback_assets=ASHARE_FALLBACK_ASSETS,
        timeframes=timeframes,
        train_end=args.get('train_end', DEFAULT_TRAIN_END),
        val_end=args.get('val_end', DEFAULT_VAL_END),
        test_start=args.get('test_start', DEFAULT_TEST_START),
        training_subdir='ashare',
    )
    selected_assets = coverage_report['asset_resolution']['selected_assets'][:len(ASHARE_PRIMARY_ASSETS)]
    file_records = discover_training_files(
        data_dir=DATA_DIR,
        market='ashare',
        assets=selected_assets,
        timeframes=timeframes,
        training_subdir='ashare',
    )

    split_results = {'val': [], 'test': []}
    for record in file_records:
        datasets, metadata = create_ashare_dataset_splits(
            file_path=record['path'],
            window_size=args.get('window_size', 20),
            horizon=args.get('horizon', 10),
            train_end=args.get('train_end', DEFAULT_TRAIN_END),
            val_end=args.get('val_end', DEFAULT_VAL_END),
            test_start=args.get('test_start', DEFAULT_TEST_START),
            fast_full=False,
            return_metadata=True,
            dataset_profile=dataset_profile,
        )
        for split_name in ('val', 'test'):
            dataset = datasets.get(split_name)
            if dataset is None or len(dataset) == 0:
                continue
            result = evaluate_dataset_split(model, dataset, device, arch_version=arch_version)
            if result is None:
                continue
            result['asset_code'] = metadata['asset_code']
            result['timeframe'] = metadata['timeframe']
            split_results[split_name].append(result)

    if not split_results['test']:
        raise RuntimeError(f'No test records evaluated for {version_name}.')

    test_model_summary = aggregate_model_records(split_results['test'], score_profile=score_profile)
    test_by_timeframe = summarize_group(split_results['test'], 'timeframe', score_profile=score_profile)
    test_by_asset = summarize_group(split_results['test'], 'asset_code', score_profile=score_profile)
    breakout_threshold = test_model_summary['breakout_eval']['threshold']
    reversion_threshold = test_model_summary['reversion_eval']['threshold']

    default_val_score, default_val_metrics, _ = evaluate_ths_records(
        split_results['val'],
        DEFAULT_THS_CORE_PARAMS,
        breakout_threshold,
        reversion_threshold,
        DEFAULT_THS_CORE_PARAMS.dir_gap,
    ) if split_results['val'] else (0.0, None, [])
    default_test_score, default_test_metrics, _ = evaluate_ths_records(
        split_results['test'],
        DEFAULT_THS_CORE_PARAMS,
        breakout_threshold,
        reversion_threshold,
        DEFAULT_THS_CORE_PARAMS.dir_gap,
    )

    calibrated_params = DEFAULT_THS_CORE_PARAMS
    calibrated_val_score = default_val_score
    calibrated_val_metrics = default_val_metrics
    calibrated_test_score = default_test_score
    calibrated_test_metrics = default_test_metrics
    if version_name != 'iterA1_ashare' and split_results['val']:
        calibrated_params, calibrated_val_score, calibrated_val_metrics = calibrate_ths_params(
            split_results['val'],
            breakout_threshold,
            reversion_threshold,
            base_params=DEFAULT_THS_CORE_PARAMS,
        )
        calibrated_test_score, calibrated_test_metrics, _ = evaluate_ths_records(
            split_results['test'],
            calibrated_params,
            breakout_threshold,
            reversion_threshold,
            calibrated_params.dir_gap,
        )

    feature_names = best_ckpt.get('feature_names', PHYSICS_FEATURE_NAMES)
    all_features = np.vstack([item['features'] for item in split_results['test']])
    all_predictions = np.vstack([item['predictions'] for item in split_results['test']])
    breakout_rules = kernel_rules(all_predictions[:, 0], all_features, feature_names)
    reversion_rules = kernel_rules(np.maximum(all_predictions[:, 1], 0.0), all_features, feature_names)

    return {
        'version_name': version_name,
        'best_path': best_path,
        'final_path': final_path,
        'best_ckpt': best_ckpt,
        'final_ckpt': final_ckpt,
        'overall_model': test_model_summary,
        'model_by_timeframe': test_by_timeframe,
        'model_by_asset': test_by_asset,
        'breakout_threshold': breakout_threshold,
        'reversion_threshold': reversion_threshold,
        'breakout_rules': breakout_rules,
        'reversion_rules': reversion_rules,
        'default_ths': {
            'val_objective': default_val_score,
            'val_metrics': default_val_metrics,
            'test_objective': default_test_score,
            'test_metrics': default_test_metrics,
        },
        'calibrated_ths': {
            'params': calibrated_params,
            'val_objective': calibrated_val_score,
            'val_metrics': calibrated_val_metrics,
            'test_objective': calibrated_test_score,
            'test_metrics': calibrated_test_metrics,
        },
    }


def write_version_outputs(
    result,
    baseline_result=None,
    baseline_ths_source='default',
    promotion_thresholds=None,
):
    version_name = result['version_name']
    log_dir = os.path.join(LOG_ROOT, version_name)
    os.makedirs(log_dir, exist_ok=True)

    current_default = result['default_ths']
    selected_params = DEFAULT_THS_CORE_PARAMS
    selected_name = 'default'
    selected_metrics = current_default
    recommendation = 'keep_current_default'
    threshold_checks = None

    if version_name != 'iterA1_ashare' and result['calibrated_ths']['test_objective'] > current_default['test_objective']:
        selected_params = result['calibrated_ths']['params']
        selected_name = 'calibrated'
        selected_metrics = result['calibrated_ths']
        recommendation = 'candidate_calibrated'

    compare_summary = None
    if baseline_result is not None:
        baseline_metric_block = resolve_ths_metric_source(baseline_result, baseline_ths_source)
        model_win = result['overall_model']['composite_score'] > baseline_result['overall_model']['composite_score']
        ths_win = selected_metrics['test_objective'] > baseline_metric_block['test_objective']
        compare_summary = {
            'baseline_version': baseline_result['version_name'],
            'baseline_ths_source': baseline_ths_source,
            'model_win': bool(model_win),
            'ths_win': bool(ths_win),
            'baseline_model_composite': baseline_result['overall_model']['composite_score'],
            'candidate_model_composite': result['overall_model']['composite_score'],
            'baseline_ths_objective': baseline_metric_block['test_objective'],
            'candidate_ths_objective': selected_metrics['test_objective'],
        }
    if promotion_thresholds:
        timeframe_thresholds = promotion_thresholds.get('timeframe_composite', {})
        threshold_checks = {
            'overall_model_composite': {
                'threshold': promotion_thresholds.get('overall_model_composite'),
                'actual': result['overall_model']['composite_score'],
            },
            'calibrated_ths_test_objective': {
                'threshold': promotion_thresholds.get('calibrated_ths_test_objective'),
                'actual': result['calibrated_ths']['test_objective'],
            },
            'timeframe_composite': {},
        }
        passes = []
        overall_threshold = threshold_checks['overall_model_composite']['threshold']
        if overall_threshold is not None:
            passes.append(result['overall_model']['composite_score'] >= overall_threshold)
        calibrated_threshold = threshold_checks['calibrated_ths_test_objective']['threshold']
        if calibrated_threshold is not None:
            passes.append(result['calibrated_ths']['test_objective'] >= calibrated_threshold)
        for timeframe, threshold in timeframe_thresholds.items():
            actual = result['model_by_timeframe'].get(timeframe, {}).get('composite_score')
            threshold_checks['timeframe_composite'][timeframe] = {
                'threshold': threshold,
                'actual': actual,
            }
            passes.append(actual is not None and actual >= threshold)
        threshold_checks['all_passed'] = bool(passes) and all(passes)
        if threshold_checks['all_passed']:
            recommendation = f'promote_{selected_name}_params'
        else:
            recommendation = 'keep_current_default'
            selected_params = DEFAULT_THS_CORE_PARAMS
            selected_name = 'default'
            selected_metrics = current_default
    elif baseline_result is not None:
        if compare_summary['model_win'] and compare_summary['ths_win']:
            recommendation = f'promote_{selected_name}_params'
        else:
            recommendation = 'keep_current_default'
            selected_params = DEFAULT_THS_CORE_PARAMS
            selected_name = 'default'
            selected_metrics = current_default

    model_json_path = os.path.join(log_dir, f'{version_name}_model_eval.json')
    ths_json_path = os.path.join(log_dir, f'{version_name}_ths_alignment.json')
    params_json_path = os.path.join(log_dir, f'{version_name}_ths_params.json')
    compare_json_path = os.path.join(log_dir, f'{version_name}_comparison.json')
    model_report_path = os.path.join(log_dir, f'{version_name}_model_report.md')
    ths_report_path = os.path.join(log_dir, f'{version_name}_ths_alignment_report.md')

    model_payload = {
        'version_name': version_name,
        'best_path': result['best_path'],
        'final_path': result['final_path'],
        'best_val_loss': result['best_ckpt'].get('val_loss'),
        'best_score': result['best_ckpt'].get('best_score'),
        'final_env': result['final_ckpt'].get('env'),
        'overall_model': result['overall_model'],
        'model_by_timeframe': result['model_by_timeframe'],
        'model_by_asset': result['model_by_asset'],
        'breakout_rules': {
            'threshold': result['breakout_rules'][0],
            'tree': result['breakout_rules'][1],
        },
        'reversion_rules': {
            'threshold': result['reversion_rules'][0],
            'tree': result['reversion_rules'][1],
        },
    }
    ths_payload = {
        'version_name': version_name,
        'formula_path': FORMULA_PATH,
        'default_params': DEFAULT_THS_CORE_PARAMS.to_dict(),
        'selected_param_source': selected_name,
        'selected_params': selected_params.to_dict(),
        'default_ths': current_default,
        'calibrated_ths': {
            'params': result['calibrated_ths']['params'].to_dict(),
            'val_objective': result['calibrated_ths']['val_objective'],
            'val_metrics': result['calibrated_ths']['val_metrics'],
            'test_objective': result['calibrated_ths']['test_objective'],
            'test_metrics': result['calibrated_ths']['test_metrics'],
        },
        'recommendation': recommendation,
        'comparison': compare_summary,
        'promotion_thresholds': promotion_thresholds,
        'threshold_checks': threshold_checks,
    }

    Path(model_json_path).write_text(json.dumps(to_builtin(model_payload), ensure_ascii=False, indent=2), encoding='utf-8')
    Path(ths_json_path).write_text(json.dumps(to_builtin(ths_payload), ensure_ascii=False, indent=2), encoding='utf-8')
    Path(params_json_path).write_text(json.dumps(to_builtin({
        'version_name': version_name,
        'selected_param_source': selected_name,
        'selected_params': selected_params.to_dict(),
        'recommendation': recommendation,
    }), ensure_ascii=False, indent=2), encoding='utf-8')
    if compare_summary is not None or threshold_checks is not None:
        Path(compare_json_path).write_text(json.dumps(to_builtin({
            'comparison': compare_summary,
            'threshold_checks': threshold_checks,
        }), ensure_ascii=False, indent=2), encoding='utf-8')

    model_lines = [
        f'# {version_name} Model Evaluation',
        '',
        f'- best_path: `{result["best_path"]}`',
        f'- final_path: `{result["final_path"]}`',
        f'- best_val_loss: `{result["best_ckpt"].get("val_loss")}`',
        f'- best_score: `{result["best_ckpt"].get("best_score")}`',
        f'- final_env: `{result["final_ckpt"].get("env")}`',
        '',
        '## Overall OOS',
        '',
        f'- breakout_corr: `{result["overall_model"]["breakout_corr"]:.4f}`',
        f'- reversion_corr: `{result["overall_model"]["reversion_corr"]:.4f}`',
        f'- breakout_eval: `{result["overall_model"]["breakout_eval"]}`',
        f'- reversion_eval: `{result["overall_model"]["reversion_eval"]}`',
        f'- direction_eval: `{result["overall_model"]["direction_eval"]}`',
        f'- composite_score: `{result["overall_model"]["composite_score"]:.4f}`',
        '',
        '## By Timeframe',
        '',
    ]
    for timeframe, summary in result['model_by_timeframe'].items():
        model_lines.append(
            f'- `{timeframe}` | breakout_f1=`{summary["breakout_eval"]["f1"]:.4f}` | '
            f'reversion_f1=`{summary["reversion_eval"]["f1"]:.4f}` | '
            f'direction_macro_f1=`{summary["direction_eval"]["macro_f1"]:.4f}` | '
            f'composite=`{summary["composite_score"]:.4f}`'
        )
    model_lines.extend(['', '## Probe Rules', '', f'- breakout_threshold: `{result["breakout_rules"][0]:.6f}`', '```text', result['breakout_rules'][1], '```'])
    model_lines.extend([f'- reversion_threshold: `{result["reversion_rules"][0]:.6f}`', '```text', result['reversion_rules'][1], '```'])
    Path(model_report_path).write_text('\n'.join(model_lines) + '\n', encoding='utf-8')

    ths_lines = [
        f'# {version_name} THS Alignment Report',
        '',
        f'- formula_path: `{FORMULA_PATH}`',
        f'- selected_param_source: `{selected_name}`',
        f'- recommendation: `{recommendation}`',
        '',
        '## Default Params',
        '',
        f'- val_objective: `{current_default["val_objective"]:.4f}`',
        f'- test_objective: `{current_default["test_objective"]:.4f}`',
        f'- label_macro_f1: `{current_default["test_metrics"]["label_metrics"]["macro_f1"]:.4f}`',
        f'- teacher_macro_f1: `{current_default["test_metrics"]["teacher_metrics"]["macro_f1"]:.4f}`',
        '',
        '## Calibrated Params',
        '',
        f'- val_objective: `{result["calibrated_ths"]["val_objective"]:.4f}`',
        f'- test_objective: `{result["calibrated_ths"]["test_objective"]:.4f}`',
        f'- label_macro_f1: `{result["calibrated_ths"]["test_metrics"]["label_metrics"]["macro_f1"]:.4f}`',
        f'- teacher_macro_f1: `{result["calibrated_ths"]["test_metrics"]["teacher_metrics"]["macro_f1"]:.4f}`',
        '',
        '## Selected Params',
        '',
    ]
    for field in TUNABLE_THS_FIELDS:
        if field in selected_params.to_dict():
            ths_lines.append(f'- `{field}` = `{selected_params.to_dict()[field]}`')
    if compare_summary is not None:
        ths_lines.extend([
            '',
            '## Baseline Comparison',
            '',
            f'- baseline_version: `{compare_summary["baseline_version"]}`',
            f'- model_win: `{compare_summary["model_win"]}`',
            f'- ths_win: `{compare_summary["ths_win"]}`',
            f'- baseline_model_composite: `{compare_summary["baseline_model_composite"]:.4f}`',
            f'- candidate_model_composite: `{compare_summary["candidate_model_composite"]:.4f}`',
            f'- baseline_ths_objective: `{compare_summary["baseline_ths_objective"]:.4f}`',
            f'- candidate_ths_objective: `{compare_summary["candidate_ths_objective"]:.4f}`',
        ])
    if threshold_checks is not None:
        ths_lines.extend([
            '',
            '## Promotion Thresholds',
            '',
            f'- overall_model_composite: `{threshold_checks["overall_model_composite"]["actual"]:.4f}` / `{threshold_checks["overall_model_composite"]["threshold"]}`',
            f'- calibrated_ths_test_objective: `{threshold_checks["calibrated_ths_test_objective"]["actual"]:.4f}` / `{threshold_checks["calibrated_ths_test_objective"]["threshold"]}`',
        ])
        for timeframe, payload in threshold_checks['timeframe_composite'].items():
            ths_lines.append(f'- `{timeframe}` composite: `{payload["actual"]:.4f}` / `{payload["threshold"]}`')
        ths_lines.append(f'- all_passed: `{threshold_checks["all_passed"]}`')
    Path(ths_report_path).write_text('\n'.join(ths_lines) + '\n', encoding='utf-8')
    return {
        'model_json_path': model_json_path,
        'ths_json_path': ths_json_path,
        'params_json_path': params_json_path,
        'compare_json_path': compare_json_path if (compare_summary is not None or threshold_checks is not None) else None,
        'model_report_path': model_report_path,
        'ths_report_path': ths_report_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='iterA2_ashare')
    parser.add_argument('--compare-version', type=str, default='iterA1_ashare')
    parser.add_argument('--no-compare', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compare_version = None if args.no_compare else args.compare_version
    baseline_result = evaluate_version(compare_version, device) if compare_version and compare_version != args.version else None
    current_result = evaluate_version(args.version, device)
    output_paths = write_version_outputs(current_result, baseline_result=baseline_result)

    print('=== Analysis Complete ===')
    print({'version': args.version, 'outputs': output_paths})


if __name__ == '__main__':
    main()
