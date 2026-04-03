import os
import sys

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = r'D:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》'
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'Finance', '02_核心代码', '源代码')
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

from khaos.数据处理.ashare_dataset import create_ashare_dataset_splits
from khaos.数据处理.ashare_support import (
    ASHARE_FALLBACK_ASSETS,
    ASHARE_PRIMARY_ASSETS,
    DEFAULT_ASHARE_TIMEFRAMES,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    build_market_coverage_report,
    discover_training_files,
)
from khaos.模型定义.kan import KHAOS_KAN
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES


DATA_DIR = os.path.join(PROJECT_ROOT, 'Finance', '01_数据中心', '03_研究数据', 'research_processed')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'Finance', '02_核心代码', '模型权重备份', 'iterA1_ashare')
BEST_PATH = os.path.join(CHECKPOINT_DIR, 'iterA1_ashare_best.pth')
FINAL_PATH = os.path.join(CHECKPOINT_DIR, 'iterA1_ashare_final.pth')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'Finance', '04_项目文档', '04_实验报告', 'KHAOS_A股_iterA1_Training_Report.md')


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
        accuracy = (tp + tn) / max(len(scores), 1)
        hard_negative_rate = np.mean(pred[hard_negative_flags]) if np.any(hard_negative_flags) else 0.0
        event_rate = np.mean(pred[event_flags]) if np.any(event_flags) else 0.0
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'event_rate': float(event_rate),
            'hard_negative_rate': float(hard_negative_rate),
            'signal_frequency': float(np.mean(pred)),
            'label_frequency': label_frequency,
        }
        if best is None or candidate['f1'] > best['f1'] or (
            candidate['f1'] == best['f1'] and candidate['hard_negative_rate'] < best['hard_negative_rate']
        ):
            best = candidate
    return best


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_model(checkpoint, device):
    args = checkpoint.get('args', {})
    model = KHAOS_KAN(
        input_dim=len(checkpoint.get('feature_names', PHYSICS_FEATURE_NAMES)),
        hidden_dim=args.get('hidden_dim', 64),
        output_dim=2,
        layers=args.get('layers', 3),
        grid_size=args.get('grid_size', 10),
        num_heads=4,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, args


def evaluate_record(model, record, args, device):
    datasets, metadata = create_ashare_dataset_splits(
        file_path=record['path'],
        window_size=args.get('window_size', 20),
        horizon=args.get('horizon', 10),
        train_end=args.get('train_end', DEFAULT_TRAIN_END),
        val_end=args.get('val_end', DEFAULT_VAL_END),
        test_start=args.get('test_start', DEFAULT_TEST_START),
        fast_full=False,
        return_metadata=True,
    )
    test_ds = datasets.get('test')
    if test_ds is None or len(test_ds) == 0:
        return None
    loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
    preds = []
    targets = []
    flags = []
    features = []
    with torch.no_grad():
        for batch_x, batch_y, _, _, _, batch_flags in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds.append(pred.detach().cpu().numpy())
            targets.append(batch_y.numpy())
            flags.append(batch_flags.numpy())
            features.append(batch_x[:, -1, :].detach().cpu().numpy())
    pred_np = np.vstack(preds)
    target_np = np.vstack(targets)
    flag_np = np.vstack(flags)
    feature_np = np.vstack(features)
    breakout_scores = pred_np[:, 0]
    reversion_scores = np.maximum(pred_np[:, 1], 0.0)
    return {
        'asset_code': metadata['asset_code'],
        'timeframe': metadata['timeframe'],
        'predictions': pred_np,
        'targets': target_np,
        'flags': flag_np,
        'features': feature_np,
        'breakout_eval': compute_event_metrics(breakout_scores, flag_np[:, 0] > 0.5, flag_np[:, 2] > 0.5),
        'reversion_eval': compute_event_metrics(reversion_scores, flag_np[:, 1] > 0.5, flag_np[:, 3] > 0.5),
        'breakout_corr': safe_corr(breakout_scores, target_np[:, 0]),
        'reversion_corr': safe_corr(reversion_scores, target_np[:, 1]),
    }


def aggregate_group(records):
    if not records:
        return None
    pred_np = np.vstack([item['predictions'] for item in records])
    target_np = np.vstack([item['targets'] for item in records])
    flag_np = np.vstack([item['flags'] for item in records])
    breakout_scores = pred_np[:, 0]
    reversion_scores = np.maximum(pred_np[:, 1], 0.0)
    return {
        'breakout_eval': compute_event_metrics(breakout_scores, flag_np[:, 0] > 0.5, flag_np[:, 2] > 0.5),
        'reversion_eval': compute_event_metrics(reversion_scores, flag_np[:, 1] > 0.5, flag_np[:, 3] > 0.5),
        'breakout_corr': safe_corr(breakout_scores, target_np[:, 0]),
        'reversion_corr': safe_corr(reversion_scores, target_np[:, 1]),
    }


def kernel_rules(kernel_scores, features, feature_names):
    threshold = np.percentile(kernel_scores, 90)
    labels = (kernel_scores >= threshold).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree.fit(features, labels)
    return threshold, export_text(tree, feature_names=feature_names)


def write_report(best_ckpt, final_ckpt, overall, by_timeframe, by_asset, breakout_rules, reversion_rules):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        '# KHAOS A股 iterA1 Training Report',
        '',
        '## Checkpoints',
        '',
        f"- best_path: `{BEST_PATH}`",
        f"- final_path: `{FINAL_PATH}`",
        f"- best_val_loss: `{best_ckpt.get('val_loss')}`",
        f"- best_score: `{best_ckpt.get('best_score')}`",
        f"- final_env: `{final_ckpt.get('env')}`",
        '',
        '## Overall OOS',
        '',
        f"- breakout_corr: `{overall['breakout_corr']:.4f}`",
        f"- reversion_corr: `{overall['reversion_corr']:.4f}`",
        f"- breakout_eval: `{overall['breakout_eval']}`",
        f"- reversion_eval: `{overall['reversion_eval']}`",
        '',
        '## By Timeframe',
        '',
    ]
    for timeframe, summary in by_timeframe.items():
        lines.append(
            f"- `{timeframe}` | breakout_f1=`{summary['breakout_eval']['f1']:.4f}` | "
            f"reversion_f1=`{summary['reversion_eval']['f1']:.4f}` | "
            f"corr=`{summary['breakout_corr']:.4f}/{summary['reversion_corr']:.4f}`"
        )
    lines.extend(['', '## By Asset', ''])
    for asset_code, summary in by_asset.items():
        lines.append(
            f"- `{asset_code}` | breakout_f1=`{summary['breakout_eval']['f1']:.4f}` | "
            f"reversion_f1=`{summary['reversion_eval']['f1']:.4f}` | "
            f"corr=`{summary['breakout_corr']:.4f}/{summary['reversion_corr']:.4f}`"
        )
    lines.extend([
        '',
        '## Probe Rules',
        '',
        f"- breakout_threshold: `{breakout_rules[0]:.6f}`",
        '```text',
        breakout_rules[1],
        '```',
        f"- reversion_threshold: `{reversion_rules[0]:.6f}`",
        '```text',
        reversion_rules[1],
        '```',
    ])
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_ckpt = torch.load(BEST_PATH, map_location=device, weights_only=False)
    final_ckpt = torch.load(FINAL_PATH, map_location=device, weights_only=False)
    model, args = build_model(best_ckpt, device)

    coverage_report = build_market_coverage_report(
        data_dir=DATA_DIR,
        market='ashare',
        primary_assets=ASHARE_PRIMARY_ASSETS,
        fallback_assets=ASHARE_FALLBACK_ASSETS,
        timeframes=DEFAULT_ASHARE_TIMEFRAMES,
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
        timeframes=DEFAULT_ASHARE_TIMEFRAMES,
        training_subdir='ashare',
    )

    evaluated_records = []
    for record in file_records:
        result = evaluate_record(model, record, args, device)
        if result is not None:
            evaluated_records.append(result)

    overall = aggregate_group(evaluated_records)
    by_timeframe = {}
    for timeframe in DEFAULT_ASHARE_TIMEFRAMES:
        group_records = [item for item in evaluated_records if item['timeframe'] == timeframe]
        summary = aggregate_group(group_records)
        if summary is not None:
            by_timeframe[timeframe] = summary

    by_asset = {}
    for asset_code in selected_assets:
        group_records = [item for item in evaluated_records if item['asset_code'] == asset_code]
        summary = aggregate_group(group_records)
        if summary is not None:
            by_asset[asset_code] = summary

    feature_names = best_ckpt.get('feature_names', PHYSICS_FEATURE_NAMES)
    all_features = np.vstack([item['features'] for item in evaluated_records])
    all_predictions = np.vstack([item['predictions'] for item in evaluated_records])
    breakout_rules = kernel_rules(all_predictions[:, 0], all_features, feature_names)
    reversion_rules = kernel_rules(np.maximum(all_predictions[:, 1], 0.0), all_features, feature_names)

    print('=== IterA1 Checkpoints ===')
    print('BEST', {'val_loss': best_ckpt.get('val_loss'), 'best_score': best_ckpt.get('best_score')})
    print('FINAL', {'env': final_ckpt.get('env'), 'args': final_ckpt.get('args', {})})
    print('\n=== Overall OOS ===')
    print(overall)
    print('\n=== By Timeframe ===')
    print(by_timeframe)
    print('\n=== By Asset ===')
    print(by_asset)

    write_report(best_ckpt, final_ckpt, overall, by_timeframe, by_asset, breakout_rules, reversion_rules)
    print(f'Report written to {REPORT_PATH}')


if __name__ == '__main__':
    main()
