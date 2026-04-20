import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import sys


SOURCE_DIR = Path('/workspace/Finance/02_核心代码/源代码').resolve()
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_asset_timeframe(csv_name: str):
    stem = Path(csv_name).stem
    if '_' not in stem:
        return stem, None
    asset, tf = stem.split('_', 1)
    tf_map = {'1h': '60m', '60m': '60m', '15m': '15m', '5m': '5m', '1d': '1d', '4h': '240m'}
    return asset, tf_map.get(tf, tf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--training_subdir', type=str, default='iter10_multiasset')
    parser.add_argument('--checkpoint', type=str, default='khaos_kan_best_gate.pth')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--assets', type=str, default=None)
    parser.add_argument('--timeframes', type=str, default=None)
    parser.add_argument('--fast_full', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path

    from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES
    from khaos.模型定义.kan import KHAOS_KAN
    from khaos.数据处理.data_loader import create_rolling_datasets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KHAOS_KAN(
        input_dim=len(PHYSICS_FEATURE_NAMES),
        hidden_dim=64,
        output_dim=2,
        layers=3,
        grid_size=10,
        arch_version='iterA4_multiscale',
        horizon_count=1,
        horizon_family_mode='legacy',
    ).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch_rows = load_jsonl(run_dir / 'epoch_metrics.jsonl')
    best_epoch = None
    best_score = None
    for row in epoch_rows:
        score = row.get('composite_score')
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = row.get('epoch')
    target_epoch = args.epoch or best_epoch

    per_asset_rows = load_jsonl(run_dir / 'per_asset_metrics.jsonl')
    threshold_map = {}
    for r in per_asset_rows:
        if target_epoch is not None and r.get('epoch') != target_epoch:
            continue
        frozen = r.get('frozen_thresholds') or {}
        key = (r.get('asset'), r.get('timeframe'))
        threshold_map[key] = (
            float(frozen.get('breakout')) if frozen.get('breakout') is not None else None,
            float(frozen.get('reversion')) if frozen.get('reversion') is not None else None,
        )

    assets = [s.strip() for s in args.assets.split(',')] if args.assets else None
    timeframes = [s.strip() for s in args.timeframes.split(',')] if args.timeframes else None
    signals_dir = run_dir / 'signals'
    signals_dir.mkdir(parents=True, exist_ok=True)

    training_ready_dir = data_dir / 'training_ready' / args.training_subdir
    csv_files = sorted([p for p in training_ready_dir.glob('*.csv') if p.is_file()])
    for csv_path in csv_files:
        asset, tf = parse_asset_timeframe(csv_path.name)
        if assets and asset not in assets:
            continue
        if timeframes and tf not in timeframes:
            continue
        th = threshold_map.get((asset, tf))
        if not th or th[0] is None or th[1] is None:
            continue

        train_ds, test_ds = create_rolling_datasets(
            str(csv_path),
            window_size=20,
            horizon=4,
            fast_full=args.fast_full,
        )
        loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        breakout_scores = []
        reversion_scores = []
        bear_scores = []
        bull_scores = []
        sample_indices = []
        with torch.no_grad():
            seen = 0
            for batch in loader:
                x = batch[0].to(device)
                main_pred, info = model(x, return_debug=True)
                bs = main_pred[:, 0].detach().cpu().numpy()
                rs = main_pred[:, 1].detach().cpu().numpy()
                breakout_scores.append(bs)
                reversion_scores.append(rs)
                bear = info.get('bear_score')
                bull = info.get('bull_score')
                if bear is None or bull is None:
                    bear = torch.zeros((main_pred.size(0), 1), device=main_pred.device, dtype=main_pred.dtype)
                    bull = torch.zeros((main_pred.size(0), 1), device=main_pred.device, dtype=main_pred.dtype)
                bear_scores.append(bear.detach().cpu().numpy().reshape(-1))
                bull_scores.append(bull.detach().cpu().numpy().reshape(-1))
                bsz = len(bs)
                sample_indices.extend(range(seen, seen + bsz))
                seen += bsz
        breakout_scores = np.concatenate(breakout_scores, axis=0)
        reversion_scores = np.concatenate(reversion_scores, axis=0)
        bear_scores = np.concatenate(bear_scores, axis=0)
        bull_scores = np.concatenate(bull_scores, axis=0)
        window_size = int(getattr(test_ds, 'window_size', 20))
        out_path = signals_dir / f'{asset}_{tf}.csv'
        with out_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'time', 'close', 'ema20', 'sigma',
                'bear_score', 'bull_score', 'direction',
                'breakout_score', 'reversion_score',
                'breakout_threshold', 'reversion_threshold',
                'breakout_signal', 'reversion_signal',
            ])
            breakout_th, reversion_th = th
            for i, si in enumerate(sample_indices):
                end_idx = si + window_size - 1
                t = test_ds.time[end_idx] if getattr(test_ds, 'time', None) is not None else end_idx
                close = float(test_ds.close[end_idx])
                ema20 = float(test_ds.ema20[end_idx])
                sigma = float(test_ds.sigma[end_idx])
                bear = float(bear_scores[i])
                bull = float(bull_scores[i])
                direction = 1 if bull >= bear else -1
                bs = float(breakout_scores[i])
                rs = float(reversion_scores[i])
                b_sig = int(bs >= breakout_th)
                r_sig = int(rs >= reversion_th)
                w.writerow([t, close, ema20, sigma, bear, bull, direction, bs, rs, breakout_th, reversion_th, b_sig, r_sig])
        print(out_path.as_posix())


if __name__ == '__main__':
    main()
