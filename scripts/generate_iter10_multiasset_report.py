import argparse
import json
from collections import defaultdict
from pathlib import Path
import csv


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


def fmt(x, digits=4):
    if x is None:
        return ''
    try:
        return f'{float(x):.{digits}f}'
    except Exception:
        return str(x)


def render_table(headers, rows):
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def as_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def total_signal_frequency(value):
    if isinstance(value, dict):
        return as_float(value.get('breakout')) + as_float(value.get('reversion'))
    return as_float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (run_dir / 'iter10_report.md')

    epoch_rows = load_jsonl(run_dir / 'epoch_metrics.jsonl')
    per_asset_rows = load_jsonl(run_dir / 'per_asset_metrics.jsonl')
    per_timeframe_rows = load_jsonl(run_dir / 'per_timeframe_metrics.jsonl')

    best_epoch = None
    best_score = None
    for row in epoch_rows:
        score = row.get('composite_score')
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = row.get('epoch')

    lines = []
    lines.append(f'# iter10 多资产闭环报告（run_dir={run_dir.as_posix()}）')
    lines.append('')
    if best_epoch is not None:
        lines.append(f'- best_epoch: {best_epoch}')
        lines.append(f'- best_score: {fmt(best_score)}')
    lines.append(f'- epochs: {len(epoch_rows)}')
    lines.append(f'- per_asset_records: {len(per_asset_rows)}')
    lines.append('')

    if epoch_rows:
        headers = ['epoch', 'val_loss', 'composite_score', 'public_violation', 'score_veto_passed', 'avg_signal_frequency']
        table = []
        for row in epoch_rows:
            score_veto = row.get('score_veto') or {}
            avg_sf = (row.get('recent_score_components') or {}).get('avg_signal_frequency')
            table.append([
                str(row.get('epoch', '')),
                fmt(row.get('val_loss')),
                fmt(row.get('composite_score')),
                fmt(row.get('public_below_directional_violation_rate')),
                str(bool(score_veto.get('passed', False))),
                fmt(avg_sf),
            ])
        lines.append('## Epoch 概览')
        lines.append(render_table(headers, table))
        lines.append('')

    if best_epoch is not None and per_asset_rows:
        best_asset_rows = [r for r in per_asset_rows if r.get('epoch') == best_epoch]
        by_tf = defaultdict(list)
        by_asset = defaultdict(list)
        for r in best_asset_rows:
            by_tf[r.get('timeframe')].append(r)
            by_asset[r.get('asset')].append(r)

        lines.append('## Best Epoch：分周期汇总')
        tf_table = []
        for tf in sorted(by_tf.keys()):
            rows = by_tf[tf]
            tf_table.append([
                tf,
                str(sum(r.get('sample_count', 0) for r in rows)),
                fmt(sum(r.get('breakout_f1', 0.0) for r in rows) / max(len(rows), 1)),
                fmt(sum(r.get('reversion_f1', 0.0) for r in rows) / max(len(rows), 1)),
                fmt(sum(r.get('public_below_directional_violation_rate', 0.0) for r in rows) / max(len(rows), 1)),
                fmt(sum(total_signal_frequency(r.get('signal_frequency')) for r in rows) / max(len(rows), 1)),
            ])
        lines.append(render_table(
            ['timeframe', 'sample_count', 'breakout_f1(avg)', 'reversion_f1(avg)', 'public_violation(avg)', 'signal_frequency(avg)'],
            tf_table,
        ))
        lines.append('')

        lines.append('## Best Epoch：分资产×分周期明细')
        detail_headers = [
            'asset', 'timeframe', 'breakout_f1', 'reversion_f1',
            'breakout_precision', 'reversion_precision',
            'public_violation', 'signal_frequency', 'thresholds_frozen', 'frozen_thresholds',
        ]
        detail_rows = []
        for r in sorted(best_asset_rows, key=lambda x: (x.get('asset', ''), x.get('timeframe', ''))):
            detail_rows.append([
                str(r.get('asset', '')),
                str(r.get('timeframe', '')),
                fmt(r.get('breakout_f1')),
                fmt(r.get('reversion_f1')),
                fmt(r.get('breakout_precision')),
                fmt(r.get('reversion_precision')),
                fmt(r.get('public_below_directional_violation_rate')),
                fmt(total_signal_frequency(r.get('signal_frequency'))),
                str(bool(r.get('thresholds_frozen', False))),
                json.dumps(r.get('frozen_thresholds'), ensure_ascii=False),
            ])
        lines.append(render_table(detail_headers, detail_rows))
        lines.append('')

    backtest_path = run_dir / 'iter10_backtest_summary.csv'
    if backtest_path.exists():
        with backtest_path.open('r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        lines.append('## 最小策略回测（signals=60m 示例）')
        bt_rows = []
        for r in rows:
            bt_rows.append([
                r.get('series', ''),
                r.get('khaos_total', ''),
                r.get('khaos_sharpe', ''),
                r.get('baseline_total', ''),
                r.get('baseline_sharpe', ''),
            ])
        lines.append(render_table(['series', 'khaos_total', 'khaos_sharpe', 'baseline_total', 'baseline_sharpe'], bt_rows))
        lines.append('')

    if per_asset_rows:
        lines.append('## 每 epoch 的冻结阈值（timeframe 级，来自 per_asset_metrics）')
        rows = []
        seen = set()
        for r in sorted(per_asset_rows, key=lambda x: (x.get('epoch', 0), x.get('timeframe', ''), x.get('asset', ''))):
            key = (r.get('epoch'), r.get('timeframe'))
            if key in seen:
                continue
            seen.add(key)
            rows.append([
                str(r.get('epoch', '')),
                str(r.get('timeframe', '')),
                str(bool(r.get('thresholds_frozen', False))),
                json.dumps(r.get('frozen_thresholds'), ensure_ascii=False),
            ])
        lines.append(render_table(['epoch', 'timeframe', 'thresholds_frozen', 'frozen_thresholds'], rows))
        lines.append('')

    out_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    print(out_path.as_posix())


if __name__ == '__main__':
    main()
