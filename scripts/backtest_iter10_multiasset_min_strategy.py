import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def sign(x):
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def safe_log_return(c0, c1):
    if c0 <= 0 or c1 <= 0:
        return 0.0
    return math.log(c1 / c0)


def run_strategy(rows, hold_bars=8):
    pos = 0.0
    hold = 0
    pnl = []
    for i in range(len(rows) - 1):
        r = rows[i]
        close = float(r['close'])
        ema20 = float(r['ema20'])
        sigma = max(float(r['sigma']), 1e-6)
        breakout_sig = int(r['breakout_signal'])
        reversion_sig = int(r['reversion_signal'])
        if reversion_sig:
            pos = -sign(close - ema20)
            hold = hold_bars
        elif breakout_sig:
            pos = sign(close - ema20)
            hold = hold_bars
        elif hold > 0:
            hold -= 1
        else:
            pos = 0.0
        ret = safe_log_return(close, float(rows[i + 1]['close']))
        pnl.append(pos * ret)
    return pnl


def run_baseline(rows, hold_bars=8, z=1.5):
    pos = 0.0
    hold = 0
    pnl = []
    for i in range(len(rows) - 1):
        r = rows[i]
        close = float(r['close'])
        ema20 = float(r['ema20'])
        sigma = max(float(r['sigma']), 1e-6)
        zscore = abs(math.log(max(close, 1e-8) / max(ema20, 1e-8))) / sigma
        if zscore >= z:
            pos = -sign(close - ema20)
            hold = hold_bars
        elif hold > 0:
            hold -= 1
        else:
            pos = 0.0
        ret = safe_log_return(close, float(rows[i + 1]['close']))
        pnl.append(pos * ret)
    return pnl


def sharpe(pnls):
    if not pnls:
        return 0.0
    mean = sum(pnls) / len(pnls)
    var = sum((x - mean) ** 2 for x in pnls) / max(len(pnls) - 1, 1)
    std = math.sqrt(max(var, 1e-12))
    return mean / std * math.sqrt(252.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--hold_bars', type=int, default=8)
    args = parser.parse_args()

    signals_dir = Path(args.signals_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (signals_dir.parent / 'iter10_backtest_summary.csv')
    results = []
    agg_khaos = []
    agg_base = []
    for fp in sorted(signals_dir.glob('*.csv')):
        with fp.open('r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 50:
            continue
        khaos_pnl = run_strategy(rows, hold_bars=args.hold_bars)
        base_pnl = run_baseline(rows, hold_bars=args.hold_bars)
        agg_khaos.extend(khaos_pnl)
        agg_base.extend(base_pnl)
        results.append({
            'series': fp.stem,
            'khaos_total': sum(khaos_pnl),
            'khaos_sharpe': sharpe(khaos_pnl),
            'baseline_total': sum(base_pnl),
            'baseline_sharpe': sharpe(base_pnl),
        })

    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['series', 'khaos_total', 'khaos_sharpe', 'baseline_total', 'baseline_sharpe'])
        for r in results:
            w.writerow([
                r['series'],
                f"{r['khaos_total']:.6f}",
                f"{r['khaos_sharpe']:.4f}",
                f"{r['baseline_total']:.6f}",
                f"{r['baseline_sharpe']:.4f}",
            ])
        w.writerow([
            '__ALL__',
            f"{sum(agg_khaos):.6f}",
            f"{sharpe(agg_khaos):.4f}",
            f"{sum(agg_base):.6f}",
            f"{sharpe(agg_base):.4f}",
        ])
    print(out_path.as_posix())


if __name__ == '__main__':
    main()

