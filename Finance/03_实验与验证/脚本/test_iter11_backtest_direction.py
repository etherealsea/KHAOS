from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backtest_iter10_multiasset_min_strategy import run_strategy


class Iter11BacktestDirectionTests(unittest.TestCase):
    def test_direction_uses_bull_bear_scores(self):
        rows = [
            {
                'close': '100',
                'ema20': '100',
                'sigma': '0.01',
                'breakout_signal': '1',
                'reversion_signal': '0',
                'bear_score': '0.2',
                'bull_score': '0.8',
            },
            {
                'close': '101',
                'ema20': '100',
                'sigma': '0.01',
                'breakout_signal': '0',
                'reversion_signal': '1',
                'bear_score': '0.9',
                'bull_score': '0.1',
            },
            {
                'close': '99',
                'ema20': '100',
                'sigma': '0.01',
                'breakout_signal': '0',
                'reversion_signal': '0',
                'bear_score': '0.0',
                'bull_score': '0.0',
            },
        ]
        pnl = run_strategy(rows, hold_bars=1)
        self.assertGreater(pnl[0], 0.0)
        self.assertGreater(pnl[1], 0.0)


if __name__ == '__main__':
    unittest.main()

