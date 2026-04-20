from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from khaos.数据处理.data_loader import build_breakout_discovery_targets


class Iter11MfeMaeTargetsTests(unittest.TestCase):
    def test_breakout_targets_rank_trend_higher_than_chop(self):
        horizon = 4
        n = 200
        t = np.arange(n, dtype=np.float32)
        log_close_trend = 0.002 * t
        log_close_chop = 0.02 * np.sin(t / 3.0).astype(np.float32)

        def _compute(log_close):
            log_close = log_close.astype(np.float32)
            close = np.exp(log_close).astype(np.float32)
            high = close * 1.01
            low = close * 0.99
            log_high = np.log(high + 1e-8).astype(np.float32)
            log_low = np.log(low + 1e-8).astype(np.float32)
            returns = np.diff(log_close, prepend=log_close[0]).astype(np.float32)
            sigma = np.full_like(log_close, 0.005, dtype=np.float32)
            entropy = np.zeros_like(log_close, dtype=np.float32)
            thresholds = {
                'event_target_threshold': 0.8,
                'hard_negative_target_threshold': 0.2,
                'hard_negative_mae_threshold': 1.5,
            }
            target, aux, event, hn = build_breakout_discovery_targets(
                log_close,
                log_high,
                log_low,
                returns,
                sigma,
                entropy,
                horizon,
                threshold_config=thresholds,
                preset='guarded_v1',
            )
            valid = slice(0, len(target) - horizon)
            return float(np.mean(target[valid])), float(np.mean(event[valid]))

        trend_target_mean, trend_event_mean = _compute(log_close_trend)
        chop_target_mean, chop_event_mean = _compute(log_close_chop)
        self.assertGreater(trend_target_mean, chop_target_mean)
        self.assertGreater(trend_event_mean, 0.5)


if __name__ == '__main__':
    unittest.main()

