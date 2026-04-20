from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from khaos.模型训练.train import compute_event_metrics


class Iter11PrecisionFirstScoringTests(unittest.TestCase):
    def test_precision_first_prefers_high_precision_threshold(self):
        scores = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40], dtype=np.float32)
        event_flags = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
        hard_negative_flags = np.zeros_like(event_flags)
        metrics = compute_event_metrics(
            scores,
            event_flags > 0.5,
            hard_negative_flags > 0.5,
            selection_mode='precision_first',
            event_type='breakout',
            score_profile='iter11_precision_first',
        )
        self.assertGreaterEqual(float(metrics['precision']), 0.70)
        self.assertGreaterEqual(float(metrics['threshold']), 0.79)


if __name__ == '__main__':
    unittest.main()

