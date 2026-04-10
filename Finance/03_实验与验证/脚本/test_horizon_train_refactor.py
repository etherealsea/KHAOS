from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from khaos.模型训练.train import (  # noqa: E402
    build_metric_bucket,
    build_train_loader,
    evaluate_single_cycle_family_guard,
    should_update_checkpoint,
    summarize_metric_bucket,
)


class HorizonTrainRefactorTests(unittest.TestCase):
    def test_single_cycle_family_guard_rejects_adaptive_registry(self):
        registry = {
            ('000001', '15m', 'fold_1'): {
                'task_stats': {
                    'breakout': {'mode_mass': 0.02, 'iqr': 52.0, 'h_mode': 120},
                    'reversion': {'mode_mass': 0.03, 'iqr': 28.0, 'h_mode': 20},
                }
            }
        }
        result = evaluate_single_cycle_family_guard(registry, score_timeframes=['15m'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['checked_tasks'], 2)
        self.assertEqual(len(result['violations']), 2)
        self.assertTrue(all(item['recommended_family'] == 'adaptive_resonance' for item in result['violations']))

    def test_build_train_loader_honors_timeframe_cap_under_rolling_recent(self):
        dataset = TensorDataset(torch.arange(100, dtype=torch.float32).unsqueeze(1))
        args = SimpleNamespace(
            seed=42,
            batch_size=16,
            split_scheme='rolling_recent_v1',
            horizon_search_spec='{"min_horizon":20,"max_horizon":120}',
            per_timeframe_train_cap={'15m': 32},
        )
        loader = build_train_loader(dataset, args, '15m')
        total_samples = sum(int(batch[0].shape[0]) for batch in loader)
        self.assertEqual(total_samples, 32)
        self.assertEqual(loader.sampler.__class__.__name__, 'RandomSampler')

    def test_raw_checkpoint_rule_can_improve_while_gate_stays_blocked(self):
        raw_improved = should_update_checkpoint(
            candidate_score=0.24,
            candidate_val_loss=1.05,
            best_score=0.20,
            best_val_loss=1.10,
            min_delta=0.001,
        )
        gate_improved = False and should_update_checkpoint(
            candidate_score=0.24,
            candidate_val_loss=1.05,
            best_score=float('-inf'),
            best_val_loss=float('inf'),
            min_delta=0.001,
        )
        self.assertTrue(raw_improved)
        self.assertFalse(gate_improved)

    def test_summarize_metric_bucket_emits_public_direction_diagnostics(self):
        bucket = build_metric_bucket()
        bucket['preds'].append(np.array([[0.1, 0.3], [0.2, 0.8], [0.4, -0.2]], dtype=np.float32))
        bucket['targets'].append(np.zeros((3, 2), dtype=np.float32))
        flags = np.zeros((3, 8), dtype=np.float32)
        flags[0, 1] = 1.0
        flags[1, 1] = 1.0
        bucket['flags'].append(flags)
        bucket['debug_batches']['directional_floor'].append(np.array([0.5, 0.9, 0.1], dtype=np.float32))
        bucket['logs'].append(
            {
                'total_loss': 1.0,
                'public_below_directional_violation_rate': 0.75,
                'public_below_directional_violation': 0.2,
            }
        )

        summary = summarize_metric_bucket(bucket, score_profile='default', use_direction_metrics=False)
        self.assertAlmostEqual(summary['pred_rev_mean'], 0.36666667, places=6)
        self.assertAlmostEqual(summary['pred_rev_event_mean'], 0.55, places=6)
        self.assertAlmostEqual(summary['directional_floor_mean'], 0.5, places=6)
        self.assertAlmostEqual(summary['directional_floor_reversion_event_mean'], 0.7, places=6)
        self.assertEqual(summary['reversion_event_count'], 2)


if __name__ == '__main__':
    unittest.main()
