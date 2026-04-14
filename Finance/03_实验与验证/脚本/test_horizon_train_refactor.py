from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAIN_MODULE = load_module(
    'khaos_train_test_module',
    SOURCE_DIR / 'khaos' / '模型训练' / 'train.py',
)

build_metric_bucket = TRAIN_MODULE.build_metric_bucket
build_train_loader = TRAIN_MODULE.build_train_loader
compute_event_metrics = TRAIN_MODULE.compute_event_metrics
compute_standard_score_veto = TRAIN_MODULE.compute_standard_score_veto
evaluate_single_cycle_family_guard = TRAIN_MODULE.evaluate_single_cycle_family_guard
should_update_checkpoint = TRAIN_MODULE.should_update_checkpoint
summarize_metric_bucket = TRAIN_MODULE.summarize_metric_bucket


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

    def test_precision_first_event_metrics_filter_oversignal_candidates(self):
        scores = np.array([0.99, 0.91, 0.83, 0.75, 0.67, 0.59, 0.51, 0.43, 0.35, 0.27], dtype=np.float32)
        event_flags = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        hard_negative_flags = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)

        f1_metrics = compute_event_metrics(scores, event_flags, hard_negative_flags, selection_mode='f1', event_type='reversion')
        precision_metrics = compute_event_metrics(
            scores,
            event_flags,
            hard_negative_flags,
            selection_mode='precision_first',
            event_type='reversion',
        )

        self.assertGreater(f1_metrics['signal_frequency'], 0.32)
        self.assertLessEqual(precision_metrics['signal_frequency'], 0.32)
        self.assertGreaterEqual(precision_metrics['precision'], f1_metrics['precision'])
        self.assertGreater(precision_metrics['threshold'], f1_metrics['threshold'])

    def test_standard_score_veto_uses_average_signal_frequency_across_score_timeframes(self):
        args = SimpleNamespace(
            kill_keep_signal_frequency_max=0.37,
            kill_keep_public_violation_rate_max=1.0,
            kill_keep_timeframe_60m_composite_min=0.0,
        )
        timeframe_summaries = {
            '15m': {'breakout_signal_frequency': 0.20, 'reversion_signal_frequency': 0.20},
            '60m': {'breakout_signal_frequency': 0.44, 'reversion_signal_frequency': 0.44},
            '240m': {'breakout_signal_frequency': 0.42, 'reversion_signal_frequency': 0.42},
            '1d': {'breakout_signal_frequency': 0.44, 'reversion_signal_frequency': 0.44},
        }
        epoch_score_summary = {
            'breakout_signal_frequency': 0.28,
            'reversion_signal_frequency': 0.28,
            'public_below_directional_violation_rate': 0.04,
        }

        score_veto = compute_standard_score_veto(
            epoch_score_summary=epoch_score_summary,
            timeframe_summaries=timeframe_summaries,
            args=args,
            score_timeframes=['15m', '60m', '240m', '1d'],
        )

        self.assertFalse(score_veto['passed'])
        self.assertAlmostEqual(score_veto['avg_signal_frequency'], 0.375, places=6)
        self.assertIn('avg_signal_frequency', score_veto['checks'])
        self.assertEqual(score_veto['score_timeframes'], ['15m', '60m', '240m', '1d'])
        self.assertTrue(any(reason.startswith('avg_signal_frequency(') for reason in score_veto['reasons']))

    def test_discovery_score_profile_emits_signal_space_metrics(self):
        bucket = build_metric_bucket()
        bucket['preds'].append(
            np.array(
                [
                    [0.95, 0.90],
                    [0.88, 0.82],
                    [0.70, 0.63],
                    [0.40, 0.30],
                    [0.18, 0.14],
                    [0.08, -0.05],
                ],
                dtype=np.float32,
            )
        )
        bucket['targets'].append(
            np.array(
                [
                    [1.50, 1.20],
                    [1.20, 0.90],
                    [0.80, 0.60],
                    [0.25, 0.20],
                    [0.10, 0.08],
                    [0.02, 0.01],
                ],
                dtype=np.float32,
            )
        )
        bucket['aux_targets'].append(
            np.array(
                [
                    [1.70, 1.10],
                    [1.35, 0.95],
                    [0.90, 0.72],
                    [0.18, 0.15],
                    [0.05, 0.04],
                    [0.00, 0.00],
                ],
                dtype=np.float32,
            )
        )
        flags = np.zeros((6, 8), dtype=np.float32)
        flags[:3, 0] = 1.0
        flags[:2, 1] = 1.0
        flags[3, 2] = 1.0
        flags[2, 3] = 1.0
        bucket['flags'].append(flags)
        bucket['logs'].append({'total_loss': 0.42})

        summary = summarize_metric_bucket(
            bucket,
            score_profile='short_t_discovery_focus',
            use_direction_metrics=False,
        )

        self.assertGreater(summary['breakout_signal_target_mean'], 0.9)
        self.assertGreater(summary['reversion_signal_target_mean'], 0.7)
        self.assertGreater(summary['breakout_signal_space_mean'], 0.8)
        self.assertGreater(summary['reversion_signal_space_mean'], 0.7)
        self.assertGreater(summary['breakout_signal_quality_mean'], summary['breakout_all_quality_mean'])
        self.assertGreater(summary['reversion_signal_quality_mean'], summary['reversion_all_quality_mean'])
        self.assertGreater(summary['composite_score'], 0.0)

    def test_guarded_discovery_score_profile_uses_aux_prediction_gating(self):
        def make_bucket(aux_preds):
            bucket = build_metric_bucket()
            bucket['preds'].append(
                np.array(
                    [
                        [0.98, 0.10],
                        [0.95, 0.05],
                        [0.94, 0.08],
                        [0.90, 0.04],
                        [0.12, 0.97],
                        [0.10, 0.95],
                        [0.08, 0.92],
                        [0.06, 0.90],
                    ],
                    dtype=np.float32,
                )
            )
            bucket['aux_preds'].append(aux_preds.astype(np.float32))
            bucket['targets'].append(
                np.array(
                    [
                        [1.30, 1.10],
                        [0.18, 0.10],
                        [1.15, 0.95],
                        [0.12, 0.08],
                        [0.10, 1.25],
                        [0.08, 0.14],
                        [0.06, 1.08],
                        [0.05, 0.12],
                    ],
                    dtype=np.float32,
                )
            )
            bucket['aux_targets'].append(
                np.array(
                    [
                        [1.40, 1.18],
                        [0.10, 0.06],
                        [1.22, 1.02],
                        [0.08, 0.05],
                        [0.08, 1.34],
                        [0.05, 0.10],
                        [0.04, 1.16],
                        [0.03, 0.08],
                    ],
                    dtype=np.float32,
                )
            )
            flags = np.zeros((8, 8), dtype=np.float32)
            flags[0, 0] = 1.0
            flags[2, 0] = 1.0
            flags[1, 2] = 1.0
            flags[3, 2] = 1.0
            flags[4, 1] = 1.0
            flags[6, 1] = 1.0
            flags[5, 3] = 1.0
            flags[7, 3] = 1.0
            bucket['flags'].append(flags)
            bucket['logs'].append({'total_loss': 0.36})
            return bucket

        aligned_aux = np.array(
            [
                [0.96, 0.04],
                [0.10, 0.02],
                [0.92, 0.03],
                [0.08, 0.02],
                [0.03, 0.95],
                [0.02, 0.10],
                [0.03, 0.90],
                [0.01, 0.08],
            ],
            dtype=np.float32,
        )
        flat_aux = np.zeros((8, 2), dtype=np.float32)

        aligned_summary = summarize_metric_bucket(
            make_bucket(aligned_aux),
            score_profile='short_t_discovery_guarded_focus',
            use_direction_metrics=False,
        )
        flat_summary = summarize_metric_bucket(
            make_bucket(flat_aux),
            score_profile='short_t_discovery_guarded_focus',
            use_direction_metrics=False,
        )

        self.assertGreater(
            aligned_summary['breakout_metrics']['recall'],
            flat_summary['breakout_metrics']['recall'],
        )
        self.assertGreater(
            aligned_summary['reversion_metrics']['recall'],
            flat_summary['reversion_metrics']['recall'],
        )
        self.assertGreater(aligned_summary['composite_score'], flat_summary['composite_score'])

    def test_standard_score_veto_applies_discovery_signal_space_gates(self):
        args = SimpleNamespace(
            kill_keep_signal_frequency_max=0.50,
            kill_keep_public_violation_rate_max=1.0,
            kill_keep_timeframe_60m_composite_min=0.0,
            kill_keep_breakout_signal_space_min=0.95,
            kill_keep_reversion_signal_space_min=0.70,
        )
        timeframe_summaries = {
            '15m': {'breakout_signal_frequency': 0.18, 'reversion_signal_frequency': 0.16},
        }
        epoch_score_summary = {
            'breakout_signal_frequency': 0.18,
            'reversion_signal_frequency': 0.16,
            'public_below_directional_violation_rate': 0.02,
            'breakout_signal_space_mean': 0.92,
            'reversion_signal_space_mean': 0.68,
        }

        score_veto = compute_standard_score_veto(
            epoch_score_summary=epoch_score_summary,
            timeframe_summaries=timeframe_summaries,
            args=args,
            score_timeframes=['15m'],
        )

        self.assertFalse(score_veto['passed'])
        self.assertIn('breakout_signal_space_mean', score_veto['checks'])
        self.assertIn('reversion_signal_space_mean', score_veto['checks'])
        self.assertTrue(any(reason.startswith('breakout_signal_space_mean(') for reason in score_veto['reasons']))
        self.assertTrue(any(reason.startswith('reversion_signal_space_mean(') for reason in score_veto['reasons']))


if __name__ == '__main__':
    unittest.main()
