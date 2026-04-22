from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


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
    'khaos_iter12_score_test_module',
    SOURCE_DIR / 'khaos' / '模型训练' / 'train.py',
)

compute_recent_precision_score = TRAIN_MODULE.compute_recent_precision_score
compute_standard_score_veto = TRAIN_MODULE.compute_standard_score_veto
build_metric_bucket = TRAIN_MODULE.build_metric_bucket
summarize_metric_bucket = TRAIN_MODULE.summarize_metric_bucket
EVENT_FLAG_INDEX = TRAIN_MODULE.EVENT_FLAG_INDEX


def make_fold_summary(
    public_violation_rate,
    directional_floor_quality,
    breakout_precision=0.58,
    reversion_precision=0.46,
    breakout_hn=0.10,
    reversion_hn=0.12,
    signal_frequency=0.18,
    direction_macro_f1=0.62,
):
    return {
        'breakout_metrics': {
            'precision': breakout_precision,
            'hard_negative_rate': breakout_hn,
        },
        'reversion_metrics': {
            'precision': reversion_precision,
            'hard_negative_rate': reversion_hn,
        },
        'breakout_signal_frequency': signal_frequency,
        'reversion_signal_frequency': signal_frequency,
        'public_below_directional_violation_rate': public_violation_rate,
        'directional_floor_mean': directional_floor_quality,
        'directional_floor_reversion_event_mean': directional_floor_quality,
        'direction_metrics': {
            'macro_f1': direction_macro_f1,
        },
    }


class Iter12GuardedPrecisionFirstTests(unittest.TestCase):
    def test_summarize_metric_bucket_handles_horizon_direction_debug_tensors(self):
        bucket = build_metric_bucket()
        bucket['preds'].append(
            [
                [0.20, 0.80],
                [0.82, 0.18],
            ]
        )
        bucket['targets'].append([[0.0, 0.0], [0.0, 0.0]])
        flags = [[0.0] * 8 for _ in range(2)]
        flags[0][EVENT_FLAG_INDEX['reversion_down_context']] = 1.0
        flags[1][EVENT_FLAG_INDEX['reversion_up_context']] = 1.0
        bucket['flags'].append(flags)
        bucket['debug_batches']['bear_score'].append(
            [
                [[0.20, 0.30, 0.40], [0.80, 0.85, 0.90]],
                [[0.60, 0.55, 0.50], [0.25, 0.20, 0.15]],
            ]
        )
        bucket['debug_batches']['bull_score'].append(
            [
                [[0.10, 0.15, 0.20], [0.25, 0.20, 0.15]],
                [[0.30, 0.35, 0.40], [0.80, 0.85, 0.90]],
            ]
        )
        bucket['logs'].append({'total_loss': 0.5})

        summary = summarize_metric_bucket(
            bucket,
            score_profile='iter12_guarded_precision_first',
            use_direction_metrics=True,
        )

        self.assertEqual(summary['direction_metrics']['support'], 2)
        self.assertGreaterEqual(summary['direction_metrics']['accuracy'], 0.0)

    def test_recent_score_rewards_structural_quality_in_addition_to_precision(self):
        args = SimpleNamespace(
            score_profile='iter12_guarded_precision_first',
            breakout_precision_floor=0.0,
            reversion_precision_floor=0.0,
            public_violation_cap=0.20,
            signal_frequency_cap_ratio=0.70,
        )
        timeframe_summaries = {
            '60m': {
                'breakout_metrics': {'precision': 0.54, 'hard_negative_rate': 0.12},
                'reversion_metrics': {'precision': 0.50, 'hard_negative_rate': 0.14},
                'breakout_signal_frequency': 0.19,
                'reversion_signal_frequency': 0.19,
            }
        }
        structurally_good = {
            'fold_3': make_fold_summary(0.03, 0.78),
            'fold_4': make_fold_summary(0.04, 0.74),
        }
        structurally_bad = {
            'fold_3': make_fold_summary(0.18, 0.12),
            'fold_4': make_fold_summary(0.17, 0.10),
        }

        good_score, good_components, good_veto = compute_recent_precision_score(
            fold_summaries=structurally_good,
            timeframe_summaries=timeframe_summaries,
            baseline_reference=None,
            args=args,
        )
        bad_score, bad_components, bad_veto = compute_recent_precision_score(
            fold_summaries=structurally_bad,
            timeframe_summaries=timeframe_summaries,
            baseline_reference=None,
            args=args,
        )

        self.assertTrue(good_veto['passed'])
        self.assertTrue(bad_veto['passed'])
        self.assertGreater(good_components['recent_public_feasibility'], bad_components['recent_public_feasibility'])
        self.assertGreater(
            good_components['recent_directional_floor_quality'],
            bad_components['recent_directional_floor_quality'],
        )
        self.assertGreater(good_score, bad_score)

    def test_standard_veto_emits_60m_diagnostics_without_reintroducing_hard_gate(self):
        args = SimpleNamespace(
            kill_keep_signal_frequency_max=0.40,
            kill_keep_public_violation_rate_max=0.20,
            kill_keep_timeframe_60m_composite_min=0.0,
        )
        epoch_score_summary = {
            'breakout_signal_frequency': 0.22,
            'reversion_signal_frequency': 0.22,
            'public_below_directional_violation_rate': 0.04,
        }
        timeframe_summaries = {
            '15m': {'breakout_signal_frequency': 0.22, 'reversion_signal_frequency': 0.22},
            '60m': {
                'breakout_signal_frequency': 0.26,
                'reversion_signal_frequency': 0.24,
                'breakout_metrics': {'precision': 0.57, 'hard_negative_rate': 0.11},
                'reversion_metrics': {'precision': 0.49, 'hard_negative_rate': 0.17},
                'composite_score': 0.21,
            },
        }

        score_veto = compute_standard_score_veto(
            epoch_score_summary=epoch_score_summary,
            timeframe_summaries=timeframe_summaries,
            args=args,
            score_timeframes=['15m', '60m'],
        )

        self.assertTrue(score_veto['passed'])
        self.assertIn('timeframe_60m_min_precision_diagnostic', score_veto['checks'])
        self.assertIn('timeframe_60m_hard_negative_rate_diagnostic', score_veto['checks'])
        self.assertAlmostEqual(
            score_veto['checks']['timeframe_60m_min_precision_diagnostic']['actual'],
            0.49,
            places=6,
        )
        self.assertAlmostEqual(
            score_veto['checks']['timeframe_60m_hard_negative_rate_diagnostic']['actual'],
            0.14,
            places=6,
        )


if __name__ == '__main__':
    unittest.main()
