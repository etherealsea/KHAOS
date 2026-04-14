from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
RUNNER_PATH = (
    PROJECT_ROOT
    / 'Finance'
    / '03_实验与验证'
    / '脚本'
    / '测试与临时脚本'
    / 'run_teacher_first_ashare_ablation_train.py'
)
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DATASET_MODULE = load_module(
    'khaos_ashare_dataset_test_module',
    SOURCE_DIR / 'khaos' / '数据处理' / 'ashare_dataset.py',
)
TRAIN_MODULE = load_module(
    'khaos_train_cache_test_module',
    SOURCE_DIR / 'khaos' / '模型训练' / 'train.py',
)
RUNNER_MODULE = load_module(
    'teacher_first_runner_test_module',
    RUNNER_PATH,
)

AshareFinancialDataset = DATASET_MODULE.AshareFinancialDataset
create_ashare_dataset_splits = DATASET_MODULE.create_ashare_dataset_splits
SHORT_T_DISCOVERY_V1_TIMEFRAME_WEIGHT = DATASET_MODULE.SHORT_T_DISCOVERY_V1_TIMEFRAME_WEIGHT
SHORT_T_DISCOVERY_GUARDED_V1_TIMEFRAME_WEIGHT = DATASET_MODULE.SHORT_T_DISCOVERY_GUARDED_V1_TIMEFRAME_WEIGHT
SHORT_T_DISCOVERY_GUARDED_V2_TIMEFRAME_WEIGHT = DATASET_MODULE.SHORT_T_DISCOVERY_GUARDED_V2_TIMEFRAME_WEIGHT
SHORT_T_PRECISION_V2_TIMEFRAME_WEIGHT = DATASET_MODULE.SHORT_T_PRECISION_V2_TIMEFRAME_WEIGHT
build_runtime_dataset_cache_path = TRAIN_MODULE.build_runtime_dataset_cache_path
create_market_datasets = TRAIN_MODULE.create_market_datasets


def make_ashare_frame(rows=220, freq='D'):
    times = pd.date_range('2023-01-01', periods=rows, freq=freq)
    base = np.linspace(10.0, 14.0, rows, dtype=np.float32)
    close = base + 0.15 * np.sin(np.linspace(0.0, 12.0, rows, dtype=np.float32))
    open_ = close - 0.05
    high = close + 0.08
    low = close - 0.08
    volume = np.linspace(1_000_000.0, 1_200_000.0, rows, dtype=np.float32)
    return pd.DataFrame(
        {
            'time': times,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }
    )


class RuntimeDatasetCacheTests(unittest.TestCase):
    def test_horizon_dataset_getitem_reuses_prebuilt_tensors(self):
        df = make_ashare_frame(rows=180, freq='D')
        dataset = AshareFinancialDataset(
            df,
            window_size=20,
            forecast_horizon=10,
            timeframe_label='1d',
            dataset_profile='shortT_precision_v2',
            horizon_search_spec={'min_horizon': 20, 'max_horizon': 24, 'max_fraction': 0.2},
            horizon_family_mode='adaptive_resonance',
            global_horizon_grid=[20, 21, 22, 23, 24],
        )

        sample = dataset[0]
        self.assertEqual(len(sample), 7)
        for value in sample[:6]:
            self.assertTrue(torch.is_tensor(value))

        horizon_payload = sample[6]
        for key, value in horizon_payload.items():
            self.assertTrue(torch.is_tensor(value), msg=key)
        self.assertTrue(hasattr(dataset, 'candidate_horizons_tensor'))
        self.assertFalse(hasattr(dataset, 'global_horizons_grid_tensor'))
        self.assertTrue(hasattr(dataset, 'global_horizon_grid_tensor'))
        self.assertTrue(dataset.use_continuation_flag)

    def test_shortt_precision_v2_weights_and_caps_match_dual_precision_plan(self):
        self.assertAlmostEqual(SHORT_T_PRECISION_V2_TIMEFRAME_WEIGHT['60m'], 0.68, places=6)
        self.assertAlmostEqual(SHORT_T_PRECISION_V2_TIMEFRAME_WEIGHT['240m'], 1.10, places=6)
        self.assertLess(
            SHORT_T_PRECISION_V2_TIMEFRAME_WEIGHT['60m'],
            SHORT_T_PRECISION_V2_TIMEFRAME_WEIGHT['15m'],
        )

        spec = RUNNER_MODULE.EXPERIMENT_SPECS['shortT_dual_precision_v1']
        self.assertEqual(spec['dataset_profile'], 'shortT_precision_v2')
        self.assertEqual(spec['loss_profile'], 'shortT_precision_v2')
        self.assertEqual(spec['score_profile'], 'short_t_precision_focus')
        self.assertEqual(spec['per_timeframe_train_cap']['60m'], 2048)
        self.assertEqual(spec['smoke_per_timeframe_train_cap']['60m'], 512)
        self.assertEqual(spec['kill_keep_signal_frequency_max'], 0.37)
        self.assertEqual(spec['kill_keep_public_violation_rate_max'], 0.10)
        self.assertEqual(spec['kill_keep_timeframe_60m_composite_min'], 0.38)
        self.assertIsNone(spec['promotion_overall_composite_threshold'])
        self.assertIsNone(spec['promotion_timeframe_composite_thresholds'])

    def test_shortt_discovery_v1_spec_and_namespace_match_space_gate_plan(self):
        self.assertAlmostEqual(SHORT_T_DISCOVERY_V1_TIMEFRAME_WEIGHT['60m'], 0.74, places=6)
        self.assertAlmostEqual(SHORT_T_DISCOVERY_V1_TIMEFRAME_WEIGHT['240m'], 1.08, places=6)
        self.assertLess(
            SHORT_T_DISCOVERY_V1_TIMEFRAME_WEIGHT['60m'],
            SHORT_T_DISCOVERY_V1_TIMEFRAME_WEIGHT['240m'],
        )

        spec = RUNNER_MODULE.EXPERIMENT_SPECS['shortT_discovery_v1']
        self.assertEqual(spec['dataset_profile'], 'shortT_discovery_v1')
        self.assertEqual(spec['loss_profile'], 'shortT_discovery_v1')
        self.assertEqual(spec['score_profile'], 'short_t_discovery_focus')
        self.assertEqual(spec['constraint_profile'], 'teacher_feasible_discovery_v1')
        self.assertEqual(spec['per_timeframe_train_cap']['60m'], 2560)
        self.assertEqual(spec['smoke_per_timeframe_train_cap']['60m'], 640)
        self.assertEqual(spec['kill_keep_breakout_signal_space_min'], 0.95)
        self.assertEqual(spec['kill_keep_reversion_signal_space_min'], 0.70)
        self.assertIsNone(spec['promotion_overall_composite_threshold'])
        self.assertIsNone(spec['promotion_timeframe_composite_thresholds'])

        with tempfile.TemporaryDirectory() as tmpdir:
            namespace = RUNNER_MODULE.build_namespace(
                experiment_name='shortT_discovery_v1',
                save_dir=Path(tmpdir) / 'weights',
                assets=['000001'],
                epochs=1,
                batch_size=8,
                fast_full=False,
                per_timeframe_train_cap=spec['per_timeframe_train_cap'],
                constraint_profile='default',
                resume=False,
            )
        self.assertEqual(namespace.dataset_profile, 'shortT_discovery_v1')
        self.assertEqual(namespace.loss_profile, 'shortT_discovery_v1')
        self.assertEqual(namespace.score_profile, 'short_t_discovery_focus')
        self.assertEqual(namespace.kill_keep_breakout_signal_space_min, 0.95)
        self.assertEqual(namespace.kill_keep_reversion_signal_space_min, 0.70)
        self.assertEqual(namespace.kill_keep_signal_frequency_max, 0.40)

    def test_shortt_discovery_guarded_v1_spec_and_namespace_match_guarded_plan(self):
        self.assertAlmostEqual(SHORT_T_DISCOVERY_GUARDED_V1_TIMEFRAME_WEIGHT['60m'], 0.60, places=6)
        self.assertAlmostEqual(SHORT_T_DISCOVERY_GUARDED_V1_TIMEFRAME_WEIGHT['240m'], 1.12, places=6)
        self.assertLess(
            SHORT_T_DISCOVERY_GUARDED_V1_TIMEFRAME_WEIGHT['60m'],
            SHORT_T_DISCOVERY_GUARDED_V1_TIMEFRAME_WEIGHT['240m'],
        )

        spec = RUNNER_MODULE.EXPERIMENT_SPECS['shortT_discovery_guarded_v1']
        self.assertEqual(spec['dataset_profile'], 'shortT_discovery_guarded_v1')
        self.assertEqual(spec['loss_profile'], 'shortT_discovery_guarded_v1')
        self.assertEqual(spec['score_profile'], 'short_t_discovery_guarded_focus')
        self.assertEqual(spec['constraint_profile'], 'teacher_feasible_discovery_v1')
        self.assertEqual(spec['per_timeframe_train_cap']['60m'], 2560)
        self.assertEqual(spec['smoke_per_timeframe_train_cap']['60m'], 640)
        self.assertEqual(spec['kill_keep_signal_frequency_max'], 0.36)
        self.assertEqual(spec['kill_keep_breakout_signal_space_min'], 0.88)
        self.assertEqual(spec['kill_keep_reversion_signal_space_min'], 0.64)
        self.assertFalse(spec['allow_resume'])
        self.assertIsNone(spec['warm_start_path'])

        with tempfile.TemporaryDirectory() as tmpdir:
            namespace = RUNNER_MODULE.build_namespace(
                experiment_name='shortT_discovery_guarded_v1',
                save_dir=Path(tmpdir) / 'weights',
                assets=['000001'],
                epochs=1,
                batch_size=8,
                fast_full=False,
                per_timeframe_train_cap=spec['per_timeframe_train_cap'],
                constraint_profile='default',
                resume=True,
            )
        self.assertEqual(namespace.dataset_profile, 'shortT_discovery_guarded_v1')
        self.assertEqual(namespace.loss_profile, 'shortT_discovery_guarded_v1')
        self.assertEqual(namespace.score_profile, 'short_t_discovery_guarded_focus')
        self.assertFalse(namespace.resume)
        self.assertFalse(namespace.warm_start_weights_only)
        self.assertEqual(namespace.kill_keep_signal_frequency_max, 0.36)
        self.assertEqual(namespace.kill_keep_breakout_signal_space_min, 0.88)
        self.assertEqual(namespace.kill_keep_reversion_signal_space_min, 0.64)

    def test_shortt_discovery_guarded_v2_spec_and_namespace_match_short_bias_plan(self):
        self.assertAlmostEqual(SHORT_T_DISCOVERY_GUARDED_V2_TIMEFRAME_WEIGHT['15m'], 1.28, places=6)
        self.assertAlmostEqual(SHORT_T_DISCOVERY_GUARDED_V2_TIMEFRAME_WEIGHT['1d'], 0.50, places=6)
        self.assertGreater(
            SHORT_T_DISCOVERY_GUARDED_V2_TIMEFRAME_WEIGHT['15m'],
            SHORT_T_DISCOVERY_GUARDED_V2_TIMEFRAME_WEIGHT['240m'],
        )

        spec = RUNNER_MODULE.EXPERIMENT_SPECS['shortT_discovery_guarded_v2']
        self.assertEqual(spec['dataset_profile'], 'shortT_discovery_guarded_v2')
        self.assertEqual(spec['loss_profile'], 'shortT_discovery_guarded_v2')
        self.assertEqual(spec['score_profile'], 'short_t_discovery_guarded_focus')
        self.assertEqual(spec['score_timeframes'], ['15m', '60m'])
        self.assertEqual(spec['aux_timeframes'], ['5m', '240m', '1d'])
        self.assertEqual(spec['per_timeframe_train_cap']['15m'], 7168)
        self.assertEqual(spec['per_timeframe_train_cap']['1d'], 1024)
        self.assertEqual(spec['kill_keep_signal_frequency_max'], 0.34)
        self.assertEqual(spec['kill_keep_timeframe_60m_composite_min'], 0.33)

        with tempfile.TemporaryDirectory() as tmpdir:
            namespace = RUNNER_MODULE.build_namespace(
                experiment_name='shortT_discovery_guarded_v2',
                save_dir=Path(tmpdir) / 'weights',
                assets=['000001'],
                epochs=1,
                batch_size=8,
                fast_full=False,
                per_timeframe_train_cap=spec['per_timeframe_train_cap'],
                constraint_profile='default',
                resume=False,
            )
        self.assertEqual(namespace.dataset_profile, 'shortT_discovery_guarded_v2')
        self.assertEqual(namespace.loss_profile, 'shortT_discovery_guarded_v2')
        self.assertEqual(namespace.score_timeframes, ['15m', '60m'])
        self.assertEqual(namespace.aux_timeframes, ['5m', '240m', '1d'])
        self.assertEqual(namespace.kill_keep_signal_frequency_max, 0.34)

    def test_shortt_discovery_dataset_builds_finite_targets_and_weights(self):
        df = make_ashare_frame(rows=240, freq='D')
        dataset = AshareFinancialDataset(
            df,
            window_size=20,
            forecast_horizon=10,
            timeframe_label='1d',
            dataset_profile='shortT_discovery_v1',
        )

        self.assertTrue(dataset.use_continuation_flag)
        self.assertEqual(dataset.targets.shape[1], 2)
        self.assertEqual(dataset.aux_targets.shape[1], 2)
        self.assertEqual(dataset.event_flags.shape[1], len(dataset.event_flag_names))
        self.assertGreaterEqual(dataset.event_flags.shape[1], 7)
        self.assertTrue(np.isfinite(dataset.targets.detach().cpu().numpy()).all())
        self.assertTrue(np.isfinite(dataset.aux_targets.detach().cpu().numpy()).all())
        self.assertTrue(np.isfinite(dataset.sample_weights.detach().cpu().numpy()).all())
        self.assertGreater(float(dataset.sample_weights.mean()), 0.0)

        sample = dataset[0]
        self.assertEqual(len(sample), 6)
        for value in sample:
            self.assertTrue(torch.is_tensor(value))

    def test_shortt_discovery_guarded_dataset_builds_thresholds_and_reuses_them_across_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / '000001_1d.csv'
            make_ashare_frame(rows=420, freq='D').to_csv(csv_path, index=False)

            datasets, metadata = create_ashare_dataset_splits(
                file_path=str(csv_path),
                window_size=20,
                horizon=10,
                train_end='2023-08-31',
                val_end='2023-11-30',
                test_start='2023-12-01',
                fast_full=False,
                return_metadata=True,
                dataset_profile='shortT_discovery_guarded_v1',
            )

            train_ds = datasets['train']
            val_ds = datasets['val']
            test_ds = datasets['test']

            self.assertIsNotNone(train_ds.profile_thresholds)
            self.assertEqual(train_ds.profile_thresholds, val_ds.profile_thresholds)
            self.assertEqual(train_ds.profile_thresholds, test_ds.profile_thresholds)
            self.assertEqual(metadata['profile_thresholds'], train_ds.profile_thresholds)
            self.assertTrue(np.isfinite(train_ds.targets.detach().cpu().numpy()).all())
            self.assertTrue(np.isfinite(val_ds.targets.detach().cpu().numpy()).all())
            self.assertTrue(np.isfinite(test_ds.targets.detach().cpu().numpy()).all())
            self.assertTrue(np.isfinite(train_ds.sample_weights.detach().cpu().numpy()).all())
            self.assertGreater(float(train_ds.sample_weights.mean()), 0.0)

    def test_create_market_datasets_uses_runtime_cache_without_changing_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / '000001_1d.csv'
            make_ashare_frame(rows=240, freq='D').to_csv(csv_path, index=False)

            args = SimpleNamespace(
                market='ashare',
                save_dir=str(tmp_path / 'weights'),
                dataset_cache_dir=None,
                disable_dataset_cache=False,
                window_size=20,
                horizon=4,
                train_end='2023-05-31',
                val_end='2023-07-31',
                test_start='2023-08-01',
                fast_full=False,
                dataset_profile='iterA2',
                split_scheme='time',
                split_label=None,
                horizon_search_spec=None,
                horizon_family_mode='legacy',
                config_fingerprint='runtime-cache-test',
            )
            record = {
                'path': str(csv_path),
                'asset_code': '000001',
                'timeframe': '1d',
            }

            train_ds_1, val_ds_1, test_ds_1, meta_1 = create_market_datasets(record, args, global_horizon_grid=None)
            cache_path = Path(build_runtime_dataset_cache_path(record, args, global_horizon_grid=None))
            self.assertTrue(cache_path.exists())

            train_ds_2, val_ds_2, test_ds_2, meta_2 = create_market_datasets(record, args, global_horizon_grid=None)

            self.assertEqual(meta_1['split_rows'], meta_2['split_rows'])
            self.assertEqual(len(train_ds_1), len(train_ds_2))
            self.assertEqual(len(val_ds_1), len(val_ds_2))
            self.assertEqual(len(test_ds_1), len(test_ds_2))

            sample_1 = train_ds_1[0]
            sample_2 = train_ds_2[0]
            self.assertEqual(len(sample_1), len(sample_2))
            for idx in range(len(sample_1)):
                if isinstance(sample_1[idx], dict):
                    self.assertEqual(sample_1[idx].keys(), sample_2[idx].keys())
                    for key in sample_1[idx]:
                        self.assertTrue(torch.equal(sample_1[idx][key], sample_2[idx][key]), msg=key)
                else:
                    self.assertTrue(torch.equal(sample_1[idx], sample_2[idx]), msg=f'payload_{idx}')


if __name__ == '__main__':
    unittest.main()
