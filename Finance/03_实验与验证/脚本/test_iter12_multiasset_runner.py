from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
RUNNER_PATH = PROJECT_ROOT / 'scripts' / 'run_iter12_multiasset_closed_loop.py'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAIN_MODULE = load_module(
    'khaos_iter12_runtime_test_module',
    SOURCE_DIR / 'khaos' / '模型训练' / 'train.py',
)
RUNNER_MODULE = load_module(
    'iter12_runner_test_module',
    RUNNER_PATH,
)

create_market_datasets = TRAIN_MODULE.create_market_datasets
expand_runtime_records = TRAIN_MODULE.expand_runtime_records


class Iter12MultiassetRunnerTests(unittest.TestCase):
    def test_resolve_project_dataset_cache_dir_defaults_to_shared_project_location(self):
        data_dir = Path('D:/repo/Finance/01_数据中心/03_研究数据/research_processed')

        resolved = RUNNER_MODULE.resolve_project_dataset_cache_dir(data_dir, None)

        self.assertEqual(
            resolved,
            data_dir / 'dataset_cache' / 'iter12_guarded_recent_v1',
        )

    def test_resolve_project_dataset_cache_dir_allows_explicit_override(self):
        data_dir = Path('D:/repo/Finance/01_数据中心/03_研究数据/research_processed')
        override = 'D:/repo/custom/cache'

        resolved = RUNNER_MODULE.resolve_project_dataset_cache_dir(data_dir, override)

        self.assertEqual(resolved, Path(override))

    def test_expand_runtime_records_supports_legacy_multiasset_recent_splits(self):
        args = SimpleNamespace(
            market='legacy_multiasset',
            split_scheme='rolling_recent_v1',
            split_labels='fold_3,fold_4',
        )
        records = [
            {'path': 'demo.csv', 'asset_code': 'BTCUSD', 'timeframe': '60m'},
        ]

        expanded = expand_runtime_records(records, args, include_final_holdout=False)

        self.assertEqual(len(expanded), 2)
        self.assertEqual([item['split_label'] for item in expanded], ['fold_3', 'fold_4'])

    def test_create_market_datasets_uses_recent_split_path_for_legacy_multiasset(self):
        original_create_ashare_dataset_splits = TRAIN_MODULE.create_ashare_dataset_splits
        original_create_rolling_datasets = TRAIN_MODULE.create_rolling_datasets
        calls = []

        def fake_create_ashare_dataset_splits(**kwargs):
            calls.append(kwargs)
            return (
                {'train': 'train_ds', 'val': 'val_ds', 'test': 'test_ds'},
                {
                    'asset_code': 'BTCUSD',
                    'timeframe': '60m',
                    'split_label': kwargs.get('split_label'),
                },
            )

        def fake_create_rolling_datasets(*args, **kwargs):
            raise AssertionError('legacy_multiasset rolling_recent_v1 should use ashare-style split creation')

        TRAIN_MODULE.create_ashare_dataset_splits = fake_create_ashare_dataset_splits
        TRAIN_MODULE.create_rolling_datasets = fake_create_rolling_datasets
        try:
            args = SimpleNamespace(
                market='legacy_multiasset',
                split_scheme='rolling_recent_v1',
                disable_dataset_cache=True,
                window_size=24,
                horizon=10,
                train_end=None,
                val_end=None,
                test_start=None,
                fast_full=False,
                dataset_profile='shortT_discovery_guarded_v2',
                split_label=None,
                horizon_search_spec=None,
                horizon_family_mode='legacy',
            )
            train_ds, val_ds, test_ds, dataset_meta = create_market_datasets(
                {
                    'path': 'demo.csv',
                    'asset_code': 'BTCUSD',
                    'timeframe': '60m',
                    'split_label': 'fold_3',
                },
                args,
                global_horizon_grid=None,
            )
        finally:
            TRAIN_MODULE.create_ashare_dataset_splits = original_create_ashare_dataset_splits
            TRAIN_MODULE.create_rolling_datasets = original_create_rolling_datasets

        self.assertEqual(train_ds, 'train_ds')
        self.assertEqual(val_ds, 'val_ds')
        self.assertEqual(test_ds, 'test_ds')
        self.assertEqual(dataset_meta['split_label'], 'fold_3')
        self.assertEqual(calls[0]['split_label'], 'fold_3')
        self.assertEqual(calls[0]['dataset_profile'], 'shortT_discovery_guarded_v2')

    def test_formal_preset_matches_iter12_doc_aligned_defaults(self):
        preset = RUNNER_MODULE.FORMAL_PRESET

        self.assertEqual(preset['dataset_profile'], 'shortT_discovery_guarded_v2')
        self.assertEqual(preset['loss_profile'], 'shortT_discovery_guarded_v2')
        self.assertEqual(preset['score_profile'], 'iter12_guarded_precision_first')
        self.assertEqual(preset['split_scheme'], 'rolling_recent_v1')
        self.assertEqual(preset['split_labels'], 'fold_1,fold_2,fold_3,fold_4')
        self.assertEqual(preset['early_stop_patience'], 8)
        self.assertEqual(preset['per_timeframe_train_cap']['60m'], 6144)
        self.assertEqual(preset['kill_keep_timeframe_60m_composite_min'], 0.0)
        self.assertEqual(preset['resume_mode'], 'auto')

    def test_smoke_preset_is_tuned_for_fast_validation(self):
        preset = RUNNER_MODULE.SMOKE_PRESET

        self.assertEqual(preset['epochs_total'], 2)
        self.assertEqual(preset['batch_size'], 256)
        self.assertEqual(preset['per_timeframe_train_cap']['60m'], 1536)
        self.assertEqual(preset['per_timeframe_train_cap']['15m'], 1024)
        self.assertEqual(preset['num_workers'], 4)
        self.assertEqual(preset['prefetch_factor'], 4)
        self.assertFalse(preset['deterministic'])

    def test_should_resume_chunk_auto_supports_same_run_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / 'khaos_kan_resume.pth'

            self.assertFalse(RUNNER_MODULE.should_resume_chunk('auto', 2, 2, resume_path))

            resume_path.write_text('resume-state', encoding='utf-8')
            self.assertTrue(RUNNER_MODULE.should_resume_chunk('auto', 2, 2, resume_path))
            self.assertTrue(RUNNER_MODULE.should_resume_chunk('auto', 20, 2, resume_path))
            self.assertFalse(RUNNER_MODULE.should_resume_chunk('never', 20, 2, resume_path))


if __name__ == '__main__':
    unittest.main()
