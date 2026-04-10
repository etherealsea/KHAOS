from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from khaos.数据处理.ashare_dataset import AshareFinancialDataset  # noqa: E402
from khaos.模型训练.train import (  # noqa: E402
    build_runtime_dataset_cache_path,
    create_market_datasets,
)


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
            dataset_profile='shortT_precision_v1',
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
        self.assertTrue(hasattr(dataset, 'global_horizon_grid_tensor'))

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
