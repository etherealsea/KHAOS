from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINANCE_ROOT = PROJECT_ROOT / "Finance"


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def find_ashare_dataset_path() -> Path:
    candidates = sorted(
        path
        for path in FINANCE_ROOT.rglob("ashare_dataset.py")
        if "khaos" in path.parts
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one ashare_dataset.py, got {candidates}")
    return candidates[0]


def find_training_ready_dir() -> Path:
    candidates = [
        path
        for path in sorted(FINANCE_ROOT.rglob("training_ready"))
        if path.is_dir() and path.parent.name == "research_processed"
    ]
    for path in candidates:
        if len(list(path.glob("*.csv"))) == 40:
            return path
    raise FileNotFoundError(f"unable to find research_processed/training_ready with 40 csv files: {candidates}")


ASHARE_DATASET_PATH = find_ashare_dataset_path()
SOURCE_DIR = ASHARE_DATASET_PATH.parents[2]
TRAINING_READY_DIR = find_training_ready_dir()
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


DATASET_MODULE = load_module(
    "khaos_iter12_recent_split_test_module",
    ASHARE_DATASET_PATH,
)

ROLLING_RECENT_SPLITS = DATASET_MODULE.ROLLING_RECENT_SPLITS
build_rolling_split_payloads = DATASET_MODULE._build_rolling_split_payloads
load_ashare_data = DATASET_MODULE.load_ashare_data


EXPECTED_SPLITS = {
    "fold_1": {
        "train_end": "2020-12-31",
        "val_start": "2021-01-01",
        "val_end": "2021-06-30",
        "test_start": "2021-07-01",
        "test_end": "2021-12-31",
    },
    "fold_2": {
        "train_end": "2021-06-30",
        "val_start": "2021-07-01",
        "val_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-06-30",
    },
    "fold_3": {
        "train_end": "2021-12-31",
        "val_start": "2022-01-01",
        "val_end": "2022-06-30",
        "test_start": "2022-07-01",
        "test_end": "2022-12-31",
    },
    "fold_4": {
        "train_end": "2022-06-30",
        "val_start": "2022-07-01",
        "val_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-06-30",
    },
    "final_holdout": {
        "train_end": "2022-12-31",
        "val_start": "2023-01-01",
        "val_end": "2023-06-30",
        "test_start": "2023-07-01",
        "test_end": "2023-12-31",
    },
}


class Iter12RecentSplitCoverageTests(unittest.TestCase):
    def test_rolling_recent_split_calendar_matches_local_data_repair_plan(self):
        self.assertEqual(ROLLING_RECENT_SPLITS, EXPECTED_SPLITS)

    def test_training_ready_root_files_have_non_empty_recent_train_val_test(self):
        csv_files = sorted(TRAINING_READY_DIR.glob("*.csv"))

        self.assertEqual(len(csv_files), 40)

        horizon_profile = {"candidate_horizons": [10]}
        for csv_path in csv_files:
            df = load_ashare_data(str(csv_path))
            for split_label in EXPECTED_SPLITS:
                payloads, resolved_config = build_rolling_split_payloads(
                    df=df,
                    split_label=split_label,
                    window_size=24,
                    horizon_profile=horizon_profile,
                )

                with self.subTest(file=csv_path.name, split=split_label):
                    self.assertEqual(resolved_config, EXPECTED_SPLITS[split_label])
                    for split_name in ("train", "val", "test"):
                        self.assertIn(split_name, payloads)
                        payload = payloads[split_name]
                        self.assertIsNotNone(payload)
                        self.assertGreater(len(payload["df"]), 0)
                        self.assertGreaterEqual(payload["start_index"], 0)
                        self.assertLess(payload["start_index"], len(payload["df"]))


if __name__ == "__main__":
    unittest.main()
