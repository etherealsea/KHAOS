import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


TIMEFRAME_TO_SUFFIX = {
    "5m": "5m",
    "15m": "15m",
    "60m": "1h",
    "240m": "4h",
    "1d": "1d",
}

DEFAULT_ASSETS = [
    "BTCUSD",
    "ETHUSD",
    "ESUSD",
    "SPXUSD",
    "UDXUSD",
    "WTIUSD",
    "XAUUSD",
]

DEFAULT_TIMEFRAMES = ["5m", "15m", "60m", "240m"]


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_file_list(source_dir, assets, timeframes):
    files = []
    for asset in assets:
        for timeframe in timeframes:
            suffix = TIMEFRAME_TO_SUFFIX[timeframe]
            file_path = source_dir / f"{asset}_{suffix}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"缺少训练文件: {file_path}")
            files.append(file_path)
    return files


def clean_target_dir(target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_path in target_dir.glob("*.csv"):
        file_path.unlink()


def write_manifest(workspace_root, payload):
    manifest_path = workspace_root / "workspace_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="为 iter15 构建干净的训练工作区")
    parser.add_argument(
        "--source_dir",
        type=str,
        default=r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\training_ready",
    )
    parser.add_argument(
        "--workspace_root",
        type=str,
        default=r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\iter15_training_workspace",
    )
    parser.add_argument("--assets", type=str, default=",".join(DEFAULT_ASSETS))
    parser.add_argument("--timeframes", type=str, default=",".join(DEFAULT_TIMEFRAMES))
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    training_ready_dir = workspace_root / "training_ready"
    dataset_cache_dir = workspace_root / "dataset_cache" / "iter15_event_first_v1"

    assets = parse_csv_list(args.assets)
    timeframes = parse_csv_list(args.timeframes)
    files = build_file_list(source_dir, assets, timeframes)

    clean_target_dir(training_ready_dir)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for file_path in files:
        target_path = training_ready_dir / file_path.name
        shutil.copy2(file_path, target_path)
        copied.append(target_path.name)

    manifest = {
        "workspace_root": str(workspace_root),
        "source_dir": str(source_dir),
        "training_ready_dir": str(training_ready_dir),
        "dataset_cache_dir": str(dataset_cache_dir),
        "assets": assets,
        "timeframes": timeframes,
        "file_count": len(copied),
        "files": copied,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_manifest(workspace_root, manifest)

    print(f"[iter15-workspace] workspace_root={workspace_root}")
    print(f"[iter15-workspace] training_ready_dir={training_ready_dir}")
    print(f"[iter15-workspace] dataset_cache_dir={dataset_cache_dir}")
    print(f"[iter15-workspace] copied_files={len(copied)}")
    for name in copied:
        print(name)


if __name__ == "__main__":
    main()
