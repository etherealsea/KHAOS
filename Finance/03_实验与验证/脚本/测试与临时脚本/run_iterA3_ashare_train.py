from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from khaos.数据处理.ashare_support import (
    ASHARE_FALLBACK_ASSETS,
    ASHARE_PRIMARY_ASSETS,
    DEFAULT_ASHARE_TIMEFRAMES,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    build_market_coverage_report,
    fetch_baostock_ashare_history,
    prepare_imported_ashare_data,
    write_coverage_reports,
)
from khaos.模型训练.train import train


DATA_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_processed'
RAW_IMPORT_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_raw' / 'ashare' / 'imports'
RAW_NORMALIZED_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_raw' / 'ashare' / 'normalized'
TRAINING_READY_DIR = DATA_DIR / 'training_ready' / 'ashare'
WEIGHT_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '模型权重备份' / 'iterA3_ashare'
SMOKE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '模型权重备份' / 'iterA3_ashare_smoke'
LOG_DIR = PROJECT_ROOT / 'logs' / 'iterA3_ashare'
LOG_PATH = LOG_DIR / 'iterA3_ashare_train.log'
FETCH_END_DATE = '2026-04-03'


def prepare_and_validate():
    RAW_IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_READY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    local_csv_count = sum(
        1
        for root, _, files in os.walk(RAW_IMPORT_DIR)
        for file_name in files
        if file_name.lower().endswith('.csv')
    )
    if local_csv_count == 0:
        fetch_baostock_ashare_history(
            asset_codes=ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS,
            output_dir=str(RAW_IMPORT_DIR),
            timeframes=DEFAULT_ASHARE_TIMEFRAMES,
            daily_start='2018-01-01',
            minute_start='2021-01-01',
            end_date=FETCH_END_DATE,
            overwrite=False,
            pause_seconds=0.15,
        )

    prepare_imported_ashare_data(
        import_dir=str(RAW_IMPORT_DIR),
        normalized_dir=str(RAW_NORMALIZED_DIR),
        training_ready_dir=str(TRAINING_READY_DIR),
        assets=ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS,
        target_timeframes=DEFAULT_ASHARE_TIMEFRAMES,
    )
    coverage_report = build_market_coverage_report(
        data_dir=str(DATA_DIR),
        market='ashare',
        primary_assets=ASHARE_PRIMARY_ASSETS,
        fallback_assets=ASHARE_FALLBACK_ASSETS,
        timeframes=DEFAULT_ASHARE_TIMEFRAMES,
        train_end=DEFAULT_TRAIN_END,
        val_end=DEFAULT_VAL_END,
        test_start=DEFAULT_TEST_START,
        training_subdir='ashare',
    )
    report_paths = write_coverage_reports(str(LOG_DIR), 'iterA3_ashare_coverage', coverage_report)

    if not coverage_report['asset_resolution']['sufficient']:
        missing_assets = sorted({
            item['asset_code']
            for item in coverage_report['missing_combinations']
            if item['asset_code'] in (ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS)
        })
        if missing_assets:
            fetch_baostock_ashare_history(
                asset_codes=missing_assets,
                output_dir=str(RAW_IMPORT_DIR),
                timeframes=DEFAULT_ASHARE_TIMEFRAMES,
                daily_start='2018-01-01',
                minute_start='2021-01-01',
                end_date=FETCH_END_DATE,
                overwrite=False,
                pause_seconds=0.15,
            )
            prepare_imported_ashare_data(
                import_dir=str(RAW_IMPORT_DIR),
                normalized_dir=str(RAW_NORMALIZED_DIR),
                training_ready_dir=str(TRAINING_READY_DIR),
                assets=ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS,
                target_timeframes=DEFAULT_ASHARE_TIMEFRAMES,
            )
            coverage_report = build_market_coverage_report(
                data_dir=str(DATA_DIR),
                market='ashare',
                primary_assets=ASHARE_PRIMARY_ASSETS,
                fallback_assets=ASHARE_FALLBACK_ASSETS,
                timeframes=DEFAULT_ASHARE_TIMEFRAMES,
                train_end=DEFAULT_TRAIN_END,
                val_end=DEFAULT_VAL_END,
                test_start=DEFAULT_TEST_START,
                training_subdir='ashare',
            )
            report_paths = write_coverage_reports(str(LOG_DIR), 'iterA3_ashare_coverage', coverage_report)

    if not coverage_report['asset_resolution']['sufficient']:
        raise RuntimeError(f'A-share coverage is insufficient for iterA3 formal training. See {report_paths["md_path"]}')

    selected_assets = coverage_report['asset_resolution']['selected_assets'][:len(ASHARE_PRIMARY_ASSETS)]
    return selected_assets, report_paths


def build_namespace(
    save_dir: Path,
    assets,
    epochs: int,
    batch_size: int,
    fast_full: bool,
    save_prefix: str,
    per_timeframe_train_cap,
    max_files=None,
    resume: bool = False,
):
    return argparse.Namespace(
        data_dir=str(DATA_DIR),
        save_dir=str(save_dir),
        market='ashare',
        training_subdir='ashare',
        assets=assets,
        timeframes=DEFAULT_ASHARE_TIMEFRAMES,
        split_mode='time',
        train_end=DEFAULT_TRAIN_END,
        val_end=DEFAULT_VAL_END,
        test_start=DEFAULT_TEST_START,
        epochs=epochs,
        batch_size=batch_size,
        lr=8e-4,
        window_size=24,
        horizon=10,
        hidden_dim=80,
        layers=4,
        grid_size=12,
        arch_version='iterA3_multiscale',
        dataset_profile='iterA3',
        seed=42,
        deterministic=True,
        test_mode=False,
        fast_full=fast_full,
        early_stop_patience=8,
        early_stop_min_delta=0.001,
        grad_clip=0.8,
        weight_decay=1e-4,
        resume=resume,
        resume_path=None,
        best_name=f'{save_prefix}_best.pth',
        final_name=f'{save_prefix}_final.pth',
        resume_name=f'{save_prefix}_resume.pth',
        per_timeframe_train_cap=per_timeframe_train_cap,
        max_files=max_files,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--formal-only', action='store_true')
    args = parser.parse_args()
    if args.smoke_only and args.formal_only:
        raise ValueError('`--smoke-only` 与 `--formal-only` 不能同时使用。')

    selected_assets, report_paths = prepare_and_validate()
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    smoke_assets = selected_assets[:2]
    smoke_args = build_namespace(
        save_dir=SMOKE_DIR,
        assets=smoke_assets,
        epochs=1,
        batch_size=64,
        fast_full=True,
        save_prefix='iterA3_ashare_smoke',
        per_timeframe_train_cap={'15m': 1024, '60m': 1024, '1d': 512},
        max_files=len(smoke_assets) * len(DEFAULT_ASHARE_TIMEFRAMES),
        resume=False,
    )
    formal_args = build_namespace(
        save_dir=WEIGHT_DIR,
        assets=selected_assets,
        epochs=32,
        batch_size=256,
        fast_full=False,
        save_prefix='iterA3_ashare',
        per_timeframe_train_cap={'15m': 4096, '60m': 4096, '1d': 4096},
        max_files=None,
        resume=True,
    )

    log_mode = 'a' if LOG_PATH.exists() else 'w'
    with LOG_PATH.open(log_mode, encoding='utf-8', buffering=1) as log_file:
        log_file.write('\niterA3_runner_start\n')
        log_file.write(f'coverage_report={report_paths}\n')
        log_file.write(f'selected_assets={selected_assets}\n')
        log_file.write(f'mode={{"smoke_only": {args.smoke_only}, "formal_only": {args.formal_only}}}\n')
        log_file.flush()
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            if not args.formal_only:
                print('=== Smoke Check ===')
                print({'assets': smoke_assets, 'save_dir': str(SMOKE_DIR)})
                train(smoke_args)
            if not args.smoke_only:
                print('=== Formal Train ===')
                print({'assets': selected_assets, 'save_dir': str(WEIGHT_DIR)})
                train(formal_args)
        log_file.write('iterA3_runner_end\n')
        log_file.flush()

    print(f'iterA3 training finished. See log: {LOG_PATH}')
    print(f'Coverage report: {report_paths}')
    print(f'Selected assets: {selected_assets}')


if __name__ == '__main__':
    main()
