import argparse
import os
import sys

PROJECT_ROOT = r'D:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》'
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'Finance', '02_核心代码', '源代码')
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

from khaos.数据处理.ashare_support import (
    ASHARE_FALLBACK_ASSETS,
    ASHARE_PRIMARY_ASSETS,
    DEFAULT_ASHARE_TIMEFRAMES,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    build_market_coverage_report,
    detect_ifind_sdk,
    fetch_baostock_ashare_history,
    prepare_imported_ashare_data,
    write_coverage_reports,
)


DEFAULT_RESEARCH_RAW_DIR = os.path.join(PROJECT_ROOT, 'Finance', '01_数据中心', '03_研究数据', 'research_raw', 'ashare')
DEFAULT_IMPORT_DIR = os.path.join(DEFAULT_RESEARCH_RAW_DIR, 'imports')
DEFAULT_NORMALIZED_DIR = os.path.join(DEFAULT_RESEARCH_RAW_DIR, 'normalized')
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'Finance', '01_数据中心', '03_研究数据', 'research_processed')
DEFAULT_TRAINING_READY_DIR = os.path.join(DEFAULT_DATA_DIR, 'training_ready', 'ashare')
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, 'logs', 'iterA1_ashare')


def main():
    parser = argparse.ArgumentParser(description='Prepare A-share training_ready data for iterA1.')
    parser.add_argument('--import_dir', type=str, default=DEFAULT_IMPORT_DIR)
    parser.add_argument('--normalized_dir', type=str, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--training_ready_dir', type=str, default=DEFAULT_TRAINING_READY_DIR)
    parser.add_argument('--report_dir', type=str, default=DEFAULT_REPORT_DIR)
    parser.add_argument('--assets', type=str, default=','.join(ASHARE_PRIMARY_ASSETS))
    parser.add_argument('--fallback_assets', type=str, default=','.join(ASHARE_FALLBACK_ASSETS))
    parser.add_argument('--timeframes', type=str, default=','.join(DEFAULT_ASHARE_TIMEFRAMES))
    parser.add_argument('--train_end', type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument('--val_end', type=str, default=DEFAULT_VAL_END)
    parser.add_argument('--test_start', type=str, default=DEFAULT_TEST_START)
    args = parser.parse_args()

    sdk_info = detect_ifind_sdk()
    print('=== iFinD SDK Detection ===')
    print(sdk_info)
    if sdk_info['available']:
        print('检测到 iFinD SDK，但当前脚本默认优先走本地导出/导入标准化流程。')
    else:
        print(f'未检测到 iFinD SDK，将先检查本地导入目录，再按需自动拉取公开 A 股历史数据: {args.import_dir}')

    os.makedirs(args.import_dir, exist_ok=True)
    os.makedirs(args.normalized_dir, exist_ok=True)
    os.makedirs(args.training_ready_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    local_csv_count = sum(
        1
        for root, _, files in os.walk(args.import_dir)
        for file_name in files
        if file_name.lower().endswith('.csv')
    )
    if local_csv_count == 0 and not sdk_info['available']:
        print('=== Public Fallback: BaoStock ===')
        fetch_summary = fetch_baostock_ashare_history(
            asset_codes=ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS,
            output_dir=args.import_dir,
            timeframes=args.timeframes,
            daily_start='2018-01-01',
            minute_start='2021-01-01',
            end_date='2026-04-01',
            overwrite=False,
            pause_seconds=0.15,
        )
        print(fetch_summary)

    prep_summary = prepare_imported_ashare_data(
        import_dir=args.import_dir,
        normalized_dir=args.normalized_dir,
        training_ready_dir=args.training_ready_dir,
        assets=args.assets,
        target_timeframes=args.timeframes,
    )
    print('=== Import Summary ===')
    print(prep_summary)

    coverage_report = build_market_coverage_report(
        data_dir=args.data_dir,
        market='ashare',
        primary_assets=args.assets,
        fallback_assets=args.fallback_assets,
        timeframes=args.timeframes,
        train_end=args.train_end,
        val_end=args.val_end,
        test_start=args.test_start,
        training_subdir='ashare',
    )
    report_paths = write_coverage_reports(args.report_dir, 'iterA1_ashare_coverage', coverage_report)
    print('=== Coverage Report ===')
    print(report_paths)

    if not coverage_report['asset_resolution']['sufficient']:
        print('=== Coverage insufficient, retrying missing assets via BaoStock ===')
        missing_assets = sorted({
            item['asset_code']
            for item in coverage_report['missing_combinations']
            if item['asset_code'] in (ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS)
        })
        if missing_assets:
            print('missing_assets', missing_assets)
            fetch_summary = fetch_baostock_ashare_history(
                asset_codes=missing_assets,
                output_dir=args.import_dir,
                timeframes=args.timeframes,
                daily_start='2018-01-01',
                minute_start='2021-01-01',
                end_date='2026-04-01',
                overwrite=False,
                pause_seconds=0.15,
            )
            print(fetch_summary)
            prep_summary = prepare_imported_ashare_data(
                import_dir=args.import_dir,
                normalized_dir=args.normalized_dir,
                training_ready_dir=args.training_ready_dir,
                assets=args.assets,
                target_timeframes=args.timeframes,
            )
            print('=== Import Summary (Retry) ===')
            print(prep_summary)
            coverage_report = build_market_coverage_report(
                data_dir=args.data_dir,
                market='ashare',
                primary_assets=args.assets,
                fallback_assets=args.fallback_assets,
                timeframes=args.timeframes,
                train_end=args.train_end,
                val_end=args.val_end,
                test_start=args.test_start,
                training_subdir='ashare',
            )
            report_paths = write_coverage_reports(args.report_dir, 'iterA1_ashare_coverage', coverage_report)
            print('=== Coverage Report (Retry) ===')
            print(report_paths)

        if not coverage_report['asset_resolution']['sufficient']:
            raise SystemExit(
                'A-share coverage is insufficient for formal iterA1 training. '
                f"See {report_paths['md_path']}"
            )

    print('A-share training_ready data is ready for iterA1.')


if __name__ == '__main__':
    main()
