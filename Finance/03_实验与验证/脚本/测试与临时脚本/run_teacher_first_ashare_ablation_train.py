from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[4]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
for candidate in (PROJECT_SRC, SCRIPT_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from khaos.数据处理.ashare_support import (
    ASHARE_FALLBACK_ASSETS,
    ASHARE_PRIMARY_ASSETS,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    build_market_coverage_report,
    fetch_public_ashare_history,
    prepare_imported_ashare_data,
    write_coverage_reports,
)
from khaos.模型训练.train import train

import analyze_iterA2_ashare_results as base_analyzer


DATA_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_processed'
RAW_IMPORT_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_raw' / 'ashare' / 'imports'
RAW_NORMALIZED_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_raw' / 'ashare' / 'normalized'
TRAINING_READY_DIR = DATA_DIR / 'training_ready' / 'ashare'
WEIGHT_ROOT = PROJECT_ROOT / 'Finance' / '02_核心代码' / '模型权重备份' / 'teacher_first_ashare'
SMOKE_ROOT = PROJECT_ROOT / 'Finance' / '02_核心代码' / '模型权重备份' / 'teacher_first_ashare_smoke'
LOG_DIR = PROJECT_ROOT / 'logs' / 'teacher_first_ashare'
PIPELINE_DIR = LOG_DIR / '_pipeline'

ITERA5_WEIGHT_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '模型权重备份' / 'iterA5_ashare'
ITERA5_LOG_DIR = PROJECT_ROOT / 'logs' / 'iterA5_ashare'

FETCH_END_DATE = '2026-04-03'
RAW_FETCH_TIMEFRAMES = ['5m', '15m', '60m', '1d']
CORE_TIMEFRAMES = ['15m', '60m', '240m', '1d']
WITH_5M_TIMEFRAMES = ['5m', '15m', '60m', '240m', '1d']

BASE_CAPS = {'15m': 4096, '60m': 5120, '240m': 3072, '1d': 4096}
CAPS_WITH_5M = {'5m': 2048, '15m': 4096, '60m': 6144, '240m': 3072, '1d': 4096}
SHORT_T_CAPS = {'5m': 4096, '15m': 6144, '60m': 6144, '240m': 2048, '1d': 1536}
SHORT_T_BALANCED_CAPS = {'5m': 3072, '15m': 6144, '60m': 6144, '240m': 3072, '1d': 2048}

SMOKE_CAPS_BASE = {'15m': 1024, '60m': 1024, '240m': 512, '1d': 512}
SMOKE_CAPS_WITH_5M = {'5m': 512, '15m': 1024, '60m': 1024, '240m': 512, '1d': 512}
SMOKE_CAPS_SHORT_T = {'5m': 1024, '15m': 1536, '60m': 1536, '240m': 384, '1d': 256}
SMOKE_CAPS_SHORT_T_BALANCED = {'5m': 768, '15m': 1536, '60m': 1536, '240m': 512, '1d': 384}

SMOKE_GATE_THRESHOLDS = {
    'composite_score': 0.34,
    'direction_macro_f1': 0.70,
    'public_below_directional_violation_rate': 0.25,
    'public_below_directional_violation': 0.05,
}

PROMOTION_THRESHOLDS = {
    'overall_model_composite': 0.4187,
    'calibrated_ths_test_objective': 0.4448,
    'timeframe_composite': {
        '60m': 0.4032,
    },
}

FORMAL_SUCCESS_THRESHOLDS = {
    'overall_model_composite': 0.4187,
    'calibrated_ths_test_objective': 0.4448,
    'timeframe_60m_composite': 0.4032,
    'direction_macro_f1': 0.90,
    'public_below_directional_violation_rate': 0.12,
    'baseline_composite_margin': 0.005,
}

MAINLINE_SMOKE_ORDER = ['iterA4_refresh', 'shortT_balanced_v1', 'shortT_balanced_v2']
MAINLINE_CANDIDATES = ['shortT_balanced_v1', 'shortT_balanced_v2']

EXPERIMENT_SPECS = {
    'iterA4_baseline': {
        'timeframes': CORE_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'iterA4',
        'loss_profile': 'iterA4',
        'score_profile': 'iterA4_event_focus',
        'epochs': 36,
        'early_stop_patience': 8,
        'per_timeframe_train_cap': BASE_CAPS,
    },
    'iterA4_refresh': {
        'timeframes': CORE_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'iterA4',
        'loss_profile': 'iterA4',
        'score_profile': 'iterA4_event_focus',
        'constraint_profile': 'teacher_feasible_v1',
        'epochs': 36,
        'early_stop_patience': 8,
        'per_timeframe_train_cap': BASE_CAPS,
    },
    'B1_5m_only': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'iterA4',
        'loss_profile': 'iterA4',
        'score_profile': 'iterA4_event_focus',
        'epochs': 16,
        'early_stop_patience': 4,
        'per_timeframe_train_cap': CAPS_WITH_5M,
    },
    'B2_arch_only': {
        'timeframes': CORE_TIMEFRAMES,
        'arch_version': 'iterA5_multiscale',
        'dataset_profile': 'iterA4',
        'loss_profile': 'iterA4',
        'score_profile': 'iterA4_event_focus',
        'epochs': 16,
        'early_stop_patience': 4,
        'per_timeframe_train_cap': BASE_CAPS,
    },
    'B3_loss_only': {
        'timeframes': CORE_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'iterA4',
        'loss_profile': 'iterA5',
        'score_profile': 'iterA4_event_focus',
        'epochs': 16,
        'early_stop_patience': 4,
        'per_timeframe_train_cap': BASE_CAPS,
    },
    'shortT_breakout_v1': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'shortT_v1',
        'loss_profile': 'shortT_breakout_v1',
        'score_profile': 'short_t_breakout_focus',
        'score_timeframes': ['5m', '15m', '60m'],
        'aux_timeframes': ['240m', '1d'],
        'constraint_profile': 'teacher_feasible_v1',
        'epochs': 18,
        'early_stop_patience': 5,
        'per_timeframe_train_cap': SHORT_T_CAPS,
        'smoke_per_timeframe_train_cap': SMOKE_CAPS_SHORT_T,
    },
    'shortT_balanced_v1': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'shortT_balanced_v1',
        'loss_profile': 'shortT_balanced_v1',
        'score_profile': 'short_t_balanced_focus',
        'score_timeframes': ['15m', '60m', '240m', '1d'],
        'aux_timeframes': ['5m'],
        'constraint_profile': 'teacher_feasible_v1',
        'epochs': 20,
        'early_stop_patience': 5,
        'per_timeframe_train_cap': SHORT_T_BALANCED_CAPS,
        'smoke_per_timeframe_train_cap': SMOKE_CAPS_SHORT_T_BALANCED,
    },
    'shortT_balanced_v2': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'shortT_balanced_v2',
        'loss_profile': 'shortT_balanced_v2',
        'score_profile': 'short_t_balanced_guarded',
        'score_timeframes': ['15m', '60m', '240m', '1d'],
        'aux_timeframes': ['5m'],
        'constraint_profile': 'teacher_feasible_v1',
        'epochs': 20,
        'early_stop_patience': 5,
        'per_timeframe_train_cap': SHORT_T_BALANCED_CAPS,
        'smoke_per_timeframe_train_cap': SMOKE_CAPS_SHORT_T_BALANCED,
    },
    'shortT_precision_v1': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'shortT_precision_v1',
        'loss_profile': 'shortT_precision_v1',
        'score_profile': 'short_t_precision_focus',
        'score_timeframes': ['15m', '60m', '240m', '1d'],
        'aux_timeframes': ['5m'],
        'constraint_profile': 'teacher_feasible_precision_v1',
        'epochs': 16,
        'early_stop_patience': 4,
        'per_timeframe_train_cap': SHORT_T_BALANCED_CAPS,
        'smoke_per_timeframe_train_cap': SMOKE_CAPS_SHORT_T_BALANCED,
        'warm_start_from': 'shortT_balanced_v1',
    },
    'horizon_single_precision_v1': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'shortT_precision_v1',
        'loss_profile': 'horizon_precision_v1',
        'score_profile': 'recent_precision_v1',
        'score_timeframes': ['15m', '60m', '240m', '1d'],
        'aux_timeframes': ['5m'],
        'constraint_profile': 'teacher_feasible_precision_v1',
        'split_scheme': 'rolling_recent_v1',
        'horizon_family_mode': 'single_cycle',
        'horizon_search_spec': {
            'min_horizon': 20,
            'max_horizon': 120,
            'max_fraction': 0.12,
            'saturation_ratio': 0.01,
            'saturation_patience': 8,
            'saturation_ema': 5,
        },
        'epochs': 18,
        'early_stop_patience': 5,
        'per_timeframe_train_cap': SHORT_T_BALANCED_CAPS,
        'smoke_per_timeframe_train_cap': SMOKE_CAPS_SHORT_T_BALANCED,
        'warm_start_from': 'shortT_balanced_v1',
        'breakout_precision_floor': 0.50,
        'reversion_precision_floor': 0.48,
        'public_violation_cap': 0.20,
        'signal_frequency_cap_ratio': 0.70,
        'retired_baseline': True,
        'enforce_horizon_family_guard': True,
    },
    'horizon_adaptive_precision_v1': {
        'timeframes': WITH_5M_TIMEFRAMES,
        'arch_version': 'iterA4_multiscale',
        'dataset_profile': 'shortT_precision_v1',
        'loss_profile': 'horizon_precision_v1',
        'score_profile': 'recent_precision_v1',
        'score_timeframes': ['15m', '60m', '240m', '1d'],
        'aux_timeframes': ['5m'],
        'constraint_profile': 'teacher_feasible_precision_v1',
        'split_scheme': 'rolling_recent_v1',
        'horizon_family_mode': 'adaptive_resonance',
        'horizon_search_spec': {
            'min_horizon': 20,
            'max_horizon': 120,
            'max_fraction': 0.12,
            'saturation_ratio': 0.01,
            'saturation_patience': 8,
            'saturation_ema': 5,
        },
        'epochs': 18,
        'early_stop_patience': 5,
        'per_timeframe_train_cap': SHORT_T_BALANCED_CAPS,
        'smoke_per_timeframe_train_cap': SMOKE_CAPS_SHORT_T_BALANCED,
        'warm_start_from': 'shortT_balanced_v1',
        'breakout_precision_floor': 0.50,
        'reversion_precision_floor': 0.48,
        'public_violation_cap': 0.20,
        'signal_frequency_cap_ratio': 0.70,
        'kill_keep_review_epoch': 3,
        'kill_keep_public_violation_rate_max': 0.95,
        'kill_keep_signal_frequency_max': 0.42,
        'kill_keep_timeframe_60m_composite_min': 0.22,
        'kill_keep_horizon_entropy_min': 0.05,
        'kill_keep_horizon_entropy_timeframes': ['15m', '60m'],
    },
}


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return value
    return value


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl_records(path: Path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_latest_epoch_snapshot(save_dir: Path):
    epoch_metrics_path = save_dir / 'epoch_metrics.jsonl'
    per_timeframe_metrics_path = save_dir / 'per_timeframe_metrics.jsonl'
    epoch_records = read_jsonl_records(epoch_metrics_path)
    if not epoch_records:
        raise FileNotFoundError(f'No epoch metrics found: {epoch_metrics_path}')
    latest_epoch = int(epoch_records[-1]['epoch'])
    per_timeframe_records = [
        item for item in read_jsonl_records(per_timeframe_metrics_path)
        if int(item.get('epoch', -1)) == latest_epoch
    ]
    return {
        'epoch_metrics_path': epoch_metrics_path,
        'per_timeframe_metrics_path': per_timeframe_metrics_path,
        'latest_epoch_metrics': epoch_records[-1],
        'latest_per_timeframe_metrics': {item['timeframe']: item for item in per_timeframe_records},
    }


def load_best_checkpoint_metrics(save_dir: Path, experiment_name: str):
    candidates = [
        save_dir / f'{experiment_name}_best.pth',
        save_dir / f'{experiment_name}_best_raw.pth',
        save_dir / f'{experiment_name}_best_gate.pth',
    ]
    best_path = next((path for path in candidates if path.exists()), None)
    if best_path is None:
        raise FileNotFoundError(f'No best checkpoint found for {experiment_name} under {save_dir}')
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    return checkpoint, checkpoint.get('metrics', {})


def copy_if_exists(source: Path, target_dir: Path):
    if not source.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / source.name
    shutil.copy2(source, destination)
    return destination


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
        fetch_public_ashare_history(
            asset_codes=ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS,
            output_dir=str(RAW_IMPORT_DIR),
            timeframes=RAW_FETCH_TIMEFRAMES,
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
        target_timeframes=WITH_5M_TIMEFRAMES,
    )
    coverage_report = build_market_coverage_report(
        data_dir=str(DATA_DIR),
        market='ashare',
        primary_assets=ASHARE_PRIMARY_ASSETS,
        fallback_assets=ASHARE_FALLBACK_ASSETS,
        timeframes=WITH_5M_TIMEFRAMES,
        train_end=DEFAULT_TRAIN_END,
        val_end=DEFAULT_VAL_END,
        test_start=DEFAULT_TEST_START,
        training_subdir='ashare',
    )
    report_paths = write_coverage_reports(str(LOG_DIR), 'teacher_first_ashare_coverage', coverage_report)

    if not coverage_report['asset_resolution']['sufficient']:
        missing_assets = sorted({
            item['asset_code']
            for item in coverage_report['missing_combinations']
            if item['asset_code'] in (ASHARE_PRIMARY_ASSETS + ASHARE_FALLBACK_ASSETS)
        })
        if missing_assets:
            fetch_public_ashare_history(
                asset_codes=missing_assets,
                output_dir=str(RAW_IMPORT_DIR),
                timeframes=RAW_FETCH_TIMEFRAMES,
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
                target_timeframes=WITH_5M_TIMEFRAMES,
            )
            coverage_report = build_market_coverage_report(
                data_dir=str(DATA_DIR),
                market='ashare',
                primary_assets=ASHARE_PRIMARY_ASSETS,
                fallback_assets=ASHARE_FALLBACK_ASSETS,
                timeframes=WITH_5M_TIMEFRAMES,
                train_end=DEFAULT_TRAIN_END,
                val_end=DEFAULT_VAL_END,
                test_start=DEFAULT_TEST_START,
                training_subdir='ashare',
            )
            report_paths = write_coverage_reports(str(LOG_DIR), 'teacher_first_ashare_coverage', coverage_report)

    if not coverage_report['asset_resolution']['sufficient']:
        raise RuntimeError(f'A-share coverage is insufficient for teacher-first ablations. See {report_paths["md_path"]}')

    selected_assets = coverage_report['asset_resolution']['selected_assets'][:len(ASHARE_PRIMARY_ASSETS)]
    return selected_assets, report_paths


def parse_experiments(value: str):
    if not value or value.strip().lower() == 'all':
        return [
            name for name, spec in EXPERIMENT_SPECS.items()
            if not spec.get('retired_baseline', False)
        ]
    experiments = [item.strip() for item in value.split(',') if item.strip()]
    unknown = [item for item in experiments if item not in EXPERIMENT_SPECS]
    if unknown:
        raise ValueError(f'Unknown experiments: {unknown}')
    return experiments


def build_namespace(
    experiment_name: str,
    save_dir: Path,
    assets,
    epochs: int,
    batch_size: int,
    fast_full: bool,
    per_timeframe_train_cap,
    constraint_profile: str,
    resume: bool,
    max_files=None,
):
    spec = EXPERIMENT_SPECS[experiment_name]
    resolved_constraint_profile = spec.get('constraint_profile', constraint_profile)
    resolved_resume = resume
    resolved_resume_path = None
    resolved_warm_start_weights_only = bool(spec.get('warm_start_weights_only', False))
    resolved_warm_start_path = spec.get('warm_start_path')
    warm_start_from = spec.get('warm_start_from')
    resolved_baseline_reference_dir = spec.get('baseline_reference_dir')
    if warm_start_from:
        warm_start_best = WEIGHT_ROOT / warm_start_from / f'{warm_start_from}_best.pth'
        if warm_start_best.exists():
            resolved_warm_start_weights_only = True
            resolved_warm_start_path = str(warm_start_best)
            if resolved_baseline_reference_dir is None:
                resolved_baseline_reference_dir = str(WEIGHT_ROOT / warm_start_from)
    return argparse.Namespace(
        data_dir=str(DATA_DIR),
        save_dir=str(save_dir),
        market='ashare',
        training_subdir='ashare',
        assets=assets,
        timeframes=spec['timeframes'],
        split_mode='time',
        split_scheme=spec.get('split_scheme', 'time'),
        split_labels=spec.get('split_labels'),
        train_end=DEFAULT_TRAIN_END,
        val_end=DEFAULT_VAL_END,
        test_start=DEFAULT_TEST_START,
        epochs=epochs,
        batch_size=batch_size,
        lr=7e-4,
        window_size=24,
        horizon=10,
        horizon_search_spec=json.dumps(spec['horizon_search_spec'], ensure_ascii=False) if isinstance(spec.get('horizon_search_spec'), dict) else spec.get('horizon_search_spec'),
        horizon_family_mode=spec.get('horizon_family_mode', 'legacy'),
        hidden_dim=80,
        layers=4,
        grid_size=12,
        arch_version=spec['arch_version'],
        dataset_profile=spec['dataset_profile'],
        loss_profile=spec['loss_profile'],
        constraint_profile=resolved_constraint_profile,
        score_profile=spec['score_profile'],
        score_timeframes=spec.get('score_timeframes', CORE_TIMEFRAMES),
        aux_timeframes=spec.get('aux_timeframes'),
        seed=42,
        deterministic=True,
        test_mode=False,
        fast_full=fast_full,
        early_stop_patience=spec['early_stop_patience'],
        early_stop_min_delta=0.001,
        grad_clip=0.8,
        weight_decay=1e-4,
        resume=resolved_resume,
        resume_path=resolved_resume_path,
        warm_start_weights_only=resolved_warm_start_weights_only,
        warm_start_path=resolved_warm_start_path,
        config_fingerprint=None,
        baseline_reference_dir=resolved_baseline_reference_dir,
        enforce_horizon_family_guard=bool(spec.get('enforce_horizon_family_guard', False) and not fast_full),
        signal_frequency_cap_ratio=spec.get('signal_frequency_cap_ratio', 0.70),
        signal_frequency_cap=spec.get('signal_frequency_cap'),
        public_violation_cap=spec.get('public_violation_cap', 0.20),
        breakout_precision_floor=spec.get('breakout_precision_floor', 0.0),
        reversion_precision_floor=spec.get('reversion_precision_floor', 0.0),
        best_name=f'{experiment_name}_best.pth',
        best_raw_name=f'{experiment_name}_best_raw.pth',
        best_gate_name=f'{experiment_name}_best_gate.pth',
        final_name=f'{experiment_name}_final.pth',
        resume_name=f'{experiment_name}_resume.pth',
        epoch_metrics_name='epoch_metrics.jsonl',
        per_timeframe_metrics_name='per_timeframe_metrics.jsonl',
        per_fold_metrics_name='per_fold_metrics.jsonl',
        kill_keep_review_epoch=spec.get('kill_keep_review_epoch', 0),
        kill_keep_public_violation_rate_max=spec.get('kill_keep_public_violation_rate_max', 1.0),
        kill_keep_signal_frequency_max=spec.get('kill_keep_signal_frequency_max', 1.0),
        kill_keep_timeframe_60m_composite_min=spec.get('kill_keep_timeframe_60m_composite_min', 0.0),
        kill_keep_horizon_entropy_min=spec.get('kill_keep_horizon_entropy_min', 0.0),
        kill_keep_horizon_entropy_timeframes=spec.get('kill_keep_horizon_entropy_timeframes'),
        promotion_overall_composite_threshold=PROMOTION_THRESHOLDS['overall_model_composite'],
        promotion_timeframe_composite_thresholds=','.join(
            f'{tf}={threshold}'
            for tf, threshold in PROMOTION_THRESHOLDS['timeframe_composite'].items()
        ),
        per_timeframe_train_cap=per_timeframe_train_cap,
        max_files=max_files,
    )


def run_training_phase(
    experiment_name: str,
    phase: str,
    selected_assets,
    report_paths,
    constraint_profile: str = 'default',
    resume: bool = False,
    skip_ths_validation: bool = False,
):
    if phase not in {'smoke', 'formal'}:
        raise ValueError(f'Unsupported phase: {phase}')

    spec = EXPERIMENT_SPECS[experiment_name]
    experiment_log_dir = LOG_DIR / experiment_name
    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_log_dir / f'{experiment_name}.log'

    if phase == 'smoke':
        phase_assets = selected_assets[:2]
        caps = spec.get(
            'smoke_per_timeframe_train_cap',
            SMOKE_CAPS_WITH_5M if '5m' in spec['timeframes'] else SMOKE_CAPS_BASE,
        )
        namespace = build_namespace(
            experiment_name=experiment_name,
            save_dir=SMOKE_ROOT / experiment_name,
            assets=phase_assets,
            epochs=1,
            batch_size=64,
            fast_full=True,
            per_timeframe_train_cap=caps,
            constraint_profile=constraint_profile,
            resume=False,
            max_files=len(phase_assets) * len(spec['timeframes']),
        )
    else:
        phase_assets = selected_assets[:2]
        namespace = build_namespace(
            experiment_name=experiment_name,
            save_dir=WEIGHT_ROOT / experiment_name,
            assets=phase_assets,
            epochs=2,
            batch_size=256,
            fast_full=True,
            per_timeframe_train_cap=spec['per_timeframe_train_cap'],
            constraint_profile=constraint_profile,
            resume=resume,
            max_files=None,
        )

    namespace.save_dir = str(Path(namespace.save_dir))
    Path(namespace.save_dir).mkdir(parents=True, exist_ok=True)

    log_mode = 'a' if log_path.exists() else 'w'
    with log_path.open(log_mode, encoding='utf-8', buffering=1) as log_file:
        log_file.write(f'\nteacher_first_experiment_start={experiment_name}\n')
        log_file.write(f'teacher_first_phase_start={phase}\n')
        log_file.write(f'coverage_report={report_paths}\n')
        log_file.write(f'selected_assets={selected_assets}\n')
        log_file.write(f'spec={spec}\n')
        log_file.write(
            f'phase_mode={{"phase": "{phase}", "constraint_profile": "{constraint_profile}", "resume": {resume}}}\n'
        )
        log_file.flush()
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print('=== Smoke Check ===' if phase == 'smoke' else '=== Formal Train ===')
            print({
                'experiment': experiment_name,
                'phase': phase,
                'assets': phase_assets,
                'save_dir': str(namespace.save_dir),
                'timeframes': spec['timeframes'],
                'score_timeframes': namespace.score_timeframes,
                'aux_timeframes': namespace.aux_timeframes,
            })
            train(namespace)

            ths_validation = None
            if phase == 'formal' and not skip_ths_validation:
                from khaos.同花顺公式.ths_validation import validate_ths_core

                formula_path = (
                    PROJECT_ROOT
                    / 'Finance'
                    / '02_核心代码'
                    / '源代码'
                    / 'khaos'
                    / '同花顺公式'
                    / 'KHAOS_THS_CORE.txt'
                )
                ths_validation = validate_ths_core(
                    formula_path=formula_path,
                    data_glob_dir=TRAINING_READY_DIR,
                    sample_count=3,
                    fragment_len=240,
                    seed=42,
                )
                print({'ths_proxy_validation': ths_validation})
                if not ths_validation.get('all_passed', False):
                    raise RuntimeError('THS proxy validation failed (all_passed=false).')
        log_file.write(f'teacher_first_phase_end={phase}\n')
        log_file.write(f'teacher_first_experiment_end={experiment_name}\n')
        log_file.flush()

    return {
        'experiment': experiment_name,
        'phase': phase,
        'save_dir': Path(namespace.save_dir),
        'log_path': log_path,
        'best_path': Path(namespace.save_dir) / namespace.best_name,
        'best_raw_path': Path(namespace.save_dir) / namespace.best_raw_name,
        'best_gate_path': Path(namespace.save_dir) / namespace.best_gate_name,
        'final_path': Path(namespace.save_dir) / namespace.final_name,
        'resume_path': Path(namespace.save_dir) / namespace.resume_name,
        'ths_proxy_validation': ths_validation,
    }


def evaluate_smoke_gate(experiment_name: str):
    save_dir = SMOKE_ROOT / experiment_name
    snapshot = load_latest_epoch_snapshot(save_dir)
    latest = snapshot['latest_epoch_metrics']
    checks = {
        'composite_score': {
            'actual': float(latest['composite_score']),
            'threshold': SMOKE_GATE_THRESHOLDS['composite_score'],
            'passed': float(latest['composite_score']) >= SMOKE_GATE_THRESHOLDS['composite_score'],
        },
        'direction_macro_f1': {
            'actual': float(latest['direction_macro_f1']),
            'threshold': SMOKE_GATE_THRESHOLDS['direction_macro_f1'],
            'passed': float(latest['direction_macro_f1']) >= SMOKE_GATE_THRESHOLDS['direction_macro_f1'],
        },
        'public_below_directional_violation_rate': {
            'actual': float(latest['public_below_directional_violation_rate']),
            'threshold': SMOKE_GATE_THRESHOLDS['public_below_directional_violation_rate'],
            'passed': float(latest['public_below_directional_violation_rate']) <= SMOKE_GATE_THRESHOLDS['public_below_directional_violation_rate'],
        },
        'public_below_directional_violation': {
            'actual': float(latest['public_below_directional_violation']),
            'threshold': SMOKE_GATE_THRESHOLDS['public_below_directional_violation'],
            'passed': float(latest['public_below_directional_violation']) <= SMOKE_GATE_THRESHOLDS['public_below_directional_violation'],
        },
        'epoch_metrics_exists': {
            'actual': snapshot['epoch_metrics_path'].exists(),
            'threshold': True,
            'passed': snapshot['epoch_metrics_path'].exists(),
        },
        'per_timeframe_metrics_exists': {
            'actual': snapshot['per_timeframe_metrics_path'].exists(),
            'threshold': True,
            'passed': snapshot['per_timeframe_metrics_path'].exists(),
        },
    }
    hard_gate_keys = (
        'composite_score',
        'direction_macro_f1',
        'epoch_metrics_exists',
        'per_timeframe_metrics_exists',
    )
    passed = all(checks[key]['passed'] for key in hard_gate_keys) and (
        checks['public_below_directional_violation_rate']['passed'] or
        checks['public_below_directional_violation']['passed']
    )
    return {
        'passed': passed,
        'strict_rate_passed': all(
            checks[key]['passed'] for key in hard_gate_keys + ('public_below_directional_violation_rate',)
        ),
        'save_dir': save_dir,
        'checks': checks,
        'latest_epoch_metrics': latest,
        'latest_per_timeframe_metrics': snapshot['latest_per_timeframe_metrics'],
        'epoch_metrics_path': snapshot['epoch_metrics_path'],
        'per_timeframe_metrics_path': snapshot['per_timeframe_metrics_path'],
    }


def snapshot_itera5_reference():
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    snapshot_root = LOG_DIR / '_reference_snapshots' / f'iterA5_quick_switch_{timestamp}'
    snapshot_root.mkdir(parents=True, exist_ok=True)

    copied = {}
    for source in (
        ITERA5_WEIGHT_DIR / 'iterA5_ashare_best.pth',
        ITERA5_WEIGHT_DIR / 'iterA5_ashare_resume.pth',
        ITERA5_WEIGHT_DIR / 'epoch_metrics.jsonl',
        ITERA5_WEIGHT_DIR / 'per_timeframe_metrics.jsonl',
        ITERA5_LOG_DIR / 'iterA5_ashare_train.log',
        ITERA5_LOG_DIR / 'iterA5_ashare_coverage.json',
        ITERA5_LOG_DIR / 'iterA5_ashare_coverage.md',
        ITERA5_LOG_DIR / 'iterA5_ashare_pause_status.md',
    ):
        destination = copy_if_exists(source, snapshot_root)
        if destination is not None:
            copied[source.name] = destination

    latest_epoch_metrics = None
    latest_per_timeframe_metrics = None
    if (ITERA5_WEIGHT_DIR / 'epoch_metrics.jsonl').exists():
        snapshot = load_latest_epoch_snapshot(ITERA5_WEIGHT_DIR)
        latest_epoch_metrics = snapshot['latest_epoch_metrics']
        latest_per_timeframe_metrics = snapshot['latest_per_timeframe_metrics']

    best_metrics = {}
    best_path = ITERA5_WEIGHT_DIR / 'iterA5_ashare_best.pth'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        best_metrics = checkpoint.get('metrics', {})

    manifest = {
        'snapshot_time': datetime.now().isoformat(),
        'source_weight_dir': ITERA5_WEIGHT_DIR,
        'source_log_dir': ITERA5_LOG_DIR,
        'copied_files': copied,
        'latest_epoch_metrics': latest_epoch_metrics,
        'latest_per_timeframe_metrics': latest_per_timeframe_metrics,
        'best_checkpoint_metrics': best_metrics,
        'note': 'iterA5 quick-switch reference snapshot captured before teacher-first balanced pipeline',
    }
    write_json(snapshot_root / 'snapshot_manifest.json', manifest)
    return manifest


def evaluate_reference_version(version_name: str, device):
    return base_analyzer.evaluate_version(version_name, device)


def evaluate_teacher_first_version(version_name: str, device, baseline_result=None, promotion_thresholds=None):
    original_weight_root = base_analyzer.WEIGHTS_ROOT
    original_log_root = base_analyzer.LOG_ROOT
    try:
        base_analyzer.WEIGHTS_ROOT = str(WEIGHT_ROOT)
        base_analyzer.LOG_ROOT = str(LOG_DIR)
        result = base_analyzer.evaluate_version(version_name, device)
        outputs = base_analyzer.write_version_outputs(
            result,
            baseline_result=baseline_result,
            baseline_ths_source='calibrated',
            promotion_thresholds=promotion_thresholds,
        )
        return result, outputs
    finally:
        base_analyzer.WEIGHTS_ROOT = original_weight_root
        base_analyzer.LOG_ROOT = original_log_root


def assess_formal_success(experiment_name: str, analysis_result, best_checkpoint_metrics, baseline_result):
    overall_composite = float(analysis_result['overall_model']['composite_score'])
    calibrated_ths_objective = float(analysis_result['calibrated_ths']['test_objective'])
    timeframe_60m_composite = analysis_result['model_by_timeframe'].get('60m', {}).get('composite_score')
    direction_macro_f1 = float(analysis_result['overall_model']['direction_eval']['macro_f1'])
    reversion_f1 = float(analysis_result['overall_model']['reversion_eval']['f1'])
    baseline_composite = float(baseline_result['overall_model']['composite_score'])
    baseline_reversion_f1 = float(baseline_result['overall_model']['reversion_eval']['f1'])
    public_violation_rate = best_checkpoint_metrics.get('constraint_rates', {}).get('public_below_directional_violation_rate')

    checks = {
        'overall_model_composite': {
            'actual': overall_composite,
            'threshold': FORMAL_SUCCESS_THRESHOLDS['overall_model_composite'],
            'passed': overall_composite >= FORMAL_SUCCESS_THRESHOLDS['overall_model_composite'],
        },
        'calibrated_ths_test_objective': {
            'actual': calibrated_ths_objective,
            'threshold': FORMAL_SUCCESS_THRESHOLDS['calibrated_ths_test_objective'],
            'passed': calibrated_ths_objective >= FORMAL_SUCCESS_THRESHOLDS['calibrated_ths_test_objective'],
        },
        'timeframe_60m_composite': {
            'actual': timeframe_60m_composite,
            'threshold': FORMAL_SUCCESS_THRESHOLDS['timeframe_60m_composite'],
            'passed': timeframe_60m_composite is not None and timeframe_60m_composite >= FORMAL_SUCCESS_THRESHOLDS['timeframe_60m_composite'],
        },
        'reversion_f1_vs_baseline': {
            'actual': reversion_f1,
            'threshold': baseline_reversion_f1,
            'passed': reversion_f1 >= baseline_reversion_f1,
        },
        'direction_macro_f1': {
            'actual': direction_macro_f1,
            'threshold': FORMAL_SUCCESS_THRESHOLDS['direction_macro_f1'],
            'passed': direction_macro_f1 >= FORMAL_SUCCESS_THRESHOLDS['direction_macro_f1'],
        },
        'public_below_directional_violation_rate': {
            'actual': public_violation_rate,
            'threshold': FORMAL_SUCCESS_THRESHOLDS['public_below_directional_violation_rate'],
            'passed': public_violation_rate is not None and public_violation_rate <= FORMAL_SUCCESS_THRESHOLDS['public_below_directional_violation_rate'],
        },
        'beats_iterA4_refresh_by_margin': {
            'actual': overall_composite,
            'threshold': baseline_composite + FORMAL_SUCCESS_THRESHOLDS['baseline_composite_margin'],
            'passed': overall_composite >= baseline_composite + FORMAL_SUCCESS_THRESHOLDS['baseline_composite_margin'],
        },
    }
    return {
        'experiment': experiment_name,
        'passed': all(item['passed'] for item in checks.values()),
        'checks': checks,
    }


def run_mainline_pipeline(args):
    selected_assets, report_paths = prepare_and_validate()
    WEIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        'pipeline': 'mainline_20260407',
        'started_at': datetime.now().isoformat(),
        'selected_assets': selected_assets,
        'coverage_report': report_paths,
        'smoke': {},
        'formal': {},
        'status': 'running',
    }

    if not args.skip_itera5_freeze:
        summary['iterA5_reference_snapshot'] = snapshot_itera5_reference()

    for experiment_name in MAINLINE_SMOKE_ORDER:
        phase_result = run_training_phase(
            experiment_name=experiment_name,
            phase='smoke',
            selected_assets=selected_assets,
            report_paths=report_paths,
            constraint_profile=args.constraint_profile,
            resume=False,
            skip_ths_validation=args.skip_ths_validation,
        )
        gate_result = evaluate_smoke_gate(experiment_name)
        summary['smoke'][experiment_name] = {
            'run': phase_result,
            'gate': gate_result,
        }
        write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)

    if not summary['smoke']['iterA4_refresh']['gate']['passed']:
        summary['status'] = 'aborted_baseline_smoke_failed'
        write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)
        return summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterA3_result = evaluate_reference_version('iterA3_ashare', device)

    baseline_run = run_training_phase(
        experiment_name='iterA4_refresh',
        phase='formal',
        selected_assets=selected_assets,
        report_paths=report_paths,
        constraint_profile=args.constraint_profile,
        resume=args.resume,
        skip_ths_validation=args.skip_ths_validation,
    )
    baseline_analysis, baseline_outputs = evaluate_teacher_first_version(
        'iterA4_refresh',
        device,
        baseline_result=iterA3_result,
        promotion_thresholds=PROMOTION_THRESHOLDS,
    )
    _, baseline_best_metrics = load_best_checkpoint_metrics(WEIGHT_ROOT / 'iterA4_refresh', 'iterA4_refresh')
    summary['formal']['iterA4_refresh'] = {
        'run': baseline_run,
        'analysis_outputs': baseline_outputs,
        'overall_model': baseline_analysis['overall_model'],
        'calibrated_ths': baseline_analysis['calibrated_ths'],
        'best_checkpoint_metrics': baseline_best_metrics,
    }
    write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)

    for experiment_name in MAINLINE_CANDIDATES:
        smoke_gate = summary['smoke'][experiment_name]['gate']
        if not smoke_gate['passed']:
            summary['formal'][experiment_name] = {
                'skipped': True,
                'reason': 'smoke_gate_failed',
                'smoke_gate': smoke_gate,
            }
            write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)
            continue

        candidate_run = run_training_phase(
            experiment_name=experiment_name,
            phase='formal',
            selected_assets=selected_assets,
            report_paths=report_paths,
            constraint_profile=args.constraint_profile,
            resume=args.resume,
            skip_ths_validation=args.skip_ths_validation,
        )
        candidate_analysis, candidate_outputs = evaluate_teacher_first_version(
            experiment_name,
            device,
            baseline_result=baseline_analysis,
            promotion_thresholds=PROMOTION_THRESHOLDS,
        )
        _, candidate_best_metrics = load_best_checkpoint_metrics(WEIGHT_ROOT / experiment_name, experiment_name)
        success = assess_formal_success(
            experiment_name=experiment_name,
            analysis_result=candidate_analysis,
            best_checkpoint_metrics=candidate_best_metrics,
            baseline_result=baseline_analysis,
        )
        ths_proxy_passed = bool(candidate_run.get('ths_proxy_validation', {}).get('all_passed', True))
        success['checks']['ths_proxy_validation'] = {
            'actual': ths_proxy_passed,
            'threshold': True,
            'passed': ths_proxy_passed,
        }
        success['passed'] = bool(success['passed'] and ths_proxy_passed)
        summary['formal'][experiment_name] = {
            'run': candidate_run,
            'analysis_outputs': candidate_outputs,
            'overall_model': candidate_analysis['overall_model'],
            'calibrated_ths': candidate_analysis['calibrated_ths'],
            'best_checkpoint_metrics': candidate_best_metrics,
            'success_gate': success,
        }
        if success['passed']:
            summary['status'] = f'success_{experiment_name}'
            summary['completed_at'] = datetime.now().isoformat()
            write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)
            return summary

        write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)

    summary['status'] = 'failed_after_shortT_balanced_v2'
    summary['completed_at'] = datetime.now().isoformat()
    write_json(PIPELINE_DIR / 'mainline_20260407_summary.json', summary)
    return summary


def run_manual_mode(args):
    experiments = parse_experiments(args.experiments)
    selected_assets, report_paths = prepare_and_validate()
    WEIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment_name in experiments:
        if not args.formal_only:
            results.append(
                run_training_phase(
                    experiment_name=experiment_name,
                    phase='smoke',
                    selected_assets=selected_assets,
                    report_paths=report_paths,
                    constraint_profile=args.constraint_profile,
                    resume=False,
                    skip_ths_validation=args.skip_ths_validation,
                )
            )
        if not args.smoke_only:
            results.append(
                run_training_phase(
                    experiment_name=experiment_name,
                    phase='formal',
                    selected_assets=selected_assets,
                    report_paths=report_paths,
                    constraint_profile=args.constraint_profile,
                    resume=args.resume,
                    skip_ths_validation=args.skip_ths_validation,
                )
            )

    print('teacher-first ablation training finished.')
    print({'coverage_report': report_paths, 'results': results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=str, default='all')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--formal-only', action='store_true')
    parser.add_argument('--constraint-profile', type=str, default='default')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pipeline', choices=['manual', 'mainline_20260407'], default='manual')
    parser.add_argument('--skip-itera5-freeze', action='store_true')
    parser.add_argument('--skip-ths-validation', action='store_true')
    args = parser.parse_args()
    if args.smoke_only and args.formal_only:
        raise ValueError('`--smoke-only` and `--formal-only` cannot be used together.')

    if args.pipeline == 'mainline_20260407':
        summary = run_mainline_pipeline(args)
        print('teacher-first mainline pipeline finished.')
        print({'summary_path': str(PIPELINE_DIR / 'mainline_20260407_summary.json'), 'status': summary['status']})
        return

    run_manual_mode(args)


if __name__ == '__main__':
    main()
