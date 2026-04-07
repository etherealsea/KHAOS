from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import analyze_iterA2_ashare_results as base_analyzer


PROMOTION_THRESHOLDS = {
    'overall_model_composite': 0.4187,
    'calibrated_ths_test_objective': 0.4448,
    'timeframe_composite': {
        '60m': 0.4032,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='iterA5_ashare')
    parser.add_argument('--compare-version', type=str, default='iterA3_ashare')
    parser.add_argument('--no-compare', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compare_version = None if args.no_compare else args.compare_version
    baseline_result = (
        base_analyzer.evaluate_version(compare_version, device)
        if compare_version and compare_version != args.version
        else None
    )
    current_result = base_analyzer.evaluate_version(args.version, device)
    output_paths = base_analyzer.write_version_outputs(
        current_result,
        baseline_result=baseline_result,
        baseline_ths_source='calibrated',
        promotion_thresholds=PROMOTION_THRESHOLDS,
    )

    print('=== Analysis Complete ===')
    print({'version': args.version, 'outputs': output_paths, 'promotion_thresholds': PROMOTION_THRESHOLDS})


if __name__ == '__main__':
    main()
