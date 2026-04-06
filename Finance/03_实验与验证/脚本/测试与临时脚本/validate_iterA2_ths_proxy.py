from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from khaos.同花顺公式.ths_core_proxy import (
    DEFAULT_THS_CORE_PARAMS,
    compute_ths_core_frame,
    load_formula_constants,
)


FORMULA_PATH = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码' / 'khaos' / '同花顺公式' / 'KHAOS_THS_CORE.txt'
DATA_GLOB_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_processed' / 'training_ready' / 'ashare'


def load_random_fragments(sample_count: int = 3, fragment_len: int = 240, seed: int = 42):
    rng = random.Random(seed)
    csv_files = sorted(DATA_GLOB_DIR.glob('*.csv'))
    chosen = rng.sample(csv_files, min(sample_count, len(csv_files)))
    fragments = []
    for path in chosen:
        df = pd.read_csv(path)
        if len(df) <= fragment_len:
            fragments.append((str(path), df.copy()))
            continue
        start = rng.randint(0, len(df) - fragment_len)
        fragments.append((str(path), df.iloc[start:start + fragment_len].copy()))
    return fragments


def main():
    formula_constants = load_formula_constants(FORMULA_PATH)
    expected = {
        'N': float(DEFAULT_THS_CORE_PARAMS.n),
        'BK_EVT_TH': DEFAULT_THS_CORE_PARAMS.bk_evt_th,
        'RV_EVT_TH': DEFAULT_THS_CORE_PARAMS.rv_evt_th,
        'RES_NODE': DEFAULT_THS_CORE_PARAMS.res_node,
        'ENT_BULL': DEFAULT_THS_CORE_PARAMS.ent_bull,
        'ENT_BEAR': DEFAULT_THS_CORE_PARAMS.ent_bear,
        'MLE_GATE': DEFAULT_THS_CORE_PARAMS.mle_gate,
        'H_TREND': DEFAULT_THS_CORE_PARAMS.h_trend,
        'DIR_GAP': DEFAULT_THS_CORE_PARAMS.dir_gap,
    }
    mismatches = {
        key: {'formula': formula_constants.get(key), 'python': value}
        for key, value in expected.items()
        if abs(float(formula_constants.get(key, np.nan)) - float(value)) > 1e-9
    }

    fragment_checks = []
    for path, fragment in load_random_fragments():
        proxy_a = compute_ths_core_frame(fragment, normalize_input=True)
        proxy_b = compute_ths_core_frame(fragment.copy(), normalize_input=True)
        fragment_checks.append({
            'path': path,
            'rows': int(len(fragment)),
            'phase_consistent': bool(np.array_equal(proxy_a['PHASE'].to_numpy(), proxy_b['PHASE'].to_numpy())),
        })

    summary = {
        'formula_path': str(FORMULA_PATH),
        'formula_constants': formula_constants,
        'expected_constants': expected,
        'mismatches': mismatches,
        'fragment_checks': fragment_checks,
        'all_passed': (not mismatches) and all(item['phase_consistent'] for item in fragment_checks),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
