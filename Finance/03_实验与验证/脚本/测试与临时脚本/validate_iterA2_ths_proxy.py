from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
PROJECT_SRC = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from khaos.同花顺公式.ths_validation import validate_ths_core


FORMULA_PATH = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码' / 'khaos' / '同花顺公式' / 'KHAOS_THS_CORE.txt'
DATA_GLOB_DIR = PROJECT_ROOT / 'Finance' / '01_数据中心' / '03_研究数据' / 'research_processed' / 'training_ready' / 'ashare'


def main():
    summary = validate_ths_core(
        formula_path=FORMULA_PATH,
        data_glob_dir=DATA_GLOB_DIR,
        sample_count=3,
        fragment_len=240,
        seed=42,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
