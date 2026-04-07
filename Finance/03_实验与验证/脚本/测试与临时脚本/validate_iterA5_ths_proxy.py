from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import validate_iterA2_ths_proxy as base_validator


if __name__ == '__main__':
    base_validator.main()
