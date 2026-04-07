from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from khaos.同花顺公式.ths_core_proxy import (
    DEFAULT_THS_CORE_PARAMS,
    compute_ths_core_frame,
    load_formula_constants,
    load_ths_core_params,
)


def load_random_fragments(
    data_glob_dir: str | Path,
    sample_count: int = 3,
    fragment_len: int = 240,
    seed: int = 42,
):
    """从 training_ready 目录抽取若干段数据，用于 THS proxy 一致性自检。"""
    rng = random.Random(seed)
    data_dir = Path(data_glob_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        return []
    chosen = rng.sample(csv_files, min(sample_count, len(csv_files)))
    fragments = []
    for path in chosen:
        df = pd.read_csv(path)
        if len(df) <= fragment_len:
            fragments.append((str(path), df.copy()))
            continue
        start = rng.randint(0, len(df) - fragment_len)
        fragments.append((str(path), df.iloc[start : start + fragment_len].copy()))
    return fragments


def validate_ths_core(
    *,
    formula_path: str | Path,
    data_glob_dir: str | Path,
    params_path: str | Path | None = None,
    sample_count: int = 3,
    fragment_len: int = 240,
    seed: int = 42,
) -> dict[str, Any]:
    """
    THS CORE 校验（用于“终端可映射性”硬约束）：
    1) 关键常量（N/BK_EVT_TH/...）在公式文件与参数包之间一致；
    2) 对随机片段重复计算，PHASE 输出稳定（避免非确定性漂移）。
    """
    formula_path = Path(formula_path)
    if params_path is None:
        candidate = Path(__file__).with_name("params.json")
        params_path = candidate if candidate.exists() else None

    params = DEFAULT_THS_CORE_PARAMS
    resolved_params_path = None
    if params_path is not None and Path(params_path).exists():
        resolved_params_path = str(params_path)
        params = load_ths_core_params(params_path)

    formula_constants = load_formula_constants(formula_path)
    expected = {
        "N": float(params.n),
        "BK_EVT_TH": float(params.bk_evt_th),
        "RV_EVT_TH": float(params.rv_evt_th),
        "RES_NODE": float(params.res_node),
        "ENT_BULL": float(params.ent_bull),
        "ENT_BEAR": float(params.ent_bear),
        "MLE_GATE": float(params.mle_gate),
        "H_TREND": float(params.h_trend),
        "DIR_GAP": float(params.dir_gap),
    }

    mismatches = {
        key: {"formula": formula_constants.get(key), "params": value}
        for key, value in expected.items()
        if abs(float(formula_constants.get(key, np.nan)) - float(value)) > 1e-9
    }

    fragments = load_random_fragments(
        data_glob_dir=data_glob_dir,
        sample_count=sample_count,
        fragment_len=fragment_len,
        seed=seed,
    )

    fragment_checks = []
    for path, fragment in fragments:
        proxy_a = compute_ths_core_frame(fragment, params=params, normalize_input=True)
        proxy_b = compute_ths_core_frame(fragment.copy(), params=params, normalize_input=True)
        fragment_checks.append(
            {
                "path": path,
                "rows": int(len(fragment)),
                "phase_consistent": bool(
                    np.array_equal(proxy_a["PHASE"].to_numpy(), proxy_b["PHASE"].to_numpy())
                ),
            }
        )

    data_available = bool(fragments)
    all_passed = (not mismatches) and data_available and all(
        item["phase_consistent"] for item in fragment_checks
    )

    summary = {
        "formula_path": str(formula_path),
        "params_path": resolved_params_path,
        "data_glob_dir": str(data_glob_dir),
        "data_available": data_available,
        "formula_constants": formula_constants,
        "expected_constants": expected,
        "mismatches": mismatches,
        "fragment_checks": fragment_checks,
        "all_passed": all_passed,
    }
    if not data_available:
        summary["data_error"] = "no_csv_files_found_under_data_glob_dir"
    return summary
