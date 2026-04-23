import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_ASSETS = "BTCUSD,ETHUSD,ESUSD,SPXUSD,EURUSD,UDXUSD,WTIUSD,XAUUSD"
DEFAULT_TIMEFRAMES = "5m,15m,60m,240m,1d"
DEFAULT_SCORE_TIMEFRAMES = "5m,15m,60m,240m,1d"
DEFAULT_AUX_TIMEFRAMES = ""
DEFAULT_SPLIT_SCHEME = "rolling_recent_v1"
DEFAULT_SPLIT_LABELS = "fold_1,fold_2,fold_3,fold_4"
ITER12_DISCOVERY_CAPS = {
    "5m": 2048,
    "15m": 4096,
    "60m": 6144,
    "240m": 4096,
    "1d": 2048,
}
ITER12_SMOKE_CAPS = {
    "5m": 512,
    "15m": 1024,
    "60m": 1536,
    "240m": 1024,
    "1d": 512,
}
SMOKE_PRESET = {
    "run_label": "smoke",
    "epochs_total": 5,
    "chunk_size": 2,
    "batch_size": 256,
    "arch_version": "iterA4",
    "dataset_profile": "iter14_ev_regression",
    "loss_profile": "iter14_ev_regression",
    "constraint_profile": "iter14_ev_regression",
    "score_profile": "iter14_precision_first",
    "score_timeframes": DEFAULT_SCORE_TIMEFRAMES,
    "aux_timeframes": DEFAULT_AUX_TIMEFRAMES,
    "split_scheme": DEFAULT_SPLIT_SCHEME,
    "split_labels": DEFAULT_SPLIT_LABELS,
    "training_subdir": None,
    "assets": DEFAULT_ASSETS,
    "timeframes": DEFAULT_TIMEFRAMES,
    "lr": 7e-4,
    "window_size": 60,
    "horizon": 10,
    "hidden_dim": 80,
    "layers": 4,
    "grid_size": 12,
    "early_stop_patience": 8,
    "early_stop_min_delta": 0.001,
    "grad_clip": 0.8,
    "per_timeframe_train_cap": ITER12_SMOKE_CAPS,
    "num_workers": 4,
    "prefetch_factor": 4,
    "deterministic": False,
    "kill_keep_signal_frequency_max": 0.40,
    "kill_keep_public_violation_rate_max": 0.14,
    "kill_keep_timeframe_60m_composite_min": 0.0,
    "kill_keep_breakout_signal_space_min": 0.95,
    "kill_keep_reversion_signal_space_min": 0.70,
    "public_violation_cap": 0.10,
    "signal_frequency_cap_ratio": 0.70,
    "breakout_precision_floor": 0.0,
    "reversion_precision_floor": 0.0,
    "resume_mode": "auto",
    "gate_mode": "soft_annealed",
    "gate_floor_breakout": 0.10,
    "gate_floor_reversion": 0.15,
    "gate_anneal_fraction": 0.40,
    "horizon_search_spec": "6,10,14,20",
}
FORMAL_PRESET = {
    "run_label": "formal",
    "epochs_total": 20,
    "chunk_size": 20,
    "batch_size": 256,
    "arch_version": "iterA4",
    "dataset_profile": "iter14_ev_regression",
    "loss_profile": "iter14_ev_regression",
    "constraint_profile": "iter14_ev_regression",
    "score_profile": "iter14_precision_first",
    "score_timeframes": DEFAULT_SCORE_TIMEFRAMES,
    "aux_timeframes": DEFAULT_AUX_TIMEFRAMES,
    "split_scheme": DEFAULT_SPLIT_SCHEME,
    "split_labels": DEFAULT_SPLIT_LABELS,
    "training_subdir": None,
    "assets": DEFAULT_ASSETS,
    "timeframes": DEFAULT_TIMEFRAMES,
    "lr": 7e-4,
    "window_size": 60,
    "horizon": 10,
    "hidden_dim": 80,
    "layers": 4,
    "grid_size": 12,
    "early_stop_patience": 8,
    "early_stop_min_delta": 0.001,
    "grad_clip": 0.8,
    "per_timeframe_train_cap": ITER12_DISCOVERY_CAPS,
    "num_workers": 4,
    "prefetch_factor": 4,
    "deterministic": False,
    "kill_keep_signal_frequency_max": 0.40,
    "kill_keep_public_violation_rate_max": 0.14,
    "kill_keep_timeframe_60m_composite_min": 0.0,
    "kill_keep_breakout_signal_space_min": 0.95,
    "kill_keep_reversion_signal_space_min": 0.70,
    "public_violation_cap": 0.05,
    "signal_frequency_cap_ratio": 0.70,
    "breakout_precision_floor": 0.0,
    "reversion_precision_floor": 0.0,
    "resume_mode": "auto",
    "gate_mode": "soft_annealed",
    "gate_floor_breakout": 0.10,
    "gate_floor_reversion": 0.15,
    "gate_anneal_fraction": 0.40,
    "horizon_search_spec": "6,10,14,20",
}
PHASE_PRESETS = {
    "smoke": SMOKE_PRESET,
    "formal": FORMAL_PRESET,
}


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            target = out_dir / Path(name).name
            with archive.open(name) as src, target.open("wb") as dst:
                dst.write(src.read())


def run_train(train_py: Path, argv: list[str]) -> None:
    cmd = [sys.executable, "-u", str(train_py)] + argv
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def locate_train_py(project_root: Path) -> Path:
    finance_root = project_root / "Finance"
    candidates = sorted(
        path
        for path in finance_root.rglob("train.py")
        if "khaos" in path.parts and path.parent.name != "khaos_kan"
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one active train.py under {finance_root}, got {candidates}")
    return candidates[0]


def resolve_project_paths() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    train_py = locate_train_py(project_root)
    report_py = project_root / "scripts" / "generate_iter10_multiasset_report.py"
    if not report_py.exists():
        raise FileNotFoundError(f"missing report generator: {report_py}")
    return project_root, train_py, report_py


def resolve_training_ready_dir(data_dir: Path, training_subdir: Optional[str]) -> Path:
    training_ready_dir = data_dir / "training_ready"
    if training_subdir:
        training_ready_dir = training_ready_dir / training_subdir
    return training_ready_dir


def resolve_project_dataset_cache_dir(data_dir: Path, dataset_cache_dir: Optional[str]) -> Path:
    if dataset_cache_dir:
        return Path(dataset_cache_dir).resolve()
    return (data_dir / "dataset_cache" / "iter12_guarded_recent_v1").resolve()


def format_timeframe_caps(value: Optional[str | dict[str, int]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    ordered = [f"{timeframe}={int(cap)}" for timeframe, cap in value.items()]
    return ",".join(ordered)


def coalesce(value, fallback):
    return fallback if value is None else value


def resolve_effective_config(args: argparse.Namespace) -> dict:
    preset = PHASE_PRESETS[args.phase]
    effective = {
        "phase": args.phase,
        "run_label": preset["run_label"],
        "data_zip": args.data_zip,
        "data_dir": args.data_dir,
        "run_root": args.run_root,
        "run_id": args.run_id,
        "assets": coalesce(args.assets, preset["assets"]),
        "timeframes": coalesce(args.timeframes, preset["timeframes"]),
        "epochs_total": coalesce(args.epochs_total, preset["epochs_total"]),
        "chunk_size": coalesce(args.chunk_size, preset["chunk_size"]),
        "batch_size": coalesce(args.batch_size, preset["batch_size"]),
        "arch_version": coalesce(args.arch_version, preset["arch_version"]),
        "dataset_profile": coalesce(args.dataset_profile, preset["dataset_profile"]),
        "loss_profile": coalesce(args.loss_profile, preset["loss_profile"]),
        "constraint_profile": coalesce(args.constraint_profile, preset["constraint_profile"]),
        "score_profile": coalesce(args.score_profile, preset["score_profile"]),
        "score_timeframes": coalesce(args.score_timeframes, preset["score_timeframes"]),
        "aux_timeframes": coalesce(args.aux_timeframes, preset["aux_timeframes"]),
        "split_scheme": coalesce(args.split_scheme, preset["split_scheme"]),
        "split_labels": coalesce(args.split_labels, preset["split_labels"]),
        "training_subdir": coalesce(args.training_subdir, preset["training_subdir"]),
        "lr": coalesce(args.lr, preset["lr"]),
        "window_size": coalesce(args.window_size, preset["window_size"]),
        "horizon": coalesce(args.horizon, preset["horizon"]),
        "hidden_dim": coalesce(args.hidden_dim, preset["hidden_dim"]),
        "layers": coalesce(args.layers, preset["layers"]),
        "grid_size": coalesce(args.grid_size, preset["grid_size"]),
        "early_stop_patience": coalesce(args.early_stop_patience, preset["early_stop_patience"]),
        "early_stop_min_delta": coalesce(args.early_stop_min_delta, preset["early_stop_min_delta"]),
        "grad_clip": coalesce(args.grad_clip, preset["grad_clip"]),
        "per_timeframe_train_cap": format_timeframe_caps(
            coalesce(args.per_timeframe_train_cap, preset["per_timeframe_train_cap"])
        ),
        "num_workers": coalesce(args.num_workers, preset["num_workers"]),
        "prefetch_factor": coalesce(args.prefetch_factor, preset["prefetch_factor"]),
        "deterministic": coalesce(args.deterministic, preset["deterministic"]),
        "kill_keep_signal_frequency_max": coalesce(
            args.kill_keep_signal_frequency_max,
            preset["kill_keep_signal_frequency_max"],
        ),
        "kill_keep_public_violation_rate_max": coalesce(
            args.kill_keep_public_violation_rate_max,
            preset["kill_keep_public_violation_rate_max"],
        ),
        "kill_keep_timeframe_60m_composite_min": coalesce(
            args.kill_keep_timeframe_60m_composite_min,
            preset["kill_keep_timeframe_60m_composite_min"],
        ),
        "kill_keep_breakout_signal_space_min": coalesce(
            args.kill_keep_breakout_signal_space_min,
            preset["kill_keep_breakout_signal_space_min"],
        ),
        "kill_keep_reversion_signal_space_min": coalesce(
            args.kill_keep_reversion_signal_space_min,
            preset["kill_keep_reversion_signal_space_min"],
        ),
        "public_violation_cap": coalesce(args.public_violation_cap, preset["public_violation_cap"]),
        "signal_frequency_cap_ratio": coalesce(
            args.signal_frequency_cap_ratio,
            preset["signal_frequency_cap_ratio"],
        ),
        "breakout_precision_floor": coalesce(
            args.breakout_precision_floor,
            preset["breakout_precision_floor"],
        ),
        "reversion_precision_floor": coalesce(
            args.reversion_precision_floor,
            preset["reversion_precision_floor"],
        ),
        "resume_mode": coalesce(args.resume_mode, preset["resume_mode"]),
        "dataset_cache_dir": args.dataset_cache_dir,
        "skip_dataset_cache_prewarm": bool(args.skip_dataset_cache_prewarm),
        "prewarm_only": bool(args.prewarm_only),
        "fast_full": bool(args.fast_full),
    }
    if effective["epochs_total"] <= 0:
        raise ValueError("epochs_total must be positive")
    if effective["chunk_size"] <= 0:
        raise ValueError("chunk_size must be positive")
    return effective


def build_chunk_ends(epochs_total: int, chunk_size: int) -> list[int]:
    chunk_ends = []
    current = min(chunk_size, epochs_total)
    while current < epochs_total:
        chunk_ends.append(current)
        current = min(current + chunk_size, epochs_total)
    chunk_ends.append(epochs_total)
    return chunk_ends


def add_optional_arg(argv: list[str], name: str, value) -> None:
    if value is None:
        return
    argv.extend([name, str(value)])


def should_resume_chunk(resume_mode: str, chunk_end: int, first_chunk_end: int, resume_path: Path) -> bool:
    mode = str(resume_mode or "auto").strip().lower()
    if mode in {"never", "off", "false"}:
        return False
    if mode in {"always", "force", "true"}:
        return True
    if chunk_end != first_chunk_end:
        return True
    return resume_path.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PHASE_PRESETS.keys()), default="smoke")
    parser.add_argument("--data_zip", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--training_subdir", type=str, default=None)
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--assets", type=str, default=None)
    parser.add_argument("--timeframes", type=str, default=None)
    parser.add_argument("--epochs_total", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--arch_version", type=str, default=None)
    parser.add_argument("--dataset_profile", type=str, default=None)
    parser.add_argument("--loss_profile", type=str, default=None)
    parser.add_argument("--constraint_profile", type=str, default=None)
    parser.add_argument("--score_profile", type=str, default=None)
    parser.add_argument("--score_timeframes", type=str, default=None)
    parser.add_argument("--aux_timeframes", type=str, default=None)
    parser.add_argument("--split_scheme", type=str, default=None)
    parser.add_argument("--split_labels", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--per_timeframe_train_cap", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--kill_keep_signal_frequency_max", type=float, default=None)
    parser.add_argument("--kill_keep_public_violation_rate_max", type=float, default=None)
    parser.add_argument("--kill_keep_timeframe_60m_composite_min", type=float, default=None)
    parser.add_argument("--kill_keep_breakout_signal_space_min", type=float, default=None)
    parser.add_argument("--kill_keep_reversion_signal_space_min", type=float, default=None)
    parser.add_argument("--public_violation_cap", type=float, default=None)
    parser.add_argument("--signal_frequency_cap_ratio", type=float, default=None)
    parser.add_argument("--breakout_precision_floor", type=float, default=None)
    parser.add_argument("--reversion_precision_floor", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true", default=None)
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--resume_mode", choices=["auto", "always", "never"], default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--skip_dataset_cache_prewarm", action="store_true", default=False)
    parser.add_argument("--prewarm_only", action="store_true", default=False)
    parser.add_argument("--fast_full", action="store_true", default=False)
    args = parser.parse_args()

    config = resolve_effective_config(args)
    data_dir = Path(args.data_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_id = args.run_id or datetime.now().strftime(f"%Y%m%d_iter14_multiasset_{config['run_label']}_%H%M%S")
    save_dir = run_root / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = resolve_project_dataset_cache_dir(data_dir, config["dataset_cache_dir"])
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    _, train_py, report_py = resolve_project_paths()
    training_ready_dir = resolve_training_ready_dir(data_dir, config["training_subdir"])
    if args.data_zip:
        extract_zip(Path(args.data_zip).resolve(), training_ready_dir)

    resume_path = save_dir / "khaos_kan_resume.pth"
    chunk_ends = [min(config["chunk_size"], config["epochs_total"])] if config["prewarm_only"] else build_chunk_ends(config["epochs_total"], config["chunk_size"])
    first_chunk_end = chunk_ends[0]
    for chunk_end in chunk_ends:
        argv = [
            "--data_dir",
            str(data_dir),
            "--save_dir",
            str(save_dir),
            "--market",
            "legacy_multiasset",
            "--assets",
            config["assets"],
            "--timeframes",
            config["timeframes"],
            "--arch_version",
            config["arch_version"],
            "--dataset_profile",
            config["dataset_profile"],
            "--loss_profile",
            config["loss_profile"],
            "--constraint_profile",
            config["constraint_profile"],
            "--score_profile",
            config["score_profile"],
            "--split_scheme",
            config["split_scheme"],
            "--epochs",
            str(chunk_end),
            "--batch_size",
            str(config["batch_size"]),
            "--lr",
            str(config["lr"]),
            "--window_size",
            str(config["window_size"]),
            "--horizon",
            str(config["horizon"]),
            "--hidden_dim",
            str(config["hidden_dim"]),
            "--layers",
            str(config["layers"]),
            "--grid_size",
            str(config["grid_size"]),
            "--early_stop_patience",
            str(config["early_stop_patience"]),
            "--early_stop_min_delta",
            str(config["early_stop_min_delta"]),
            "--grad_clip",
            str(config["grad_clip"]),
        ]
        add_optional_arg(argv, "--training_subdir", config["training_subdir"])
        add_optional_arg(argv, "--dataset_cache_dir", str(dataset_cache_dir))
        add_optional_arg(argv, "--score_timeframes", config["score_timeframes"])
        add_optional_arg(argv, "--aux_timeframes", config["aux_timeframes"])
        add_optional_arg(argv, "--split_labels", config["split_labels"])
        add_optional_arg(argv, "--per_timeframe_train_cap", config["per_timeframe_train_cap"])
        add_optional_arg(argv, "--num_workers", config["num_workers"])
        add_optional_arg(argv, "--prefetch_factor", config["prefetch_factor"])
        add_optional_arg(argv, "--kill_keep_signal_frequency_max", config["kill_keep_signal_frequency_max"])
        add_optional_arg(
            argv,
            "--kill_keep_public_violation_rate_max",
            config["kill_keep_public_violation_rate_max"],
        )
        add_optional_arg(
            argv,
            "--kill_keep_timeframe_60m_composite_min",
            config["kill_keep_timeframe_60m_composite_min"],
        )
        add_optional_arg(
            argv,
            "--kill_keep_breakout_signal_space_min",
            config["kill_keep_breakout_signal_space_min"],
        )
        add_optional_arg(
            argv,
            "--kill_keep_reversion_signal_space_min",
            config["kill_keep_reversion_signal_space_min"],
        )
        add_optional_arg(argv, "--public_violation_cap", config["public_violation_cap"])
        add_optional_arg(argv, "--signal_frequency_cap_ratio", config["signal_frequency_cap_ratio"])
        add_optional_arg(argv, "--breakout_precision_floor", config["breakout_precision_floor"])
        add_optional_arg(argv, "--reversion_precision_floor", config["reversion_precision_floor"])
        add_optional_arg(argv, "--gate_mode", config.get("gate_mode"))
        add_optional_arg(argv, "--gate_floor_breakout", config.get("gate_floor_breakout"))
        add_optional_arg(argv, "--gate_floor_reversion", config.get("gate_floor_reversion"))
        add_optional_arg(argv, "--gate_anneal_fraction", config.get("gate_anneal_fraction"))
        add_optional_arg(argv, "--horizon_search_spec", config.get("horizon_search_spec"))
        if not config["deterministic"]:
            argv.append("--non_deterministic")
        if config["skip_dataset_cache_prewarm"]:
            argv.append("--skip_dataset_cache_prewarm")
        if config["prewarm_only"]:
            argv.append("--prewarm_dataset_cache_only")
        if config["fast_full"]:
            argv.append("--fast_full")
        if not config["prewarm_only"] and should_resume_chunk(config["resume_mode"], chunk_end, first_chunk_end, resume_path):
            argv.append("--resume")
        run_train(train_py, argv)

    report_title = None
    if not config["prewarm_only"]:
        report_title = f"iter12 multi-asset {config['run_label']} report"
        subprocess.run(
            [
                sys.executable,
                str(report_py),
                "--run_dir",
                str(save_dir),
                "--out",
                str(save_dir / "iter12_report.md"),
                "--title",
                report_title,
            ],
            check=False,
        )
    manifest = dict(config)
    manifest.update(
        {
            "run_id": run_id,
            "data_dir": str(data_dir),
            "run_root": str(run_root),
            "market": "legacy_multiasset",
            "training_ready_dir": str(training_ready_dir),
            "dataset_cache_dir": str(dataset_cache_dir),
            "report_title": report_title,
            "split_scheme": config["split_scheme"],
            "split_labels": config["split_labels"],
            "resume_mode": config["resume_mode"],
            "resume_path": str(resume_path),
        }
    )
    (save_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(save_dir))


if __name__ == "__main__":
    main()
