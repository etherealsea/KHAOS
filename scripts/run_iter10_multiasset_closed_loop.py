import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def extract_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith('.csv'):
                continue
            target = out_dir / Path(name).name
            with z.open(name) as src, target.open('wb') as dst:
                dst.write(src.read())


def run_train(train_py: Path, argv):
    cmd = [sys.executable, str(train_py)] + argv
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_zip', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--training_subdir', type=str, default='iter10_multiasset')
    parser.add_argument('--run_root', type=str, required=True)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--assets', type=str, required=True)
    parser.add_argument('--timeframes', type=str, default='15m,1h,1d')
    parser.add_argument('--epochs_total', type=int, default=6)
    parser.add_argument('--chunk_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--arch_version', type=str, default='iterA4_multiscale')
    parser.add_argument('--constraint_profile', type=str, default='teacher_feasible_discovery_v1')
    parser.add_argument('--score_profile', type=str, default='iter11_precision_first')
    parser.add_argument('--fast_full', action='store_true', default=False)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_iter10_multiasset_%H%M%S')
    save_dir = run_root / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    training_ready_dir = data_dir / 'training_ready' / args.training_subdir
    if args.data_zip:
        extract_zip(Path(args.data_zip).resolve(), training_ready_dir)

    train_py = Path('/workspace/Finance/02_核心代码/源代码/khaos/模型训练/train.py').resolve()
    report_py = Path('/workspace/scripts/generate_iter10_multiasset_report.py').resolve()

    for chunk_end in range(args.chunk_size, args.epochs_total + 1, args.chunk_size):
        argv = [
            '--data_dir', str(data_dir),
            '--training_subdir', args.training_subdir,
            '--save_dir', str(save_dir),
            '--market', 'legacy_multiasset',
            '--assets', args.assets,
            '--timeframes', args.timeframes,
            '--arch_version', args.arch_version,
            '--constraint_profile', args.constraint_profile,
            '--score_profile', args.score_profile,
            '--epochs', str(chunk_end),
            '--batch_size', str(args.batch_size),
        ]
        if args.fast_full:
            argv.append('--fast_full')
        if chunk_end != args.chunk_size:
            argv.append('--resume')
        run_train(train_py, argv)

    subprocess.run([sys.executable, str(report_py), '--run_dir', str(save_dir)], check=False)
    (save_dir / 'run_manifest.json').write_text(
        json.dumps(
            {
                'run_id': run_id,
                'data_dir': str(data_dir),
                'training_subdir': args.training_subdir,
                'assets': args.assets,
                'timeframes': args.timeframes,
                'epochs_total': args.epochs_total,
                'chunk_size': args.chunk_size,
                'arch_version': args.arch_version,
                'constraint_profile': args.constraint_profile,
                'score_profile': args.score_profile,
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    print(str(save_dir))


if __name__ == '__main__':
    main()
