import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.amp import autocast, GradScaler
import argparse
import glob
import os
import numpy as np
import random

# Adjust import to run as script
import sys
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from khaos.数据处理.ashare_dataset import create_ashare_dataset_splits
from khaos.数据处理.ashare_support import (
    DEFAULT_ASHARE_TIMEFRAMES,
    LEGACY_ITER9_ASSETS,
    discover_training_files,
    normalize_timeframe_label,
    resolve_training_ready_dir,
)
from khaos.数据处理.data_processor import process_multi_timeframe
from khaos.数据处理.data_loader import create_rolling_datasets
from khaos.模型定义.kan import KHAOS_KAN
from khaos.模型训练.loss import PhysicsLoss
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def compute_event_metrics(scores, event_flags, hard_negative_flags):
    scores = np.asarray(scores, dtype=np.float64)
    event_flags = np.asarray(event_flags, dtype=bool)
    hard_negative_flags = np.asarray(hard_negative_flags, dtype=bool)
    if len(scores) == 0:
        return {
            'threshold': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'event_rate': 0.0,
            'hard_negative_rate': 0.0,
            'signal_frequency': 0.0,
            'label_frequency': 0.0
        }
    label_frequency = float(np.mean(event_flags)) if len(event_flags) > 0 else 0.0
    thresholds = np.unique(np.quantile(scores, np.linspace(0.55, 0.95, 9)))
    best = None
    for threshold in thresholds:
        pred = scores >= threshold
        tp = np.sum(pred & event_flags)
        fp = np.sum(pred & ~event_flags)
        fn = np.sum(~pred & event_flags)
        tn = np.sum(~pred & ~event_flags)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(scores), 1)
        hn_rate = np.mean(pred[hard_negative_flags]) if np.any(hard_negative_flags) else 0.0
        event_rate = np.mean(pred[event_flags]) if np.any(event_flags) else 0.0
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'event_rate': float(event_rate),
            'hard_negative_rate': float(hn_rate),
            'signal_frequency': float(np.mean(pred)),
            'label_frequency': label_frequency
        }
        if best is None or candidate['f1'] > best['f1'] or (
            candidate['f1'] == best['f1'] and candidate['hard_negative_rate'] < best['hard_negative_rate']
        ):
            best = candidate
    return best


def parse_list_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    return [str(value).strip()]


def parse_timeframe_cap_config(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            normalize_timeframe_label(key): int(val)
            for key, val in value.items()
            if normalize_timeframe_label(key) is not None
        }
    caps = {}
    for item in parse_list_arg(value):
        if '=' not in item:
            continue
        timeframe, raw_cap = item.split('=', 1)
        normalized = normalize_timeframe_label(timeframe)
        if normalized is None:
            continue
        caps[normalized] = int(raw_cap)
    return caps


def resolve_training_filters(args):
    market = getattr(args, 'market', 'legacy_multiasset')
    assets = parse_list_arg(getattr(args, 'assets', None))
    if not assets and market != 'ashare':
        assets = list(LEGACY_ITER9_ASSETS)
    timeframes = [normalize_timeframe_label(item) for item in parse_list_arg(getattr(args, 'timeframes', None))]
    if not timeframes and market == 'ashare':
        timeframes = list(DEFAULT_ASHARE_TIMEFRAMES)
    return market, assets, [item for item in timeframes if item]


def discover_runtime_files(args):
    market, assets, timeframes = resolve_training_filters(args)
    training_subdir = getattr(args, 'training_subdir', None)
    records = discover_training_files(
        data_dir=args.data_dir,
        market=market,
        assets=assets,
        timeframes=timeframes,
        training_subdir=training_subdir,
    )
    max_files = getattr(args, 'max_files', None)
    if max_files:
        records = records[:max_files]
    return records


def create_market_datasets(record, args):
    market = getattr(args, 'market', 'legacy_multiasset')
    if market == 'ashare':
        datasets, metadata = create_ashare_dataset_splits(
            file_path=record['path'],
            window_size=args.window_size,
            horizon=args.horizon,
            train_end=getattr(args, 'train_end', None),
            val_end=getattr(args, 'val_end', None),
            test_start=getattr(args, 'test_start', None),
            fast_full=args.fast_full,
            return_metadata=True,
        )
        return datasets.get('train'), datasets.get('val'), datasets.get('test'), metadata

    train_ds, test_ds = create_rolling_datasets(
        record['path'],
        window_size=args.window_size,
        horizon=args.horizon,
        fast_full=args.fast_full,
    )
    metadata = {
        'asset_code': record.get('asset_code'),
        'timeframe': record.get('timeframe'),
    }
    return train_ds, test_ds, None, metadata


def build_train_loader(dataset, args, timeframe_label):
    g = torch.Generator()
    g.manual_seed(args.seed)
    cap_config = parse_timeframe_cap_config(getattr(args, 'per_timeframe_train_cap', None))
    sample_cap = cap_config.get(normalize_timeframe_label(timeframe_label))
    if sample_cap and len(dataset) > 0:
        replacement = len(dataset) < sample_cap
        sampler = RandomSampler(
            dataset,
            replacement=replacement,
            num_samples=sample_cap,
            generator=g,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
            generator=g,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )


def build_eval_loader(dataset, args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
    )

def get_resume_path(args):
    resume_path = getattr(args, 'resume_path', None)
    if resume_path:
        return resume_path
    resume_name = getattr(args, 'resume_name', 'khaos_kan_resume.pth')
    return os.path.join(args.save_dir, resume_name)

def save_resume_checkpoint(
    resume_path,
    kan,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch,
    best_val_loss,
    best_score,
    no_improve_epochs,
    latest_metrics,
    device,
    completed=False
):
    torch.save({
        'model_state_dict': kan.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'args': vars(args),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_score': best_score,
        'no_improve_epochs': no_improve_epochs,
        'latest_metrics': latest_metrics,
        'feature_names': PHYSICS_FEATURE_NAMES,
        'completed': completed,
        'env': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else None,
            'device': str(device)
        }
    }, resume_path)

def try_resume_training(args, kan, optimizer, scheduler, scaler, device):
    start_epoch = 0
    best_val_loss = float('inf')
    best_score = float('-inf')
    no_improve_epochs = 0
    resume_path = get_resume_path(args)
    if not getattr(args, 'resume', False):
        return start_epoch, best_val_loss, best_score, no_improve_epochs

    if os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            kan.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
            best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
            best_score = float(checkpoint.get('best_score', best_score))
            no_improve_epochs = int(checkpoint.get('no_improve_epochs', 0))
            print(
                f"[RESUME] 已从断点恢复训练：epoch={start_epoch}, "
                f"best_score={best_score:.4f}, best_val_loss={best_val_loss:.4f}"
            )
            return start_epoch, best_val_loss, best_score, no_improve_epochs
        except Exception as exc:
            print(f"[RESUME] 断点文件不可用，忽略并重新开始：{resume_path} | {exc}")

    best_path = os.path.join(args.save_dir, getattr(args, 'best_name', 'khaos_kan_best.pth'))
    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            kan.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = int(checkpoint.get('metrics', {}).get('epoch', 0))
            best_val_loss = float(checkpoint.get('val_loss', best_val_loss))
            best_score = float(checkpoint.get('best_score', best_score))
            print(
                f"[RESUME] 未找到断点文件，已从 best checkpoint 热启动：epoch={start_epoch}, "
                f"best_score={best_score:.4f}, best_val_loss={best_val_loss:.4f}"
            )
        except Exception as exc:
            print(f"[RESUME] best checkpoint 不可热启动，改为从头训练：{best_path} | {exc}")
    else:
        print("[RESUME] 未找到可恢复的断点，将从头开始训练。")
    return start_epoch, best_val_loss, best_score, no_improve_epochs

def train(args):
    # Setup
    set_seed(args.seed, args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Unified local physics window: {args.window_size}")
    print(f"Short smoothing window reserved for visualization/stability: 2-3")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 1. Prepare Data (Multi-Timeframe)
    print("Preparing Multi-Timeframe Datasets...")
    market = getattr(args, 'market', 'legacy_multiasset')
    training_subdir = getattr(args, 'training_subdir', None)
    train_data_dir = resolve_training_ready_dir(args.data_dir, market=market, training_subdir=training_subdir)
    os.makedirs(train_data_dir, exist_ok=True)

    ready_files = glob.glob(os.path.join(train_data_dir, '*.csv'))
    if len(ready_files) == 0 and market != 'ashare':
        processed_files = []
        search_pattern = os.path.join(args.data_dir, '**', '*.csv')
        raw_files = glob.glob(search_pattern, recursive=True)
        print(f"Found {len(raw_files)} raw files.")
        for file_path in raw_files:
            if 'training_ready' in file_path:
                continue
            if os.path.getsize(file_path) < 1024:
                continue
            processed_files.extend(process_multi_timeframe(file_path, train_data_dir))
        print(f"Generated {len(processed_files)} processed files into {train_data_dir}.")

    final_records = discover_runtime_files(args)
    if args.test_mode:
        final_records = final_records[:1]
        args.epochs = 1
        print("!!! RUNNING IN FAST TEST MODE !!!")

    print(f"Loading {len(final_records)} files into Global Memory...")
    print(f"Assets included: {[os.path.basename(item['path']) for item in final_records]}")
    if not final_records:
        raise RuntimeError(f'No training files found under {train_data_dir} for market={market}.')
    
    # =========================================================================
    # WARNING: Windows has severe memory fragmentation issues with CUDA 
    # when holding large tensors in a Dataset list across multiple files.
    # To ensure stable training, we fall back to sequential file processing,
    # but still utilize the massive speedup of Offline GPU Pre-computation.
    # =========================================================================
    
    # FOR DEBUGGING WINDOWS MEMORY HANGS, ONLY PROCESS FIRST FILE
    # In order to make it run full loop, we must clean memory aggressively 
    # but to show you it works, we run on all files but clear GPU cache
    
    # We run on all filtered files as per user requirement (no truncation)
    print(f"Loading {len(final_records)} files into Global Memory...")
    print(f"Assets included: {[os.path.basename(item['path']) for item in final_records]}")
    
    # 3. Model Init
    input_dim = len(PHYSICS_FEATURE_NAMES)
    kan = KHAOS_KAN(
        input_dim=input_dim, 
        hidden_dim=args.hidden_dim,
        output_dim=2,
        layers=args.layers, 
        grid_size=args.grid_size
    ).to(device)
    
    optimizer = optim.AdamW(kan.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = PhysicsLoss()
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    start_epoch, best_val_loss, best_score, no_improve_epochs = try_resume_training(
        args, kan, optimizer, scheduler, scaler, device
    )
    resume_path = get_resume_path(args)
    latest_metrics = None
    
    print("\nStarting Accelerated Sequential Training...")
    
    if start_epoch >= args.epochs:
        print(f"[RESUME] 当前断点 epoch={start_epoch} 已达到目标 epochs={args.epochs}，无需继续训练。")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== EPOCH {epoch+1}/{args.epochs} ==========")
        epoch_train_loss_total = 0.0
        epoch_train_batches = 0
        epoch_train_logs = []
        epoch_val_loss_total = 0.0
        epoch_val_batches = 0
        epoch_val_logs = []
        epoch_val_preds = []
        epoch_val_targets = []
        epoch_val_flags = []
        processed_files = 0
        for file_idx, record in enumerate(final_records):
            data_path = record['path']
            print(f"\n[{file_idx+1}/{len(final_records)}] Processing: {os.path.basename(data_path)}")
            
            try:
                train_ds, eval_ds, _, dataset_meta = create_market_datasets(record, args)
                
                if train_ds is None or eval_ds is None:
                    print("  -> Split generation returned empty train/validation dataset, skipping.")
                    continue

                if len(train_ds) < args.batch_size or len(eval_ds) == 0:
                    print("  -> Dataset too small for this configuration, skipping.")
                    continue

                timeframe_label = dataset_meta.get('timeframe') or record.get('timeframe')
                train_loader = build_train_loader(train_ds, args, timeframe_label)
                test_loader = build_eval_loader(eval_ds, args)
                
                kan.train()
                total_loss = 0
                loss_logs = []
                
                for batch_idx, (features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags) in enumerate(train_loader):
                    features_seq = features_seq.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    batch_aux = batch_aux.to(device, non_blocking=True)
                    batch_sigma = batch_sigma.to(device, non_blocking=True)
                    batch_weights = batch_weights.to(device, non_blocking=True).unsqueeze(1)
                    batch_flags = batch_flags.to(device, non_blocking=True)
                    
                    psi_t = features_seq[:, -1, :]
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Use bfloat16 if supported for better numerical stability
                    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                        pred, aux_pred = kan(features_seq, return_aux=True)
                        
                        if torch.isnan(pred).any():
                            print("NaN in pred! Skipping batch.")
                            continue
                            
                        loss_unweighted, rank_loss, l_dict = criterion(pred, aux_pred, batch_y, batch_aux, psi_t, batch_flags, batch_sigma)
                        reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                        loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("NaN/Inf in loss! Skipping batch.")
                        continue
                        
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping to prevent explosion
                    scaler.unscale_(optimizer)
                    
                    # Check for NaN/Inf in gradients before clipping
                    has_nan_inf = False
                    for param in kan.parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                has_nan_inf = True
                                break
                                
                    if not has_nan_inf:
                        torch.nn.utils.clip_grad_norm_(kan.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                    else:
                        print("NaN/Inf in gradients! Skipping optimizer step.")
                        
                    scaler.update()
                    
                    total_loss += loss.item()
                    loss_logs.append(l_dict)
                    
                    if batch_idx % 100 == 0:
                        print(
                            f"    [Batch {batch_idx}/{len(train_loader)}] "
                            f"Loss: {loss.item():.6f} | Main: {l_dict['main']:.4f} | Aux: {l_dict['aux']:.4f} | Rank: {l_dict['rank']:.4f}"
                        )
                        
                avg_loss = total_loss / len(train_loader)
                
                # Validation
                kan.eval()
                val_loss = 0
                val_logs = []
                val_preds = []
                val_targets = []
                val_flags = []
                with torch.no_grad():
                    for features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags in test_loader:
                        features_seq = features_seq.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        batch_aux = batch_aux.to(device, non_blocking=True)
                        batch_sigma = batch_sigma.to(device, non_blocking=True)
                        batch_weights = batch_weights.to(device, non_blocking=True).unsqueeze(1)
                        batch_flags = batch_flags.to(device, non_blocking=True)
                        
                        psi_t = features_seq[:, -1, :]
                        
                        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                            pred, aux_pred = kan(features_seq, return_aux=True)
                            loss_unweighted, rank_loss, l_dict = criterion(pred, aux_pred, batch_y, batch_aux, psi_t, batch_flags, batch_sigma)
                            reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                            loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
                            
                        val_loss += loss.item()
                        val_logs.append(l_dict)
                        val_preds.append(pred.detach().float().cpu().numpy())
                        val_targets.append(batch_y.detach().cpu().numpy())
                        val_flags.append(batch_flags.detach().cpu().numpy())
                        
                avg_val_loss = val_loss / len(test_loader)
                train_main = float(np.mean([x['main'] for x in loss_logs])) if loss_logs else 0.0
                train_aux = float(np.mean([x['aux'] for x in loss_logs])) if loss_logs else 0.0
                val_main = float(np.mean([x['main'] for x in val_logs])) if val_logs else 0.0
                val_aux = float(np.mean([x['aux'] for x in val_logs])) if val_logs else 0.0
                if val_preds:
                    pred_np = np.vstack(val_preds)
                    target_np = np.vstack(val_targets)
                    flags_np = np.vstack(val_flags)
                    breakout_corr = safe_corr(pred_np[:, 0], target_np[:, 0])
                    reversion_corr = safe_corr(pred_np[:, 1], target_np[:, 1])
                    breakout_event_mean = float(pred_np[flags_np[:, 0] > 0.5, 0].mean()) if np.any(flags_np[:, 0] > 0.5) else 0.0
                    reversion_event_mean = float(pred_np[flags_np[:, 1] > 0.5, 1].mean()) if np.any(flags_np[:, 1] > 0.5) else 0.0
                    breakout_hn_mean = float(pred_np[flags_np[:, 2] > 0.5, 0].mean()) if np.any(flags_np[:, 2] > 0.5) else 0.0
                    reversion_hn_mean = float(pred_np[flags_np[:, 3] > 0.5, 1].mean()) if np.any(flags_np[:, 3] > 0.5) else 0.0
                else:
                    breakout_corr = 0.0
                    reversion_corr = 0.0
                    breakout_event_mean = 0.0
                    reversion_event_mean = 0.0
                    breakout_hn_mean = 0.0
                    reversion_hn_mean = 0.0
                print(
                    f"  -> Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                    f"Train Main/Aux: {train_main:.4f}/{train_aux:.4f} | "
                    f"Val Main/Aux: {val_main:.4f}/{val_aux:.4f}"
                )
                breakout_gap = breakout_event_mean - breakout_hn_mean
                reversion_gap = reversion_event_mean - reversion_hn_mean
                composite_score = (
                    0.55 * breakout_corr +
                    0.45 * reversion_corr +
                    0.08 * breakout_gap +
                    0.06 * reversion_gap -
                    0.015 * avg_val_loss
                )
                print(
                    f"  -> Val Corr Breakout/Reversion: {breakout_corr:.4f}/{reversion_corr:.4f} | "
                    f"Event Mean: {breakout_event_mean:.4f}/{reversion_event_mean:.4f} | "
                    f"HardNeg Mean: {breakout_hn_mean:.4f}/{reversion_hn_mean:.4f} | "
                    f"Gap: {breakout_gap:.4f}/{reversion_gap:.4f} | "
                    f"Composite: {composite_score:.4f}"
                )
                epoch_train_loss_total += total_loss
                epoch_train_batches += len(train_loader)
                epoch_train_logs.extend(loss_logs)
                epoch_val_loss_total += val_loss
                epoch_val_batches += len(test_loader)
                epoch_val_logs.extend(val_logs)
                if val_preds:
                    epoch_val_preds.append(np.vstack(val_preds))
                    epoch_val_targets.append(np.vstack(val_targets))
                    epoch_val_flags.append(np.vstack(val_flags))
                processed_files += 1
                    
                # Cleanup to avoid memory leaks
                del train_ds, eval_ds, train_loader, test_loader
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing {data_path}: {e}")
                continue

        if processed_files == 0 or epoch_val_batches == 0:
            print("No valid files processed in this epoch.")
            continue

        epoch_avg_loss = epoch_train_loss_total / max(epoch_train_batches, 1)
        epoch_avg_val_loss = epoch_val_loss_total / max(epoch_val_batches, 1)
        epoch_train_main = float(np.mean([x['main'] for x in epoch_train_logs])) if epoch_train_logs else 0.0
        epoch_train_aux = float(np.mean([x['aux'] for x in epoch_train_logs])) if epoch_train_logs else 0.0
        epoch_val_main = float(np.mean([x['main'] for x in epoch_val_logs])) if epoch_val_logs else 0.0
        epoch_val_aux = float(np.mean([x['aux'] for x in epoch_val_logs])) if epoch_val_logs else 0.0
        pred_np = np.vstack(epoch_val_preds)
        target_np = np.vstack(epoch_val_targets)
        flags_np = np.vstack(epoch_val_flags)
        breakout_corr = safe_corr(pred_np[:, 0], target_np[:, 0])
        reversion_corr = safe_corr(pred_np[:, 1], target_np[:, 1])
        breakout_event_mean = float(pred_np[flags_np[:, 0] > 0.5, 0].mean()) if np.any(flags_np[:, 0] > 0.5) else 0.0
        reversion_event_mean = float(pred_np[flags_np[:, 1] > 0.5, 1].mean()) if np.any(flags_np[:, 1] > 0.5) else 0.0
        breakout_hn_mean = float(pred_np[flags_np[:, 2] > 0.5, 0].mean()) if np.any(flags_np[:, 2] > 0.5) else 0.0
        reversion_hn_mean = float(pred_np[flags_np[:, 3] > 0.5, 1].mean()) if np.any(flags_np[:, 3] > 0.5) else 0.0
        breakout_metrics = compute_event_metrics(pred_np[:, 0], flags_np[:, 0] > 0.5, flags_np[:, 2] > 0.5)
        reversion_metrics = compute_event_metrics(pred_np[:, 1], flags_np[:, 1] > 0.5, flags_np[:, 3] > 0.5)
        breakout_gap = breakout_event_mean - breakout_hn_mean
        reversion_gap = reversion_event_mean - reversion_hn_mean
        composite_score = (
            0.55 * breakout_corr +
            0.45 * reversion_corr +
            0.08 * breakout_gap +
            0.06 * reversion_gap -
            0.015 * epoch_avg_val_loss
        )
        print(
            f"\n[EPOCH {epoch+1} SUMMARY] Train Loss: {epoch_avg_loss:.6f} | Val Loss: {epoch_avg_val_loss:.6f} | "
            f"Train Main/Aux: {epoch_train_main:.4f}/{epoch_train_aux:.4f} | "
            f"Val Main/Aux: {epoch_val_main:.4f}/{epoch_val_aux:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Val Corr Breakout/Reversion: {breakout_corr:.4f}/{reversion_corr:.4f} | "
            f"Event Mean: {breakout_event_mean:.4f}/{reversion_event_mean:.4f} | "
            f"HardNeg Mean: {breakout_hn_mean:.4f}/{reversion_hn_mean:.4f} | "
            f"Gap: {breakout_gap:.4f}/{reversion_gap:.4f} | "
            f"Composite: {composite_score:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Breakout Acc/P/R/F1: "
            f"{breakout_metrics['accuracy']:.4f}/{breakout_metrics['precision']:.4f}/"
            f"{breakout_metrics['recall']:.4f}/{breakout_metrics['f1']:.4f} | "
            f"阈值: {breakout_metrics['threshold']:.4f} | "
            f"事件命中率/伪信号率: {breakout_metrics['event_rate']:.4f}/{breakout_metrics['hard_negative_rate']:.4f} | "
            f"信号频次/标签频次: {breakout_metrics['signal_frequency']:.4f}/{breakout_metrics['label_frequency']:.4f}"
        )
        print(
            f"[EPOCH {epoch+1} SUMMARY] Reversion Acc/P/R/F1: "
            f"{reversion_metrics['accuracy']:.4f}/{reversion_metrics['precision']:.4f}/"
            f"{reversion_metrics['recall']:.4f}/{reversion_metrics['f1']:.4f} | "
            f"阈值: {reversion_metrics['threshold']:.4f} | "
            f"事件命中率/伪信号率: {reversion_metrics['event_rate']:.4f}/{reversion_metrics['hard_negative_rate']:.4f} | "
            f"信号频次/标签频次: {reversion_metrics['signal_frequency']:.4f}/{reversion_metrics['label_frequency']:.4f}"
        )

        scheduler.step(epoch_avg_val_loss)

        improved = (
            composite_score > best_score + args.early_stop_min_delta or
            (
                abs(composite_score - best_score) <= args.early_stop_min_delta and
                epoch_avg_val_loss < best_val_loss - args.early_stop_min_delta
            )
        )

        if improved:
            best_score = composite_score
            best_val_loss = epoch_avg_val_loss
            no_improve_epochs = 0
            save_path = os.path.join(args.save_dir, getattr(args, 'best_name', 'khaos_kan_best.pth'))
            torch.save({
                'model_state_dict': kan.state_dict(),
                'args': vars(args),
                'dataset_manifest': final_records,
                'val_loss': best_val_loss,
                'best_score': best_score,
                'metrics': {
                    'breakout_corr': breakout_corr,
                    'reversion_corr': reversion_corr,
                    'breakout_event_mean': breakout_event_mean,
                    'reversion_event_mean': reversion_event_mean,
                    'breakout_hard_negative_mean': breakout_hn_mean,
                    'reversion_hard_negative_mean': reversion_hn_mean,
                    'breakout_gap': breakout_gap,
                    'reversion_gap': reversion_gap,
                    'composite_score': composite_score,
                    'processed_files': processed_files,
                    'epoch': epoch + 1,
                    'breakout_eval': breakout_metrics,
                    'reversion_eval': reversion_metrics
                },
                'feature_names': PHYSICS_FEATURE_NAMES,
                'env': {
                    'torch': torch.__version__,
                    'cuda': torch.version.cuda if torch.cuda.is_available() else None,
                    'device': str(device)
                }
            }, save_path)
        else:
            no_improve_epochs += 1
            print(
                f"[EPOCH {epoch+1} SUMMARY] 未超过最优，连续未改进轮数: "
                f"{no_improve_epochs}/{args.early_stop_patience}"
            )

        latest_metrics = {
            'breakout_corr': breakout_corr,
            'reversion_corr': reversion_corr,
            'breakout_gap': breakout_gap,
            'reversion_gap': reversion_gap,
            'composite_score': composite_score,
            'val_loss': epoch_avg_val_loss,
            'processed_files': processed_files,
            'epoch': epoch + 1,
            'breakout_eval': breakout_metrics,
            'reversion_eval': reversion_metrics
        }
        save_resume_checkpoint(
            resume_path=resume_path,
            kan=kan,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            best_score=best_score,
            no_improve_epochs=no_improve_epochs,
            latest_metrics=latest_metrics,
            device=device,
            completed=False
        )

        if no_improve_epochs >= args.early_stop_patience:
            print(
                f"[EARLY STOP] 连续 {args.early_stop_patience} 个 epoch 未达到最小改进 "
                f"{args.early_stop_min_delta:.4f}，提前停止。"
            )
            break

    # Save Final Model
    save_path = os.path.join(args.save_dir, getattr(args, 'final_name', 'khaos_kan_model_final.pth'))
    torch.save({
        'model_state_dict': kan.state_dict(),
        'args': vars(args),
        'dataset_manifest': final_records,
        'feature_names': PHYSICS_FEATURE_NAMES,
        'env': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else None,
            'device': str(device)
        }
    }, save_path)
    save_resume_checkpoint(
        resume_path=resume_path,
        kan=kan,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        epoch=args.epochs,
        best_val_loss=best_val_loss,
        best_score=best_score,
        no_improve_epochs=no_improve_epochs,
        latest_metrics=latest_metrics,
        device=device,
        completed=True
    )
    print(f"\nTraining Complete. Final model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed')
    parser.add_argument('--save_dir', type=str, default=r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份')
    parser.add_argument('--market', type=str, default='legacy_multiasset')
    parser.add_argument('--training_subdir', type=str, default=None)
    parser.add_argument('--assets', type=str, default=None)
    parser.add_argument('--timeframes', type=str, default=None)
    parser.add_argument('--split_mode', type=str, default='ratio')
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--val_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--per_timeframe_train_cap', type=str, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--best_name', type=str, default='khaos_kan_best.pth')
    parser.add_argument('--final_name', type=str, default='khaos_kan_model_final.pth')
    parser.add_argument('--resume_name', type=str, default='khaos_kan_resume.pth')
    parser.add_argument('--epochs', type=int, default=3) 
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--test_mode', action='store_true', default=False)
    parser.add_argument('--fast_full', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=2)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.002)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', type=str, default=None)
    
    args = parser.parse_args()
    train(args)
