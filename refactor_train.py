import re
import sys

def refactor_train_py():
    with open('Finance/02_核心代码/源代码/khaos/模型训练/train.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to insert the data loading BEFORE the epoch loop
    data_loading_code = """
    latest_metrics = None
    horizon_registry = {}

    print("Loading all datasets for global ConcatDataset...")
    all_train_datasets = []
    eval_jobs = []
    for job_idx, record in enumerate(runtime_records):
        data_path = record['path']
        split_label = record.get('split_label')
        try:
            train_ds, val_ds, test_ds, dataset_meta = create_market_datasets(
                record,
                args,
                global_horizon_grid=global_horizon_grid,
            )
            update_horizon_registry(horizon_registry, dataset_meta)
            eval_ds = val_ds if val_ds is not None else test_ds
            timeframe_label = normalize_timeframe_label(dataset_meta.get('timeframe') or record.get('timeframe')) or 'unknown'
            
            if train_ds is not None and len(train_ds) >= max(1, min(args.batch_size, 8)):
                capped_train_ds = build_capped_dataset(train_ds, args, timeframe_label)
                all_train_datasets.append(capped_train_ds)
            
            if eval_ds is not None and len(eval_ds) > 0:
                eval_jobs.append({
                    'eval_ds': eval_ds,
                    'timeframe_label': timeframe_label,
                    'split_label': split_label,
                    'data_path': data_path
                })
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Error processing {data_path} split={split_label}: {exc}")
            continue

    if not all_train_datasets:
        raise RuntimeError("No valid training datasets found.")

    global_train_ds = ConcatDataset(all_train_datasets)
    global_train_loader = DataLoader(
        global_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(global_train_ds) >= args.batch_size,
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Global training dataset created: {len(global_train_ds)} samples across {len(all_train_datasets)} datasets.")

    for epoch in range(start_epoch, args.epochs):
"""

    epoch_loop_inner = """
        print(f"\\n========== EPOCH {epoch + 1}/{args.epochs} ==========")
        epoch_train_loss_total = 0.0
        epoch_train_batches = 0
        epoch_train_logs = []
        epoch_all_bucket = build_metric_bucket()
        epoch_timeframe_buckets = defaultdict(build_metric_bucket)
        epoch_fold_buckets = defaultdict(build_metric_bucket)
        
        kan.train()
        total_loss = 0.0
        loss_logs = []
        job_effective_train_samples = 0
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        for batch_idx, batch in enumerate(global_train_loader):
            features_seq, batch_y, batch_aux, batch_sigma, batch_weights, batch_flags, horizon_payload = unpack_batch(batch, device)
            psi_t = features_seq[:, -1, :]
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
                pred, aux_pred, debug_info = forward_model(
                    kan,
                    features_seq,
                    args,
                    horizon_payload=horizon_payload,
                    return_debug=use_debug_metrics,
                )
                if torch.isnan(pred).any():
                    print("NaN in predictions. Skipping batch.")
                    continue
                    
                loss_unweighted, rank_loss, l_dict = criterion(
                    pred,
                    aux_pred,
                    batch_y,
                    batch_aux,
                    psi_t,
                    batch_flags,
                    batch_sigma,
                    debug_info=debug_info,
                    horizon_payload=horizon_payload,
                )
                reg_loss = kan.get_regularization_loss(regularize_activation=1e-4, regularize_entropy=1e-4)
                loss = (loss_unweighted * batch_weights).mean() + rank_loss + reg_loss
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(kan.parameters(), max_norm=getattr(args, 'grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += float(loss.item())
            
            batch_size_val = int(batch_y.size(0))
            job_effective_train_samples += batch_size_val
            
            l_dict_log = dict(l_dict)
            l_dict_log['total_loss'] = float(loss.item())
            l_dict_log['batch_size'] = batch_size_val
            loss_logs.append(l_dict_log)
            
            if batch_idx % 100 == 0:
                print(
                    f"  [EPOCH {epoch + 1}] batch {batch_idx}/{len(global_train_loader)} loss={loss.item():.6f}"
                )

        epoch_train_loss_total = total_loss
        epoch_train_batches = len(global_train_loader)
        epoch_train_logs.extend(loss_logs)
        
        # We don't track per-timeframe train samples precisely anymore because of the global shuffle,
        # but we can just use an empty dict or fake it.
        epoch_effective_train_samples_by_timeframe = defaultdict(int)

        kan.eval()
        processed_jobs = 0
        for eval_job in eval_jobs:
            data_path = eval_job['data_path']
            split_label = eval_job['split_label']
            timeframe_label = eval_job['timeframe_label']
            
            try:
                eval_loader = build_eval_loader(eval_job['eval_ds'], args)
                file_bucket = evaluate_dataset_loader(
                    kan=kan,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    args=args,
                    device=device,
                    use_debug_metrics=use_debug_metrics,
                )
                file_summary = summarize_metric_bucket(
                    file_bucket,
                    score_profile=score_profile,
                    use_direction_metrics=use_direction_metrics,
                )
                
                print(
                    f"  -> {os.path.basename(data_path)} [{split_label or 'default'}|{timeframe_label}] "
                    f"val={file_summary['avg_val_loss']:.6f} "
                    f"precision={file_summary['breakout_metrics']['precision']:.4f}/{file_summary['reversion_metrics']['precision']:.4f} "
                    f"composite={file_summary['composite_score']:.4f}"
                )
                
                merge_metric_bucket(epoch_all_bucket, file_bucket)
                merge_metric_bucket(epoch_timeframe_buckets[timeframe_label], file_bucket)
                if split_label:
                    merge_metric_bucket(epoch_fold_buckets[split_label], file_bucket)
                processed_jobs += 1
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"Error evaluating {data_path} split={split_label}: {exc}")
                continue

        if processed_jobs == 0 or not epoch_all_bucket['preds']:
"""

    search_pattern = r"    latest_metrics = None\n    horizon_registry = \{\}\n    for epoch in range\(start_epoch, args\.epochs\):[\s\S]*?        if processed_jobs == 0 or not epoch_all_bucket\['preds'\]:"
    
    new_content = re.sub(search_pattern, data_loading_code + epoch_loop_inner, content)
    
    if new_content == content:
        print("Failed to replace content!")
        sys.exit(1)
        
    with open('Finance/02_核心代码/源代码/khaos/模型训练/train.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully refactored train.py!")

if __name__ == '__main__':
    refactor_train_py()
