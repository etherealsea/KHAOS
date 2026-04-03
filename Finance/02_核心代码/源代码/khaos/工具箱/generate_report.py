import torch
import torch.nn as nn
import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from khaos.模型定义.kan import KHAOS_KAN
from khaos.核心引擎.physics import PhysicsLayer
from khaos.数据处理.data_loader import create_rolling_datasets

import torch.nn.functional as F

def generate_report(model_path, data_dir, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path} on {device}")
    
    # Load Model
    # Input dim is 10 (5 Physics + 5 Stats)
    # We need to load args to know hidden dim and layers if possible, 
    # but for now assuming defaults or standard values if not saved in checkpoint
    # The train script saved: {'model_state_dict': ..., 'args': ...}
    
    # Use weights_only=False because we saved argparse.Namespace in the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint.get('args')
    
    hidden_dim = args.hidden_dim if args else 16
    layers = args.layers if args else 2
    grid_size = args.grid_size if args and hasattr(args, 'grid_size') else 10
    
    # Input Dim: 16 steps * 14 features = 224
    input_dim = 16 * 14
    
    # If checkpoint has args, respect them strictly.
    
    # If checkpoint has args, respect them strictly.
    if args:
        # Check if the checkpoint args actually match what we expect from the trained model
        # The training script used hidden=64, layers=3.
        
        # Force correct config for the new training run if needed
        hidden_dim = 64
        layers = 3
        print(f"Forcing Config: hidden={hidden_dim}, layers={layers}")
    else:
        hidden_dim = 64
        layers = 3
        
    kan = KHAOS_KAN(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=1, 
        layers=layers, 
        grid_size=grid_size
    ).to(device)
    
    print(f"Loading state dict with input_dim={input_dim}")
    try:
        kan.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Check if input_dim matches checkpoint.")
        raise e
        
    kan.eval()
    
    physics = PhysicsLayer().to(device)
    
    # Select Representative Data
    # Try to find one from each category
    categories = ['Crypto', 'Forex', 'Commodity', 'Index']
    selected_files = []
    
    for cat in categories:
        pattern = os.path.join(data_dir, cat, '*1h.csv') # Prefer 1h for standardization
        files = glob.glob(pattern)
        if not files:
            pattern = os.path.join(data_dir, cat, '*.csv')
            files = glob.glob(pattern)
            
        if files:
            # Pick the largest file usually means more history
            largest = max(files, key=os.path.getsize)
            selected_files.append((cat, largest))
    
    if not selected_files:
        # Fallback
        all_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        selected_files = [('General', f) for f in all_files[:5]]
        
    # selected_files = selected_files[:1] # Removed debug limit
        
    results = []
    
    print(f"Evaluating {len(selected_files)} files...")
    
    for cat, file_path in selected_files:
        print(f"  Processing {cat}: {os.path.basename(file_path)}")
        try:
            # Disable subsampling for full evaluation
            # Use data_loader directly with no subsampling
            _, test_ds = create_rolling_datasets(file_path, window_size=32, horizon=4, subsample=False)
            if len(test_ds) == 0:
                print("    Empty dataset")
                continue
                
            loader = DataLoader(test_ds, batch_size=256, shuffle=False)
            
            total_samples = 0
            setup_samples = 0
            total_loss = 0
            correct_direction = 0
            setup_correct_direction = 0
            
            # Physics Metrics
            p1_violations = 0 # Kinetic consistency
            p5_violations = 0 # Velocity consistency
            
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_sigma, batch_weights in loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).unsqueeze(1)
                    batch_weights = batch_weights.to(device).unsqueeze(1)
                    
                    # Physics
                    # Returns [B, T, 14]
                    psi_seq = physics(batch_x)
                    
                    # Temporal Flattening (Last 16 steps)
                    steps_to_use = 16
                    if psi_seq.shape[1] < steps_to_use:
                        steps_to_use = psi_seq.shape[1]
                    
                    features_seq = psi_seq[:, -steps_to_use:, :] # [B, 16, 14]
                    features = features_seq.reshape(features_seq.shape[0], -1) # [B, 224]
                    
                    pred = kan(features)
                    
                    # Metrics
                    # Regression Metrics (Log Volatility Change)
                    
                    # MAE
                    mae = torch.abs(pred - batch_y).mean()
                    
                    # MSE
                    loss = torch.nn.functional.mse_loss(pred, batch_y, reduction='sum')
                    total_loss += loss.item()
                    
                    # Store for R2
                    predictions.append(pred.cpu().numpy())
                    targets.append(batch_y.cpu().numpy())
                    
                    # Physics metrics (Vol consistency)
                    # P3: Ent > 0.7 and Pred < Curr
                    # P4: Res > 0.01 and Pred < Curr
                    # Just track raw violation rate
                    
                    # Need psi_t_physics for current vol
                    psi_t_physics = psi_seq[:, -1, :] 
                    curr_log_vol = torch.log(psi_t_physics[:, 1] + 1e-8).unsqueeze(1)
                    
                    # "Trend" Violation in Volatility context?
                    # Maybe "Anti-Persistence Violation":
                    # If Vol is rising, Pred should be rising? Not necessarily (mean reversion).
                    # Let's skip physics violation rate for now as it's redefined.
                    
                    total_samples += batch_x.size(0)
            
            if total_samples > 0:
                mse = total_loss / total_samples
                
                # R2 Score
                preds_flat = np.concatenate(predictions).flatten()
                targets_flat = np.concatenate(targets).flatten()
                
                ss_res = np.sum((targets_flat - preds_flat) ** 2)
                ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                
                # MAE
                mae = np.mean(np.abs(targets_flat - preds_flat))
                
                # Correlation
                if len(preds_flat) > 1:
                    corr = np.corrcoef(preds_flat, targets_flat)[0, 1]
                else:
                    corr = 0.0
                    
                print(f"    Done: MSE={mse:.4f}, R2={r2:.4f}, MAE={mae:.4f}, Corr={corr:.4f}")
                
                results.append({
                    'Category': cat,
                    'Asset': os.path.basename(file_path),
                    'Samples': total_samples,
                    'Setup Samples': setup_samples,
                    'MSE': mse,
                    'R2 Score': r2,
                    'MAE': mae,
                    'Correlation': corr,
                    'Physics Violation Rate': 0.0 # Not applicable
                })
            else:
                print("    No samples processed")
            
        except Exception as e:
            print(f"Error evaluating {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"Finished evaluation loop. Results count: {len(results)}")
            
    # Generate Markdown Report
    df = pd.DataFrame(results)
    
    report = "# KHAOS-KAN 模型训练评估报告\n\n"
    report += f"**模型路径**: `{model_path}`  \n"
    report += f"**评估时间**: {pd.Timestamp.now()}\n\n"
    
    report += "## 1. 总体表现摘要\n"
    if not df.empty:
        report += f"- **平均 R2 Score**: {df['R2 Score'].mean():.4f}\n"
        report += f"- **平均 Correlation**: {df['Correlation'].mean():.4f}\n"
        report += f"- **平均 MSE**: {df['MSE'].mean():.4f}\n"
    else:
        report += "无有效评估结果。\n"
        
    report += "\n## 2. 分资产类别详细评估\n"
    # report += df.to_markdown(index=False)
    
    # Manual Markdown Table
    cols = ['Category', 'Asset', 'Samples', 'MSE', 'R2 Score', 'MAE', 'Correlation']
    report += "| " + " | ".join(cols) + " |\n"
    report += "| " + " | ".join(['---'] * len(cols)) + " |\n"
    
    for _, row in df.iterrows():
        row_str = [
            str(row['Category']),
            str(row['Asset']),
            str(row['Samples']),
            f"{row['MSE']:.4f}",
            f"{row['R2 Score']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['Correlation']:.4f}"
        ]
        report += "| " + " | ".join(row_str) + " |\n"
    
    report += "\n## 3. 结果分析\n"
    report += "- **R2 Score**: 衡量模型对未来波动率变化（Vol Return）的解释程度。\n"
    report += "- **Correlation**: 预测变化与真实变化的相关性。正相关表示能捕捉到波动率扩张/收缩的趋势。\n"
    report += "- **Target**: Log(FutureVol / CurrentVol)。预测波动率的相对变化。\n"
    
    report += "\n## 4. 后续建议\n"
    report += "- 如果 **Correlation** > 0.1，说明模型具有预测波动率方向的能力（Gamma Trading）。\n"
    report += "- 此版本已针对多资产（BTC, SPX, EUR）进行了自适应标准化（Target = Vol Change）。\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Report generated at {output_file}")
    print("-" * 50)
    print(report)
    print("-" * 50)

if __name__ == "__main__":
    model_path = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\khaos_kan_model_final.pth'
    # Ensure we are loading the NEWEST model.
    # If shapes mismatch, it means we are loading an OLD checkpoint.
    # The training script overwrites 'khaos_kan_model_final.pth' at the end.
    # If train.py ran successfully, this file should be the 240-dim one.
     # The error logs suggest we are loading a 208-dim model.
     # This means training likely failed or didn't save properly?
    # Or we are pointing to the wrong file?
    # Let's check the timestamp of the model file
    import time
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        print(f"Model modified time: {time.ctime(mtime)}")
    else:
        # Fallback to best if final not found
        model_path = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\khaos_kan_best.pth'
        if os.path.exists(model_path):
            print(f"Final model not found, using best model: {model_path}")
        else:
            print("No model found!")
            sys.exit(1)
    
    data_dir = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed'
    output_file = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\04_项目文档\实验报告\KHAOS_KAN_Training_Report.md'
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    generate_report(model_path, data_dir, output_file)
