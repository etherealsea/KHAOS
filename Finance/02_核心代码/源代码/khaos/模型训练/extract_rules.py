import os
import sys
import glob

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from khaos.模型定义.kan import KHAOS_KAN
from khaos.数据处理.data_loader import create_rolling_datasets
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES

def build_model_from_checkpoint(checkpoint, device):
    args = checkpoint.get('args', {})
    return KHAOS_KAN(
        input_dim=len(checkpoint.get('feature_names', PHYSICS_FEATURE_NAMES)),
        hidden_dim=args.get('hidden_dim', 64),
        output_dim=2,
        layers=args.get('layers', 3),
        grid_size=args.get('grid_size', 10),
        num_heads=4
    ).to(device), args

def print_kernel_rules(kernel_name, kernel_scores, features, feature_names):
    threshold = np.percentile(kernel_scores, 90)
    labels = (kernel_scores >= threshold).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree.fit(features, labels)
    rules = export_text(tree, feature_names=feature_names)
    print(f"=== {kernel_name} 核规则提取 (Top 10% >= {threshold:.4f}) ===")
    print(rules)

def extract_rules():
    print("=== Extracting Highly Interpretable Rules from PI-KAN ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\khaos_kan_model_final.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model, model_args = build_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()
    
    feature_names = checkpoint.get('feature_names', PHYSICS_FEATURE_NAMES)
    training_dir = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\training_ready'
    sample_files = sorted(glob.glob(os.path.join(training_dir, '*.csv')))[:8]
    
    all_physics = []
    all_preds = []
    all_targets = []
    for sample_file in sample_files:
        print(f"Loading sample data from {os.path.basename(sample_file)} to probe model behavior...")
        train_ds, _ = create_rolling_datasets(
            sample_file,
            window_size=model_args.get('window_size', 20),
            horizon=model_args.get('horizon', 4),
            fast_full=True
        )
        loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch_x, batch_y, _, _, _, _ in loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                psi_t = batch_x[:, -1, :].cpu().numpy()
                all_physics.append(psi_t)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())
                if sum(chunk.shape[0] for chunk in all_physics) > 30000:
                    break
        if sum(chunk.shape[0] for chunk in all_physics) > 30000:
            break

    if not all_physics:
        print("No samples collected for rule extraction.")
        return

    X = np.vstack(all_physics)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    breakout_scores = y_pred[:, 0]
    reversion_scores = np.maximum(y_pred[:, 1], 0.0)
    print()
    print_kernel_rules("Breakout", breakout_scores, X, feature_names)
    print_kernel_rules("Reversion", reversion_scores, X, feature_names)

    print("\n=== Kernel Responsibility Check ===")
    high_chaos = (X[:, 4] > np.percentile(X[:, 4], 80)) & (X[:, 5] > np.percentile(X[:, 5], 60))
    compression_setup = X[:, 13] > np.percentile(X[:, 13], 80)
    strong_dislocation = (np.abs(X[:, 3]) > np.percentile(np.abs(X[:, 3]), 80)) & (np.abs(X[:, 7]) > np.percentile(np.abs(X[:, 7]), 80))
    entropy_turn = (X[:, 8] > np.percentile(X[:, 8], 65)) & (X[:, 9] > np.percentile(X[:, 9], 65))
    if np.sum(high_chaos) > 0:
        print(
            f"高混沌样本 -> Breakout 均值 {np.mean(breakout_scores[high_chaos]):.4f}, "
            f"Reversion 均值 {np.mean(reversion_scores[high_chaos]):.4f}"
        )
    if np.sum(compression_setup) > 0:
        print(
            f"压缩样本 -> Breakout 均值 {np.mean(breakout_scores[compression_setup]):.4f}, "
            f"Reversion 均值 {np.mean(reversion_scores[compression_setup]):.4f}"
        )
    if np.sum(strong_dislocation) > 0:
        print(
            f"强残差+强 EMA 偏离样本 -> Reversion 均值 {np.mean(reversion_scores[strong_dislocation]):.4f}, "
            f"Breakout 均值 {np.mean(breakout_scores[strong_dislocation]):.4f}"
        )
    if np.sum(entropy_turn) > 0:
        print(
            f"熵回升样本 -> Reversion 均值 {np.mean(reversion_scores[entropy_turn]):.4f}, "
            f"Breakout 均值 {np.mean(breakout_scores[entropy_turn]):.4f}"
        )

    print("\n=== Pine Semantic Alignment Check ===")
    overbought = X[:, 7] > np.percentile(X[:, 7], 90)
    oversold = X[:, 7] < np.percentile(X[:, 7], 10)
    positive_dislocation = (X[:, 3] + X[:, 7]) > 0
    negative_dislocation = (X[:, 3] + X[:, 7]) < 0
    if np.sum(overbought & positive_dislocation) > 0:
        print(
            "超买+正向失衡 -> "
            f"Reversion 均值 {np.mean(reversion_scores[overbought & positive_dislocation]):.4f}"
        )
    if np.sum(oversold & negative_dislocation) > 0:
        print(
            "超卖+负向失衡 -> "
            f"Reversion 均值 {np.mean(reversion_scores[oversold & negative_dislocation]):.4f}"
        )

    print("\n=== Target Consistency Snapshot ===")
    print(
        f"Breakout 预测/目标相关性: {np.corrcoef(breakout_scores, y_true[:, 0])[0, 1]:.4f}"
    )
    print(
        f"Reversion 预测/目标相关性: {np.corrcoef(reversion_scores, y_true[:, 1])[0, 1]:.4f}"
    )

if __name__ == "__main__":
    extract_rules()
