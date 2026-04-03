import glob
import os
import sys

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_SRC = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\源代码'
BEST_PATH = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\iter9_round\khaos_kan_best.pth'
FINAL_PATH = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\iter9_round\khaos_kan_model_final.pth'
TRAINING_DIR = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\training_ready'

sys.path.append(PROJECT_SRC)

from khaos.模型定义.kan import KHAOS_KAN
from khaos.数据处理.data_loader import create_rolling_datasets
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES

def build_model(checkpoint, device):
    args = checkpoint.get('args', {})
    model = KHAOS_KAN(
        input_dim=len(checkpoint.get('feature_names', PHYSICS_FEATURE_NAMES)),
        hidden_dim=args.get('hidden_dim', 64),
        output_dim=2,
        layers=args.get('layers', 3),
        grid_size=args.get('grid_size', 10),
        num_heads=4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, args

def compute_event_metrics(scores, event_flags, hard_negative_flags):
    scores = np.asarray(scores, dtype=np.float64)
    event_flags = np.asarray(event_flags, dtype=bool)
    hard_negative_flags = np.asarray(hard_negative_flags, dtype=bool)
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
        hard_negative_rate = np.mean(pred[hard_negative_flags]) if np.any(hard_negative_flags) else 0.0
        event_rate = np.mean(pred[event_flags]) if np.any(event_flags) else 0.0
        candidate = {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'event_rate': float(event_rate),
            'hard_negative_rate': float(hard_negative_rate),
            'signal_frequency': float(np.mean(pred)),
            'label_frequency': label_frequency
        }
        if best is None or candidate['f1'] > best['f1'] or (
            candidate['f1'] == best['f1'] and candidate['hard_negative_rate'] < best['hard_negative_rate']
        ):
            best = candidate
    return best

def extract_dataset_probe(model, args, device):
    sample_files = sorted(glob.glob(os.path.join(TRAINING_DIR, '*.csv')))[:8]
    all_physics = []
    all_preds = []
    all_targets = []
    all_flags = []
    for sample_file in sample_files:
        train_ds, _ = create_rolling_datasets(sample_file, window_size=args.get('window_size', 20), horizon=args.get('horizon', 10), fast_full=True)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch_x, batch_y, _, _, _, batch_flags in loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                all_physics.append(batch_x[:, -1, :].cpu().numpy())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())
                all_flags.append(batch_flags.numpy())
                if sum(chunk.shape[0] for chunk in all_physics) > 30000:
                    break
        if sum(chunk.shape[0] for chunk in all_physics) > 30000:
            break
    return np.vstack(all_physics), np.vstack(all_preds), np.vstack(all_targets), np.vstack(all_flags)

def kernel_rules(kernel_scores, features, feature_names):
    threshold = np.percentile(kernel_scores, 90)
    labels = (kernel_scores >= threshold).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree.fit(features, labels)
    return threshold, export_text(tree, feature_names=feature_names)

def summarize(breakout_eval, reversion_eval):
    breakout_desc = '更偏中期高质量爆发过滤器' if breakout_eval['precision'] >= 0.62 else '仍在平衡中期爆发覆盖与纯度'
    reversion_desc = '更偏确认型回归核' if reversion_eval['precision'] >= 0.45 else '仍偏广覆盖回归核'
    return breakout_desc, reversion_desc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_ckpt = torch.load(BEST_PATH, map_location=device, weights_only=False)
    final_ckpt = torch.load(FINAL_PATH, map_location=device, weights_only=False)
    print('=== Iter9 Checkpoints ===')
    print('BEST', {'val_loss': best_ckpt.get('val_loss'), 'best_score': best_ckpt.get('best_score'), 'metrics': best_ckpt.get('metrics')})
    print('FINAL', {'epochs': final_ckpt.get('args', {}).get('epochs'), 'feature_count': len(final_ckpt.get('feature_names', [])), 'env': final_ckpt.get('env')})
    model, args = build_model(best_ckpt, device)
    X, y_pred, y_true, y_flags = extract_dataset_probe(model, args, device)
    feature_names = best_ckpt.get('feature_names', PHYSICS_FEATURE_NAMES)
    breakout_scores = y_pred[:, 0]
    reversion_scores = np.maximum(y_pred[:, 1], 0.0)
    breakout_eval = compute_event_metrics(breakout_scores, y_flags[:, 0] > 0.5, y_flags[:, 2] > 0.5)
    reversion_eval = compute_event_metrics(reversion_scores, y_flags[:, 1] > 0.5, y_flags[:, 3] > 0.5)
    breakout_desc, reversion_desc = summarize(breakout_eval, reversion_eval)
    print('\n=== 直观评估 ===')
    print('breakout_eval', breakout_eval)
    print('reversion_eval', reversion_eval)
    print('breakout_desc', breakout_desc)
    print('reversion_desc', reversion_desc)
    print('\n=== Probe Correlations ===')
    print('breakout_corr', float(np.corrcoef(breakout_scores, y_true[:, 0])[0, 1]))
    print('reversion_corr', float(np.corrcoef(reversion_scores, y_true[:, 1])[0, 1]))
    print('\n=== Rules ===')
    b_th, b_rules = kernel_rules(breakout_scores, X, feature_names)
    r_th, r_rules = kernel_rules(reversion_scores, X, feature_names)
    print('breakout_threshold', b_th)
    print(b_rules)
    print('reversion_threshold', r_th)
    print(r_rules)

if __name__ == '__main__':
    main()
