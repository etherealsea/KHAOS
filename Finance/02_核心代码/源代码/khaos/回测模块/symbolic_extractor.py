import torch
import numpy as np
import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from khaos.模型定义.kan import KHAOS_KAN
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES
from khaos.数据处理.data_loader import create_rolling_datasets

def extract_piecewise_formulas(model, feature_names):
    print("=== Extracting Non-Linear/Piecewise Formulas from KHAOS-KAN ===\n")
    x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
    head_specs = [
        ('Breakout', model.breakout_head.layers[0]),
        ('Reversion', model.reversion_head.layers[0])
    ]
    for kernel_name, layer0 in head_specs:
        device = layer0.base_weight.device
        print(f"Detected {kernel_name} Rules:")
        for i, name in enumerate(feature_names):
            dummy = torch.zeros(100, len(feature_names), device=device)
            dummy[:, i] = x_test.squeeze().to(device)
            with torch.no_grad():
                bases = layer0.b_splines(dummy)
                if layer0.enable_standalone_scale_spline:
                    scaled_weight = layer0.spline_weight * layer0.spline_scaler.unsqueeze(-1)
                    y = torch.einsum("bid,oid->bo", bases, scaled_weight)
                else:
                    y = torch.einsum("bid,oid->bo", bases, layer0.spline_weight)
                response = y.abs().mean(dim=1).cpu().numpy()
                x_np = x_test.squeeze().cpu().numpy()
                gradients = np.gradient(response, x_np)
                max_grad_idx = np.argmax(np.abs(gradients))
                threshold = x_np[max_grad_idx]
                print(
                    f"  [{name}] 阈值 ≈ {threshold:.2f} | 响应放大倍数 {np.max(response) / (np.mean(response) + 1e-8):.2f}"
                )

def extract_attention_rules(model, dummy_input):
    print("\n=== Extracting Temporal Attention Distribution ===")
    model.eval()
    with torch.no_grad():
        _, attn_info = model(dummy_input, return_attn_weights=True)
    attn_weights = attn_info['attn']
    pool_weights = attn_info['pool'][0].cpu().numpy()
    last_step_attn = attn_weights[0, :, -1, :].mean(dim=0).cpu().numpy()
    seq_len = len(last_step_attn)
    
    print("\nAttention Weights Distribution (How the model looks at the past):")
    recent_weight = np.sum(last_step_attn[-3:])
    mid_weight = np.sum(last_step_attn[-8:-3])
    far_weight = np.sum(last_step_attn[:-8])
    
    print(f"  -> 最近期 (t-0 to t-2) 权重占比: {recent_weight*100:.1f}%")
    print(f"  -> 中期   (t-3 to t-7) 权重占比: {mid_weight*100:.1f}%")
    print(f"  -> 远期记忆 (t-8 之前) 权重占比: {far_weight*100:.1f}%")
    print(f"  -> 汇聚池最近期权重占比: {np.sum(pool_weights[-3:]) * 100:.1f}%")
    
    max_past_idx = np.argmax(last_step_attn)
    lag = seq_len - 1 - max_past_idx
    
    print("\n[Attention Interpretation]")
    print(f"1. 序列全局感知: 模型并未只看 {lag} 根 K 线！它吸收了长达 {seq_len} 周期的完整上下文（远期记忆占比 {far_weight*100:.1f}%）。")
    print(f"2. 局部触发锚点: t-{lag} 位置拥有最高单点权重 ({last_step_attn[max_past_idx]*100:.1f}%)，这代表网络将 t-{lag} 时刻的物理突变（如剧烈放量或残差脉冲）作为了确认当前相变的【核心触发器 (Trigger Anchor)】。")
    print(f"-> Pine Script 落地建议: 统一使用 20 周期计算核心物理背景，仅保留 2~3 周期平滑或锚点扫描来表达短时确认。")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_names = PHYSICS_FEATURE_NAMES
    
    model_path = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\khaos_kan_model_final.pth'
    model = KHAOS_KAN(input_dim=len(feature_names), hidden_dim=64, output_dim=2, layers=3, grid_size=10, num_heads=2).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded trained model weights successfully.\n")
            except:
                pass
    
    extract_piecewise_formulas(model, feature_names)
    training_dir = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\training_ready'
    sample_files = sorted(glob.glob(os.path.join(training_dir, '*.csv')))
    if sample_files:
        sample_file = sample_files[0]
        train_ds, _ = create_rolling_datasets(sample_file, window_size=20, horizon=4, fast_full=True)
        sample_seq = train_ds[0][0].unsqueeze(0).to(device)
        print(f"\nUsing real sequence from {os.path.basename(sample_file)} for attention extraction.")
    else:
        sample_seq = torch.randn(1, 20, len(feature_names)).to(device)
    extract_attention_rules(model, sample_seq)
