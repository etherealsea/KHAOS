import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from khaos.模型定义.kan import KHAOS_KAN
from khaos.核心引擎.physics import PhysicsLayer

def analyze_model_weights():
    print("=== PI-KAN Model Feature Analysis ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\khaos_kan_model_final.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Init Model
    input_dim = 8
    model = KHAOS_KAN(input_dim=input_dim, hidden_dim=64, output_dim=1, layers=3, grid_size=10, num_heads=4).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    feature_names = [
        "Hurst Exponent (Long Memory)", 
        "Current Volatility", 
        "EKF Velocity (Trend Inertia)", 
        "EKF Residual (Sudden Shock)", 
        "Permutation Entropy (Chaos/Disorder)", 
        "MLE Proxy (Lyapunov Edge of Chaos)",
        "Price Momentum (Log Return)",
        "EMA20 Divergence (Mean Reversion)"
    ]
    
    print("1. Layer 0 (Input to Hidden) Base Weight Analysis:")
    # Look at the base_weight of the first KAN layer.
    # shape: [hidden_dim, input_dim]
    base_weight = model.layers[0].base_weight.detach().cpu().numpy()
    
    # Calculate the mean absolute weight for each input feature across all hidden nodes
    feature_importance = np.mean(np.abs(base_weight), axis=0)
    
    # Normalize for display
    feature_importance = feature_importance / np.sum(feature_importance) * 100
    
    # Sort and display
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    for idx in sorted_indices:
        print(f"  - {feature_names[idx]:<40}: {feature_importance[idx]:.2f}%")
        
    print("\n2. Attention Mechanism Analysis:")
    # If possible, let's analyze the attention projections
    # W_q, W_k, W_v weights
    W_q_norm = torch.norm(model.attention_block.attn.W_q.weight).item()
    W_k_norm = torch.norm(model.attention_block.attn.W_k.weight).item()
    print(f"  - Query Matrix Norm: {W_q_norm:.2f}")
    print(f"  - Key Matrix Norm: {W_k_norm:.2f}")
    print("  (High norms indicate the model relies heavily on self-attention for temporal dependencies.)")

    print("\n3. Nonlinear Spline (KAN) Dynamics Analysis:")
    # Analyze spline scalers to see how much the model relies on non-linear B-splines vs linear base
    spline_scaler = model.layers[0].spline_scaler.detach().cpu().numpy()
    spline_importance = np.mean(np.abs(spline_scaler), axis=0)
    spline_importance = spline_importance / np.sum(spline_importance) * 100
    
    print("  Top features utilizing highly non-linear B-Spline transformations:")
    spline_sorted = np.argsort(spline_importance)[::-1]
    for idx in spline_sorted[:4]:
        print(f"  - {feature_names[idx]:<40}: {spline_importance[idx]:.2f}% non-linear activation")

if __name__ == "__main__":
    analyze_model_weights()
