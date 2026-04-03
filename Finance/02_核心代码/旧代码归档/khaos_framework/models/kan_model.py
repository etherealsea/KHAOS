import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer (Simplified)
    Uses B-splines or simple basis functions for activation.
    """
    def __init__(self, input_dim, output_dim, num_basis=5, order=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        
        # Learnable weights for the basis functions
        # Shape: [output_dim, input_dim, num_basis]
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim, num_basis) * 0.1)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Base activation (SiLU-like) to help convergence
        self.base_weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 1. Base Linear Transformation (Residual connection equivalent)
        # [batch, in] @ [out, in].T -> [batch, out]
        base_out = F.linear(x, self.base_weight)
        
        # 2. Basis Function Expansion (Spline approximation)
        # Simple basis: sin(k * x), cos(k * x), etc.
        # For simplicity in this pure PyTorch version without external spline libs:
        # We use a set of RBFs or Polynomials
        
        # Let's use simple Chebyshev polynomials for efficiency
        # T_n(x)
        # Normalize x to [-1, 1] (assuming input is roughly standardized)
        x_norm = torch.tanh(x) 
        
        basis_outputs = []
        for k in range(self.num_basis):
            # T_k(x)
            if k == 0:
                b = torch.ones_like(x_norm)
            elif k == 1:
                b = x_norm
            else:
                # T_n = 2xT_{n-1} - T_{n-2}
                # Here we just use powers x^k for extreme simplicity in this demo
                b = x_norm.pow(k)
            basis_outputs.append(b)
            
        # Stack: [batch, input_dim, num_basis]
        basis_stack = torch.stack(basis_outputs, dim=-1)
        
        # Compute Weighted Sum
        # weights: [output_dim, input_dim, num_basis]
        # y[j] = sum_i sum_k (w[j,i,k] * basis[i,k])
        
        # Einstein summation: b=batch, i=input, o=output, k=basis
        spline_out = torch.einsum('bik,oik->bo', basis_stack, self.weights)
        
        return base_out + spline_out + self.bias

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input -> Hidden
        self.layers.append(KANLayer(input_dim, hidden_dim))
        
        # Hidden -> Hidden
        for _ in range(layers - 2):
            self.layers.append(KANLayer(hidden_dim, hidden_dim))
            
        # Hidden -> Output
        self.layers.append(KANLayer(hidden_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class KhaosKanPredictor(nn.Module):
    """
    Predicts Market Regime Parameters using KAN
    Input: [Hurst, Volatility, Bias_EMA, Slope, Entropy]
    Output: [Q_noise_scale, R_noise_scale, Momentum_Factor]
    """
    def __init__(self):
        super().__init__()
        # Input: 5 features (Updated: RSI removed, Bias_EMA & Entropy added)
        # Output: 3 EKF control parameters
        self.kan = KAN(input_dim=5, hidden_dim=8, output_dim=3)
        
    def forward(self, x):
        # Sigmoid output to bound parameters to (0, 1) range mostly
        out = torch.sigmoid(self.kan(x))
        return out
