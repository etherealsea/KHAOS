import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResidualBlock(nn.Module):
    """
    带有门控残差连接的模块，用于防止深层网络退化。
    参考 Kimi 论文思路中的 Scaled/Gated Residuals。
    公式: output = x + gate * f(x) 
    其中 gate 可以是可学习标量，也可以是依赖于 x 的门控网络。
    """
    def __init__(self, d_model):
        super().__init__()
        # 可学习的缩放因子，初始化为极小值以实现类似 Zero-Init 的效果
        self.alpha = nn.Parameter(torch.zeros(1))
        # 或者使用依赖于输入的门控
        self.gate_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, residual):
        # 计算门控系数 (0 到 1 之间)
        gate = torch.sigmoid(self.gate_proj(x))
        # 结合可学习标量与门控
        return x + self.alpha * gate * residual

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, return_attn_weights=False):
        B, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn_weights)
        
        # Apply attention to V
        out = torch.matmul(attn, V)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        out = self.W_o(out)
        
        if return_attn_weights:
            return out, attn_weights
        return out

class AttentionResidualBlock(nn.Module):
    """
    结合多头自注意力与门控残差机制的模块
    """
    def __init__(self, d_model, num_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.gated_res1 = GatedResidualBlock(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.gated_res2 = GatedResidualBlock(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn_weights=False):
        # Attention sub-layer
        if return_attn_weights:
            attn_out, attn_weights = self.attn(self.norm1(x), mask, return_attn_weights=True)
        else:
            attn_out = self.attn(self.norm1(x), mask)
            
        attn_out = self.dropout(attn_out)
        x = self.gated_res1(x, attn_out)
        
        # FFN sub-layer
        ffn_out = self.ffn(self.norm2(x))
        ffn_out = self.dropout(ffn_out)
        x = self.gated_res2(x, ffn_out)
        
        if return_attn_weights:
            return x, attn_weights
        return x
