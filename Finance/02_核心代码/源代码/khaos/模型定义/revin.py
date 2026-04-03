import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization (RevIN)
        消除多周期时间序列分布偏移的利器。
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # [1, 1, num_features]
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # x shape: [Batch, Seq_len, Features]
        # Calculate mean and std over the Seq_len dimension (dim=1)
        # 转换为 float32 避免混合精度下的溢出
        x_fp32 = x.float()
        self.mean = torch.mean(x_fp32, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x_fp32, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x_dtype = x.dtype
        x = x.float()
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.float()
            x = x + self.affine_bias.float()
        return x.to(x_dtype)

    def _denormalize(self, x):
        x_dtype = x.dtype
        x = x.float()
        if self.affine:
            x = x - self.affine_bias.float()
            x = x / (self.affine_weight.float() + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x.to(x_dtype)