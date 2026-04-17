import torch
from Finance.02_核心代码.源代码.khaos.模型定义.kan import KHAOS_KAN

model = KHAOS_KAN(input_dim=14, horizon_count=1, arch_version='iterA4_multiscale')
x = torch.randn(256, 5, 14)
attn = torch.randn(256, 1)
try:
    model(x, return_debug=True)
except Exception as e:
    import traceback
    traceback.print_exc()
