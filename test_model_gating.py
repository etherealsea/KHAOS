import torch
import sys
sys.path.append("/workspace/Finance/02_核心代码/源代码")
from khaos.模型定义.kan import KHAOS_KAN

model = KHAOS_KAN(input_dim=14, horizon_count=1, arch_version='iterA4_multiscale')
x = torch.randn(256, 5, 14)
try:
    main_pred, info = model(x, return_debug=True)
    print(f"main_pred shape: {main_pred.shape}")
    print("SUCCESS KAN GATING")
except Exception as e:
    import traceback
    traceback.print_exc()
