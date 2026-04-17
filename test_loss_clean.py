import torch
import sys
sys.path.append("/workspace/Finance/02_核心代码/源代码")
from khaos.模型训练.loss import PhysicsLoss

criterion = PhysicsLoss()
batch = 256
pred = torch.randn(batch, 2, 2).abs()
aux_pred = torch.randn(batch, 2)
target = torch.randn(batch, 2)
aux_target = torch.randn(batch, 2)
physics_state = torch.randn(batch, 14)
event_flags = torch.randint(0, 2, (batch, 4)).float()
sigma = torch.ones(batch)

debug_info = {
    'bear_score': torch.randn(batch, 2).abs(),
    'bull_score': torch.randn(batch, 2).abs(),
    'public_reversion_score': torch.randn(batch, 2).abs()
}

loss, rank, metrics = criterion(pred, aux_pred, target, aux_target, physics_state, event_flags, sigma, debug_info)
assert 'directional_violation' not in metrics, "directional_violation should be removed"
print("SUCCESS LOSS CLEAN")
