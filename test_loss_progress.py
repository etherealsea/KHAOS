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

print("=== Epoch 0 / 100 (Progress 0.0) ===")
criterion.set_epoch(0, 100)
loss, rank, metrics = criterion(pred, aux_pred, target, aux_target, physics_state, event_flags, sigma, debug_info)
print(f"Loss: {loss.mean().item():.4f}, Rank: {rank.item():.4f}, Epsilon: {0.15 * torch.exp(torch.tensor(0.0)):.4f}, Lambda_Phys: 0.10")

print("\n=== Epoch 50 / 100 (Progress 0.5) ===")
criterion.set_epoch(50, 100)
loss, rank, metrics = criterion(pred, aux_pred, target, aux_target, physics_state, event_flags, sigma, debug_info)
print(f"Loss: {loss.mean().item():.4f}, Rank: {rank.item():.4f}, Epsilon: {0.15 * torch.exp(torch.tensor(-2.5)):.4f}, Lambda_Phys: 0.55")

print("\n=== Epoch 100 / 100 (Progress 1.0) ===")
criterion.set_epoch(100, 100)
loss, rank, metrics = criterion(pred, aux_pred, target, aux_target, physics_state, event_flags, sigma, debug_info)
print(f"Loss: {loss.mean().item():.4f}, Rank: {rank.item():.4f}, Epsilon: {0.15 * torch.exp(torch.tensor(-5.0)):.4f}, Lambda_Phys: 1.00")
