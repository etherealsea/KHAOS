import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, weights={'p1': 0.1, 'p2': 0.0, 'p3': 0.1, 'p4': 0.1, 'p5': 0.1}):
        super().__init__()
        self.weights = weights
        self.main_loss_fn = nn.HuberLoss(delta=1.0)
        
    def forward(self, pred, target, physics_state):
        """
        pred: [B, 1] Log Volatility
        target: [B, 1] Log Realized Volatility
        physics_state: [B, 14]
        """
        # Ensure shapes match
        if pred.shape != target.shape:
            target = target.view_as(pred)
            
        main_loss = self.main_loss_fn(pred, target)
        
        # Unpack physics state
        H = physics_state[..., 0]
        Vol = physics_state[..., 1] # Current Volatility
        V_ekf = physics_state[..., 2]
        Res = physics_state[..., 3]
        Ent = physics_state[..., 4]
        
        # Volatility Physics
        
        # P3: Entropy -> Volatility
        # Hypothesis: High Entropy -> High Future Volatility
        # If Ent is high, we expect Pred_Vol to be high (less negative log vol)
        # Ent is [0, 1]. Pred is Log Vol (e.g. -5 to -2).
        # We can penalize anti-correlation?
        # Or simpler:
        # If Ent > 0.8, penalize if Pred < Current_Vol (Log)
        
        # Let's align gradients:
        # We want Grad(Pred) to be positive if Ent is high.
        # Loss = - Ent * Pred ? No, unbounded.
        
        # Let's enforce Consistency with Current Volatility
        # Pred should be related to Current Vol
        # But that's what the main loss learns.
        
        # Physics Constraint:
        # Energy Conservation / Clustering
        # If Current Vol is High, Future Vol should be High (Clustering).
        # This is already in data.
        
        # KHAOS:
        # High Entropy -> High Volatility Change (Expansion)
        # So if Ent is High, Pred (Change) should be positive.
        # Violation: Ent > 0.7 AND Pred < 0
        
        # P3 Violation: Entropy High but Predicted Vol Decrease
        # ReLU( (Ent - 0.7) * (0.0 - pred) )
        p3 = torch.relu((Ent - 0.7).unsqueeze(1) * (0.0 - pred)).mean()
        
        # P4: Residual -> Volatility
        # Large Residual -> High Volatility Change
        # Violation: Res > Threshold AND Pred < 0
        res_mag = Res.abs().unsqueeze(1)
        p4 = torch.relu((res_mag - 0.01) * (0.0 - pred)).mean()
        
        total_loss = main_loss + \
                     self.weights.get('p3', 0.0) * p3 + \
                     self.weights.get('p4', 0.0) * p4
                     
        return total_loss, {
            'main': main_loss.item(),
            'p3_ent_vol': p3.item(),
            'p4_res_vol': p4.item()
        }
