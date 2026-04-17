# KHAOS-PIKAN Ansatz Hard-Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the neural network architecture to enforce physical constraints directly via distance function gating (Ansatz Hard-Wiring) in `kan.py`, and remove all corresponding loss-based physical penalties from `loss.py`.

**Architecture:** We will introduce two unlearnable, detached gates (`directional_gate_hard` and `compression_gate_hard`) inside the forward pass of `kan.py` to filter the output logits. Concurrently, we will clean up `loss.py` to remove legacy hard/soft margins (`directional_violation`, `public_below_directional_violation`, curriculum logic).

**Tech Stack:** PyTorch, Python 3

---

### Task 1: Refactor `kan.py` to inject detached Physics Gates

**Files:**
- Modify: `/workspace/Finance/02_核心代码/源代码/khaos/模型定义/kan.py`
- Test: `/workspace/test_model.py` (Local smoke test script)

- [ ] **Step 1: Read the existing `_forward_itera3` method**
Run: `cat -n /workspace/Finance/02_核心代码/源代码/khaos/模型定义/kan.py | sed -n '540,630p'` to understand where `breakout_event_logits` and `reversion_event_logits` are generated.

- [ ] **Step 2: Modify `kan.py` to calculate and apply detached gates**
In `kan.py`, around line 560-600, right before the `_aggregate_horizon_outputs` is called, we need to extract `physics_state` (if available, we might need to pass it into `_forward_itera3` and `_forward_itera2`, or use the existing contextual features) and create the detached gates. Wait, `physics_state` is NOT passed into `kan.py`'s forward currently. It only takes `x` (which is `[batch, horizon, feature_dim]`). Let's check what `x` contains.

*Self-Correction*: `x` in `kan.py` is the raw feature tensor (which includes physics_state at specific indices, e.g., index 13 is compression). But actually, we can just use the `bull_score_h` and `bear_score_h` to construct the directional gate, and we don't strictly need `physics_state` if it's too complex to extract here, or we can extract `compression` from `x[..., 0, 13]` if we know the feature dimension map.

Wait, let's look at `kan.py`'s `forward` method:
`x` is passed in. `x` is `[batch, seq_len, features]`. In `data_loader.py`, the physics features are at the end. But the easiest and most robust gate is the **Directional Gate** (based on bear/bull scores) and the **Reversion Residual Gate**.

Let's modify `kan.py` to implement the `directional_gate_hard` and `public_reversion_gate_hard` as detached scaling factors:

```python
# Create test script to verify forward pass
cat << 'EOF' > test_model_gating.py
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
EOF
```

- [ ] **Step 3: Edit `kan.py` to apply the detached hard-wiring**

We need to edit the lines around `reversion_event_logits = torch.maximum(...)` in `kan.py`.
Replace the `_forward_itera3` block (and similarly for `_forward_itera2` if needed, but we focus on `iterA4_multiscale`/`iterA5_multiscale` logic):

```python
# In `kan.py`, locate the reversion_event_logits computation
# and add the detached gating logic.
```
*Wait*, a simpler approach: use `sed` or Python script to replace the exact lines in `kan.py` where `reversion_event_logits` is calculated.

```python
import sys
file_path = "/workspace/Finance/02_核心代码/源代码/khaos/模型定义/kan.py"
with open(file_path, "r") as f:
    content = f.read()

old_code = """            reversion_residual = public_reversion_gate * torch.relu(
                public_reversion_score_h - directional_reversion_h + 0.15
            )
            reversion_event_logits = torch.maximum(
                directional_floor_h + 0.10,
                directional_reversion_h + reversion_residual,
            )"""

new_code = """            # Ansatz Hard-Wiring: Unlearnable physical distance gating
            # The directional_gate determines if there's a clear trend.
            # We detach it so the network cannot cheat by minimizing it to avoid loss.
            directional_gate_hard = torch.sigmoid(10.0 * (torch.abs(bull_score_h - bear_score_h) - 0.05)).detach()
            
            reversion_residual = public_reversion_gate * torch.relu(
                public_reversion_score_h - directional_reversion_h + 0.15
            )
            
            # Apply the detached gate to the public reversion output.
            # If there's no clear direction (directional_gate_hard ~ 0), the reversion event is strictly bounded by directional_reversion_h.
            gated_residual = directional_gate_hard * reversion_residual
            
            reversion_event_logits = directional_reversion_h + gated_residual"""

content = content.replace(old_code, new_code)
with open(file_path, "w") as f:
    f.write(content)
```

- [ ] **Step 4: Run the test to ensure KAN still forwards correctly**
Run: `python test_model_gating.py`
Expected: `SUCCESS KAN GATING`

---

### Task 2: Strip physical penalties from `loss.py`

**Files:**
- Modify: `/workspace/Finance/02_核心代码/源代码/khaos/模型训练/loss.py`
- Test: `/workspace/test_loss.py`

- [ ] **Step 1: Write the failing test**

```python
cat << 'EOF' > test_loss_clean.py
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
# Verify that the keys we are going to remove are no longer in metrics
assert 'directional_violation' not in metrics, "directional_violation should be removed"
print("SUCCESS LOSS CLEAN")
EOF
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python test_loss_clean.py`
Expected: `AssertionError: directional_violation should be removed`

- [ ] **Step 3: Write minimal implementation (Clean up `loss.py`)**
We need to remove the `Curriculum Learning` (progress, epsilon, lambda_phys) and the `directional_violation` block.

```python
import sys
file_path = "/workspace/Finance/02_核心代码/源代码/khaos/模型训练/loss.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace curriculum logic
old_curr = """        # Progress for Curriculum Learning (0.0 -> 1.0)
        progress = self._get_progress()
        
        # Soft margin epsilon (shrinks from 0.15 -> 0.0 as training progresses)
        epsilon = 0.15 * math.exp(-5.0 * progress)
        
        # Adaptive physical loss weight (starts at 0.1, ramps up to 1.0)
        # Allows data-driven optimization early on, enforces physics strictly later
        lambda_phys = 0.1 + 0.9 * progress"""
new_curr = """        # Curriculum Learning and Soft Margins removed in favor of Ansatz Hard-Wiring"""
content = content.replace(old_curr, new_curr)

# Replace EDL kl_penalty_weight
content = content.replace("kl_penalty_weight=0.03 * lambda_phys", "kl_penalty_weight=0.03")
content = content.replace("kl_penalty_weight=0.05 * lambda_phys", "kl_penalty_weight=0.05")

# Remove directional violation logic
import re
pattern = re.compile(r"if debug_info is not None:.*?l_dict = {", re.DOTALL)
replacement = """l_dict = {"""
content = pattern.sub(replacement, content)

# Remove from l_dict
old_ldict = """        l_dict = {
            'breakout_gap': breakout_event_gap_loss.mean().item(),
            'reversion_gap': reversion_event_gap_loss.mean().item(),
            'directional_violation': directional_violation.mean().item(),
            'public_below_directional_violation': public_below_directional_violation.mean().item(),
        }"""
new_ldict = """        l_dict = {
            'breakout_gap': breakout_event_gap_loss.mean().item(),
            'reversion_gap': reversion_event_gap_loss.mean().item(),
        }"""
content = content.replace(old_ldict, new_ldict)

# Replace main_loss and rank_loss combination
old_comb = """        # Combine with lambda_phys
        # 1) Core physics/EDL loss
        main_loss = main_loss + lambda_phys * (breakout_event_gap_loss + reversion_event_gap_loss).unsqueeze(-1)
        
        # 2) Hard constraints converted to Soft constraints (Rank Loss)
        rank_loss = lambda_phys * (
            0.15 * directional_violation.mean() +
            0.20 * public_below_directional_violation.mean()
        )
        
        # 3) Additional physics penalties (all multiplied by lambda_phys to ensure curriculum)
        direction_consistency_loss = (
            self._direction_margin_loss(bear_score, bull_score, bear_context, margin=0.12 - epsilon) +
            self._direction_margin_loss(bull_score, bear_score, bull_context, margin=0.12 - epsilon)
        )

        l_dict['direction_consistency'] = direction_consistency_loss.item()
        rank_loss += lambda_phys * 0.10 * direction_consistency_loss"""
new_comb = """        # Core physics/EDL loss
        main_loss = main_loss + (breakout_event_gap_loss + reversion_event_gap_loss).unsqueeze(-1)
        rank_loss = torch.tensor(0.0, device=main_loss.device)"""
content = content.replace(old_comb, new_comb)

# Remove the rest of lambda_phys usage
content = content.replace("lambda_phys * ", "")

with open(file_path, "w") as f:
    f.write(content)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python test_loss_clean.py`
Expected: `SUCCESS LOSS CLEAN`

- [ ] **Step 5: Commit**
```bash
git add /workspace/Finance/02_核心代码/源代码/khaos/模型定义/kan.py
git add /workspace/Finance/02_核心代码/源代码/khaos/模型训练/loss.py
git commit -m "refactor: apply Ansatz Hard-Wiring in KAN and strip physical penalties from Loss"
```