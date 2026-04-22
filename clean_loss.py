import re

with open('/workspace/Finance/02_核心代码/源代码/khaos/模型训练/loss.py', 'r') as f:
    content = f.read()

# 1. Remove p3, p4, p6, p7 from LOSS_WEIGHT_PRESETS
content = re.sub(r"^\s*'p[3467]':.*?\n", "", content, flags=re.MULTILINE)

# 2. Remove calculations
calc_old = """        p3 = torch.relu(Ent - 0.7) * torch.relu(0.0 - pred_vol[..., 1])
        res_score = Res.abs() / sigma_ref
        ema_score = EMA_Div.abs() / sigma_ref
        alignment = (Res + EMA_Div).abs() / (Res.abs() + EMA_Div.abs() + 1e-6)
        reversion_setup = torch.relu(res_score - 1.0) * torch.relu(ema_score - 0.5) * alignment
        p4 = reversion_setup * torch.relu(0.0 - pred[..., 1, 1])
        p6_lyapunov = torch.relu(MLE) * torch.relu(0.0 - pred_vol[..., 1])
        vol_mean = Vol.mean()
        p7_csd = torch.relu(H - 0.6) * torch.relu(vol_mean - Vol) * torch.relu(MLE - 0.1) * torch.relu(0.0 - pred_vol[..., 1])
        continuation_bias = torch.relu(H - 0.55) * torch.relu(MLE) * torch.relu(Vol - vol_mean)
        weak_dislocation = torch.relu(0.5 - reversion_setup)
        p7_false_reversion = continuation_bias * weak_dislocation * pred_rev[..., 1]
"""
calc_new = """        res_score = Res.abs() / sigma_ref
        ema_score = EMA_Div.abs() / sigma_ref
        alignment = (Res + EMA_Div).abs() / (Res.abs() + EMA_Div.abs() + 1e-6)
        reversion_setup = torch.relu(res_score - 1.0) * torch.relu(ema_score - 0.5) * alignment
"""
if calc_old in content:
    content = content.replace(calc_old, calc_new)
else:
    print("Warning: Calculation block not found!")

# 3. Remove from logs
log_old = """            'p3_ent_vol': p3.mean().item(),
            'p4_reversion_setup': p4.mean().item(),
            'p6_mle_chaos': p6_lyapunov.mean().item(),
            'p7_csd': p7_csd.mean().item(),
            'p7_false_reversion': p7_false_reversion.mean().item(),
"""
if log_old in content:
    content = content.replace(log_old, "")
else:
    print("Warning: Log block not found!")

with open('/workspace/Finance/02_核心代码/源代码/khaos/模型训练/loss.py', 'w') as f:
    f.write(content)

print("loss.py cleaned successfully!")
