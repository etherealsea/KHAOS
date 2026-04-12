import re

log_file = '/workspace/logs/teacher_first_ashare/shortT_balanced_v2/shortT_balanced_v2.log'

in_epoch_5 = False
assets = []

with open(log_file, 'r') as f:
    for line in f:
        if "========== EPOCH 5/20 ==========" in line:
            in_epoch_5 = True
            continue
        if "========== EPOCH 6/20 ==========" in line:
            break
            
        if in_epoch_5 and "->" in line and "val=" in line:
            m = re.search(r'->\s+([0-9a-zA-Z_]+)\.csv.*precision=([\d\.]+)/([\d\.]+)\s+composite=([\d\.]+)', line)
            if m:
                asset_tf = m.group(1)
                p1 = float(m.group(2))
                p2 = float(m.group(3))
                comp = float(m.group(4))
                assets.append({'name': asset_tf, 'p1': p1, 'p2': p2, 'comp': comp})

# Sort by composite score ascending
assets.sort(key=lambda x: x['comp'])

print("=== Bottom 5 Worst Performing Asset/Timeframes in Epoch 5 ===")
for a in assets[:5]:
    print(f"Asset: {a['name']:<12} | Composite: {a['comp']:.4f} | Prec: {a['p1']:.4f} / {a['p2']:.4f}")
