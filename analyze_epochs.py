import json

epoch_metrics_file = 'Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_balanced_v2/epoch_metrics.jsonl'
tf_metrics_file = 'Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_balanced_v2/per_timeframe_metrics.jsonl'

print("=== Overall Epoch Trend ===")
print(f"{'Epoch':<6} | {'Comp Score':<12} | {'Breakout Prec':<15} | {'Reversion Prec':<15}")
with open(epoch_metrics_file, 'r') as f:
    for line in f:
        d = json.loads(line)
        ep = d.get('epoch')
        comp = d.get('composite_score', 0)
        bp = d.get('breakout_precision', 0)
        rp = d.get('reversion_precision', 0)
        print(f"{ep:<6} | {comp:.4f}       | {bp:.4f}          | {rp:.4f}")

print("\n=== Best Epoch (5) Timeframe Details ===")
tf_stats = {}
with open(tf_metrics_file, 'r') as f:
    for line in f:
        d = json.loads(line)
        if d.get('epoch') == 5:
            tf = d.get('timeframe')
            comp = d.get('composite_score', 0)
            bp = d.get('breakout_precision', 0)
            rp = d.get('reversion_precision', 0)
            if tf not in tf_stats:
                tf_stats[tf] = {'comp': [], 'bp': [], 'rp': []}
            tf_stats[tf]['comp'].append(comp)
            tf_stats[tf]['bp'].append(bp)
            tf_stats[tf]['rp'].append(rp)

print(f"{'Timeframe':<10} | {'Avg Score':<10} | {'Avg Breakout Prec':<18} | {'Avg Reversion Prec':<18}")
for tf, s in tf_stats.items():
    c = sum(s['comp'])/len(s['comp'])
    b = sum(s['bp'])/len(s['bp'])
    r = sum(s['rp'])/len(s['rp'])
    print(f"{tf:<10} | {c:.4f}     | {b:.4f}             | {r:.4f}")

