import json
from collections import defaultdict

metrics_file = 'Finance/02_核心代码/模型权重备份/teacher_first_ashare/shortT_balanced_v2/per_timeframe_metrics.jsonl'

best_epoch = 5
timeframe_stats = defaultdict(lambda: {'precision_long': [], 'precision_short': [], 'composite': []})
asset_stats = defaultdict(lambda: {'composite': []})

try:
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('epoch') == best_epoch:
                tf = data.get('timeframe')
                asset = data.get('asset', 'unknown')
                
                metrics = data.get('metrics', {})
                # precision formats: typically precision_0 (short) and precision_1 (long) or similar
                # Let's check what keys are in metrics
                comp = metrics.get('composite', 0)
                prec = metrics.get('precision', [0, 0]) # Assuming [short, long] or [class0, class1]
                
                timeframe_stats[tf]['composite'].append(comp)
                if isinstance(prec, list) and len(prec) >= 2:
                    timeframe_stats[tf]['precision_short'].append(prec[0])
                    timeframe_stats[tf]['precision_long'].append(prec[1])
                
                asset_stats[asset]['composite'].append(comp)

    print("=== Best Epoch (5) Analysis ===")
    print(f"{'Timeframe':<10} | {'Avg Comp':<10} | {'Avg Prec (Short)':<15} | {'Avg Prec (Long)':<15}")
    for tf, stats in timeframe_stats.items():
        avg_comp = sum(stats['composite']) / len(stats['composite']) if stats['composite'] else 0
        avg_ps = sum(stats['precision_short']) / len(stats['precision_short']) if stats['precision_short'] else 0
        avg_pl = sum(stats['precision_long']) / len(stats['precision_long']) if stats['precision_long'] else 0
        print(f"{tf:<10} | {avg_comp:.4f}     | {avg_ps:.4f}          | {avg_pl:.4f}")

    print("\n=== Top 5 Assets by Composite Score ===")
    asset_avg = []
    for asset, stats in asset_stats.items():
        avg_comp = sum(stats['composite']) / len(stats['composite']) if stats['composite'] else 0
        asset_avg.append((asset, avg_comp))
    
    asset_avg.sort(key=lambda x: x[1], reverse=True)
    for asset, score in asset_avg[:5]:
        print(f"Asset: {asset:<10} | Score: {score:.4f}")

except Exception as e:
    print(f"Error: {e}")

