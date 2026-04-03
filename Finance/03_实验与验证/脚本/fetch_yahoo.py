import urllib.request
import json
import pandas as pd
from datetime import datetime
import os

def fetch_es():
    print("Fetching ES=F...")
    url = "https://query1.finance.yahoo.com/v8/finance/chart/ES=F?range=60d&interval=5m"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'date': [datetime.fromtimestamp(ts) for ts in timestamps],
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote['volume']
        })
        
        df.dropna(inplace=True)
        
        out_dir = r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\Index"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ESUSD_5m.csv")
        
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    fetch_es()
