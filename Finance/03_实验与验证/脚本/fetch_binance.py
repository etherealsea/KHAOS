import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

def fetch_binance(symbol, years=5):
    print(f"Fetching {symbol} from Binance for {years} years...")
    url = "https://api.binance.com/api/v3/klines"
    interval = '5m'
    limit = 1000
    
    # Calculate start time (5 years ago)
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (years * 365 * 24 * 60 * 60 * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'startTime': current_start,
            'endTime': end_time
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code != 200:
                print(f"Error {res.status_code}: {res.text}")
                break
                
            data = res.json()
            if not data:
                break
                
            all_data.extend(data)
            
            current_start = data[-1][0] + 1
            print(f"Fetched up to {pd.to_datetime(data[-1][0], unit='ms')} (Total: {len(all_data)})")
            
            if len(data) < limit:
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(2)
            
    print()
    if not all_data:
        print("No data fetched.")
        return None
        
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
    df = pd.DataFrame(all_data, columns=cols)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
        
    return df

def main():
    save_dir = r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\Crypto"
    os.makedirs(save_dir, exist_ok=True)
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = fetch_binance(symbol, years=2) # 2 years to be safe and quick
        if df is not None:
            # save as 5m
            base_name = symbol.replace("USDT", "USD")
            out_path = os.path.join(save_dir, f"{base_name}_5m.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
