import yfinance as yf
import pandas as pd
import os

def fetch_es_futures():
    print("Fetching ES=F (E-mini S&P 500) from Yahoo Finance...")
    symbol = "ES=F"
    
    # We want 5m data. yfinance limit is 60 days. 
    # But wait, to train properly across different timeframes, 
    # if we only have 60 days of 5m, it might be small. Let's try 1m for 7 days, 5m for 60d.
    # Actually, we can fetch 5m for 60d and that will give us 60 * 24 * 12 = 17,280 rows.
    # We can also fetch 1h for 730d (2 years) = 730 * 24 = 17,520 rows.
    
    save_dir = r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\Index"
    os.makedirs(save_dir, exist_ok=True)
    
    ticker = yf.Ticker(symbol)
    
    # 5m
    try:
        df_5m = ticker.history(period="60d", interval="5m")
        if not df_5m.empty:
            df_5m.reset_index(inplace=True)
            df_5m.columns = [c.lower() for c in df_5m.columns]
            df_5m = df_5m[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df_5m.rename(columns={'datetime': 'date'}, inplace=True)
            out_path = os.path.join(save_dir, "ESUSD_5m.csv")
            df_5m.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df_5m)} rows)")
    except Exception as e:
        print(f"Error 5m: {e}")
        
    # 1h
    try:
        df_1h = ticker.history(period="730d", interval="1h")
        if not df_1h.empty:
            df_1h.reset_index(inplace=True)
            df_1h.columns = [c.lower() for c in df_1h.columns]
            df_1h = df_1h[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df_1h.rename(columns={'datetime': 'date'}, inplace=True)
            out_path = os.path.join(save_dir, "ESUSD_1h.csv")
            df_1h.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df_1h)} rows)")
    except Exception as e:
        print(f"Error 1h: {e}")
        
    # 1d
    try:
        df_1d = ticker.history(period="10y", interval="1d")
        if not df_1d.empty:
            df_1d.reset_index(inplace=True)
            df_1d.columns = [c.lower() for c in df_1d.columns]
            df_1d = df_1d[['date', 'open', 'high', 'low', 'close', 'volume']]
            out_path = os.path.join(save_dir, "ESUSD_1d.csv")
            df_1d.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df_1d)} rows)")
    except Exception as e:
        print(f"Error 1d: {e}")

if __name__ == "__main__":
    fetch_es_futures()
