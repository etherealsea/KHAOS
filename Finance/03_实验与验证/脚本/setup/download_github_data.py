import os
import requests
import pandas as pd
import gzip
import shutil

def download_file(url, save_path):
    print(f"Attempting download: {url}")
    try:
        # Use a user-agent
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            # Save raw content
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"  [SUCCESS] Saved to {save_path}")
            
            # Check if it's GZIP
            if url.endswith(".gz"):
                unzipped_path = save_path.replace(".gz", "")
                with gzip.open(save_path, 'rb') as f_in:
                    with open(unzipped_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  [SUCCESS] Unzipped to {unzipped_path}")
                os.remove(save_path) # Remove zip
                return True
            return True
        else:
            print(f"  [FAILED] Status Code: {response.status_code}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    return False

def fetch_real_data():
    base_dir = r"d:\Finance\Finance\model_research\data_raw"
    
    # Using jsdelivr as a mirror for GitHub to bypass potential blocks
    # Structure: https://cdn.jsdelivr.net/gh/{user}/{repo}@{branch}/{path}
    
    targets = [
        # Crypto: BTC (Using freqtrade sample data which is small and reliable)
        ("Crypto", "BTCUSD", [
            "https://cdn.jsdelivr.net/gh/freqtrade/freqtrade@develop/tests/testdata/BTC_USDT-1h.json" 
            # Note: This is JSON, we'll need to handle it. But let's try CSVs first.
        ]),
        # Let's stick to CSVs from FutureSharks via Mirror
        ("Index", "SPXUSD", [
            "https://cdn.jsdelivr.net/gh/FutureSharks/financial-data@master/data/stocks/histdata/sp500/DAT_ASCII_SPXUSD_M1_2015.csv",
            "https://cdn.jsdelivr.net/gh/FutureSharks/financial-data@master/data/stocks/histdata/sp500/DAT_ASCII_SPXUSD_M1_2016.csv"
        ]),
        # Forex: EURUSD
        ("Forex", "EURUSD", [
            "https://cdn.jsdelivr.net/gh/philipperemy/FX-1-Minute-Data@master/data/eurusd.csv.gz"
        ])
    ]
    
    for category, symbol, urls in targets:
        save_dir = os.path.join(base_dir, category, symbol.lower())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for url in urls:
            filename = url.split("/")[-1]
            save_path = os.path.join(save_dir, filename)
            
            if os.path.exists(save_path.replace(".gz", "")):
                print(f"Data for {symbol} already exists. Skipping.")
                continue
                
            if download_file(url, save_path):
                break

if __name__ == "__main__":
    fetch_real_data()
