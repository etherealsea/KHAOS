import requests
import pandas as pd
import io

def fetch_cdd(symbol, pair, exchange="Binance", timeframe="1h"):
    url = f"https://www.cryptodatadownload.com/cdd/{exchange}_{pair}_{timeframe}.csv"
    print(f"Fetching {url}...")
    try:
        response = requests.get(url, verify=False)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            content = response.content.decode('utf-8')
            print(f"Content length: {len(content)}")
            print(f"First 100 chars: {content[:100]}")
            
            # Try parsing
            lines = content.splitlines()
            header_row = 0
            for i, line in enumerate(lines[:10]):
                if "Date" in line and "Symbol" in line:
                    header_row = i
                    break
            print(f"Header row found at: {header_row}")
            
            df = pd.read_csv(io.StringIO(content), header=header_row)
            print(f"DataFrame shape: {df.shape}")
            
            if 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
                except:
                    # Fallback
                    df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
            if not df.empty:
                print("Success!")
                # Save to data dir
                import os
                DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
                filename = f"{pair}_{timeframe}.csv"
                filepath = os.path.join(DATA_DIR, filename)
                df.to_csv(filepath)
                print(f"Saved to {filepath}")
                return True
        else:
            print("Failed.")
    except Exception as e:
        print(f"Exception: {e}")

# fetch_cdd("BTC_USD", "BTCUSDT", "Binance", "d")
fetch_cdd("BTC_USD", "BTCUSDT", "Binance", "1h")
