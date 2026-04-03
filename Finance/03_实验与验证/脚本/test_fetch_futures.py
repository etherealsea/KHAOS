import requests
import pandas as pd
import io
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def fetch_stooq(symbol, filename_prefix):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    print(f"Fetching Stooq (Daily): {symbol}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
             print(f"  Error: HTTP {response.status_code}")
             return False
             
        content = response.content.decode('utf-8')
        print(f"Content length: {len(content)}")
        
        if "<html" in content.lower():
            print(f"  Error: Stooq returned HTML (blocked/invalid symbol)")
            return False
            
        df = pd.read_csv(io.StringIO(content))
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            filepath = os.path.join(DATA_DIR, f"{filename_prefix}_1d.csv")
            df.to_csv(filepath)
            print(f"  Saved {filepath}: {len(df)} rows")
            return True
        else:
            print(f"  Error: Unexpected columns from Stooq for {symbol}")
            print(f"  Columns: {df.columns}")
            return False
    except Exception as e:
        print(f"  Error fetching {symbol} from Stooq: {e}")
        return False

# Try ES.F (E-mini S&P 500)
fetch_stooq('ES.F', 'ES_Future')
# Try ^SPX (S&P 500 Index) as fallback
fetch_stooq('^SPX', 'SPX_Index')
