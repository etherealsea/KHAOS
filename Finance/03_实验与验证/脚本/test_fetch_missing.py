import requests
import pandas as pd
import io

def test_stooq(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    print(f"Testing {symbol}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.content.decode('utf-8')
            if "<html" in content.lower():
                print(f"  {symbol}: Failed (HTML blocked)")
            elif not content.strip():
                print(f"  {symbol}: Failed (Empty content)")
            else:
                try:
                    df = pd.read_csv(io.StringIO(content))
                    if 'Date' in df.columns and len(df) > 10:
                        print(f"  {symbol}: Success! ({len(df)} rows)")
                    else:
                        print(f"  {symbol}: Failed (Invalid CSV/No data columns: {df.columns})")
                except Exception as e:
                    print(f"  {symbol}: Failed parsing CSV: {e}")
        else:
            print(f"  {symbol}: HTTP {response.status_code}")
    except Exception as e:
        print(f"  {symbol}: Error: {e}")

candidates = [
    'GC.F', 'GC.F.US', 'XAUUSD', # Gold
    'CL.F', 'CL.F.US', 'XTIUSD', # Oil
    '^VIX', 'VIX.XO', 'VIX', 'VIX.M' # VIX
]

for c in candidates:
    test_stooq(c)
