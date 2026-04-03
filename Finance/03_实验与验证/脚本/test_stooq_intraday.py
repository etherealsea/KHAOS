
import requests
import pandas as pd
import io

def test_stooq_interval(symbol, interval_code):
    # Stooq intervals: d=daily, w=weekly, m=monthly, q=quarterly, y=yearly
    # Unknown if 'h' or '5m' works on public interface.
    url = f"https://stooq.com/q/d/l/?s={symbol}&i={interval_code}"
    print(f"Testing {symbol} with interval '{interval_code}'...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.content.decode('utf-8')
            if "<html" in content.lower():
                print(f"  Failed (HTML blocked or invalid param)")
            elif not content.strip():
                print(f"  Failed (Empty)")
            else:
                try:
                    df = pd.read_csv(io.StringIO(content))
                    if 'Date' in df.columns or 'date' in df.columns:
                        print(f"  Success! Rows: {len(df)}")
                        print(df.head(2))
                        # Check time diff to verify interval
                        if len(df) > 1:
                            df['Date'] = pd.to_datetime(df['Date'])
                            diff = df['Date'].iloc[1] - df['Date'].iloc[0]
                            print(f"  Time Diff: {diff}")
                    else:
                        print(f"  Failed (Invalid Columns)")
                except Exception as e:
                    print(f"  Failed Parsing: {e}")
        else:
            print(f"  HTTP {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Test 'h' (hourly?), '300' (5m?), 'o' (other?)
    # Usually Stooq only allows 'd', 'w', 'm', 'q', 'y' publicly.
    test_stooq_interval('AAPL.US', 'h') 
    test_stooq_interval('AAPL.US', '5m')
    test_stooq_interval('AAPL.US', '300') 
