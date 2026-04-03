
import requests
import warnings
warnings.filterwarnings("ignore")

def check_cdd_minute():
    # Try different timeframes for CDD
    base_url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_{}.csv"
    timeframes = ['minute', '5m', '15m']
    
    for tf in timeframes:
        url = base_url.format(tf)
        print(f"Testing {tf} at {url}...")
        try:
            response = requests.get(url, verify=False)
            if response.status_code == 200:
                print(f"  SUCCESS: {tf} exists! Length: {len(response.content)}")
                print(response.content[:200])
            else:
                print(f"  Failed: {tf} (HTTP {response.status_code})")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    check_cdd_minute()
