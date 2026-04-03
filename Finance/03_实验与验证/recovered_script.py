import requests
import pandas as pd
import os
import time
from datetime import datetime

# Configuration
SYMBOL_BINANCE = 'BTCUSDT'
SYMBOL_COINGECKO = 'bitcoin'
INTERVAL = '1h'
LIMIT = 1000
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, f'BTC_USDT_{INTERVAL}.csv')
LOG_FILE = os.path.join(DATA_DIR, 'fetch_log.txt')

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now()} - {msg}\n")

def fetch_binance_data(symbol, interval, limit):
    """
    Fetch OHLCV data from Binance Public API.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    log(f"Attempting Binance fetch for {symbol}...")
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Binance returns: [Open Time, Open, High, Low, Close, Volume, Close Time, ...]
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Select relevant columns
        final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return final_df
        
    except Exception as e:
        log(f"Binance fetch failed: {e}")
        return None

def fetch_coingecko_data(coin_id, days):
    """
    Fetch OHLC data from CoinGecko (Volume is included but structure different).
    CoinGecko granularity depends on days: 1-90 days = hourly.
    """
    base_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    
    log(f"Attempting CoinGecko fetch for {coin_id}...")
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # CoinGecko OHLC: [time, open, high, low, close]
        # Note: No volume in OHLC endpoint, but good enough for price model testing if volume not critical yet.
        # Alternatively use /market_chart for volume but it gives separate arrays.
        
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['volume'] = 0.0 # Placeholder if using OHLC endpoint
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        log(f"CoinGecko fetch failed: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write("Starting data fetch...\n")

    # Try Binance
    df = fetch_binance_data(SYMBOL_BINANCE, INTERVAL, LIMIT)
    
    # Try CoinGecko if Binance failed
    if df is None:
        log("Switching to CoinGecko fallback...")
        df = fetch_coingecko_data(SYMBOL_COINGECKO, days=30)
    
    if df is not None:
        log(f"Data fetched successfully. Rows: {len(df)}")
        df.to_csv(OUTPUT_FILE, index=False)
        log(f"Data saved to {OUTPUT_FILE}")
    else:
        log("All data fetch attempts failed.")

if __name__ == "__main__":
    main()
