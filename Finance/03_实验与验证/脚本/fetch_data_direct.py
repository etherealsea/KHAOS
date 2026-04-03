
import pandas as pd
import os
import time
from datetime import datetime
import yfinance as yf
import warnings
import requests

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def save_dataframe(df, filename):
    if df is not None and not df.empty:
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath)
        print(f"    Saved {filename}: {len(df)} rows")
    else:
        print(f"    Failed/Empty: {filename}")

def fetch_yfinance(symbol, name, category):
    """
    Fetches data from Yahoo Finance for multiple timeframes.
    Timeframes: 5m, 15m, 1h, 1d.
    Derived: 4h (resampled from 1h).
    Limits:
        - 1d: Max history (10y+)
        - 1h: Max 730 days (~2y)
        - 5m, 15m: Max 60 days
    """
    print(f"\nFetching {category}: {name} ({symbol})...")
    
    # Define tasks: (interval, period, suffix)
    tasks = [
        ('1d', '10y', '1d'),
        ('1h', '730d', '1h'),   # Max for hourly
        ('15m', '60d', '15m'),  # Max for intraday
        ('5m', '60d', '5m')     # Max for intraday
    ]
    
    # Create a session with User-Agent
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

    for interval, period, suffix in tasks:
        try:
            print(f"  Requesting {interval} (Period: {period})...")
            # Fetch data using Ticker with session might not work directly in all yf versions, 
            # but yf.download allows it? Ticker is better for history.
            # Let's try just setting the session or sleeping more.
            # Actually, just sleeping and retrying is often enough.
            
            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(period=period, interval=interval)
            
            if df is None or df.empty:
                print(f"    Warning: No data for {symbol} {interval}")
                # Retry once with simple download
                try:
                    print("    Retrying with yf.download...")
                    df = yf.download(symbol, period=period, interval=interval, progress=False, session=session)
                except:
                    pass
            
            if df is None or df.empty:
                 print(f"    Failed {symbol} {interval}")
                 continue

            # Standardize columns
            # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
            # We just need OHLCV
            # Sometimes MultiIndex columns if using download
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[keep_cols]
            
            df.index.name = 'Date'
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Save direct timeframe
            save_dataframe(df, f"{name}_{suffix}.csv")
            
            # Handle 4h Resampling (from 1h data)
            if interval == '1h':
                try:
                    # Resample to 4h
                    # Rule: Open from first, High=max, Low=min, Close=last, Volume=sum
                    ohlc_dict = {
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }
                    df_4h = df.resample('4h').agg(ohlc_dict).dropna()
                    if not df_4h.empty:
                        save_dataframe(df_4h, f"{name}_4h.csv")
                except Exception as e:
                    print(f"    Error resampling 4h: {e}")
                    
            time.sleep(2) # Increased delay
            
        except Exception as e:
            print(f"    Error fetching {symbol} {interval}: {e}")
            time.sleep(5) # Wait longer on error

def main():
    print("Starting Multi-Timeframe Data Fetch (yfinance)...")
    print("Target: 5m, 15m, 1h, 4h, 1d")
    print("Note: 5m/15m limited to last 60 days by Yahoo. 1h limited to ~2 years.")
    
    # 1. Stocks
    stocks = [
        ('AAPL', 'AAPL'), 
        ('MSFT', 'MSFT'), 
        ('GOOGL', 'GOOGL'), 
        ('AMZN', 'AMZN'), 
        ('TSLA', 'TSLA')
    ]
    for s, n in stocks:
        fetch_yfinance(s, n, "Stock")

    # 2. Forex
    # Yahoo symbol for EURUSD is 'EURUSD=X'
    fetch_yfinance('EURUSD=X', 'EURUSD', "Forex")
    
    # 3. Crypto
    # Yahoo symbol for BTC is 'BTC-USD'
    fetch_yfinance('BTC-USD', 'BTC', "Crypto")
    
    # 4. Futures Proxies (Indices/ETFs)
    futures = [
        ('^GSPC', 'SPX_Index'), # S&P 500
        ('^NDX', 'NDX_Index'),  # Nasdaq 100
        ('^DJI', 'DJI_Index'),  # Dow Jones
        ('GLD', 'Gold_ETF'),    # Gold Proxy
        ('USO', 'Oil_ETF')      # Oil Proxy
    ]
    for s, n in futures:
        fetch_yfinance(s, n, "Futures_Proxy")
        
    # 5. Options Proxy (Volatility)
    fetch_yfinance('VIXY', 'VIX_ETF', "Options_Proxy")
    
    print("\nFetch Complete.")

if __name__ == "__main__":
    main()
