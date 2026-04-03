import yfinance as yf
import pandas as pd
import os
import time

def fetch_data(symbol, category, interval='1h', period='730d'):
    """
    Fetch data from Yahoo Finance and save to processed directory.
    """
    print(f"Fetching {category}: {symbol} ({interval})...")
    
    try:
        # Create session with headers
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"  Warning: No data for {symbol}")
            return
            
        # Clean columns
        df.columns = [c.lower() for c in df.columns]
        
        # Keep only OHLCV
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        
        # Reset index to get 'date' or 'datetime' column
        df.reset_index(inplace=True)
        
        # Rename index col to 'date' if needed
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'date'}, inplace=True)
            
        # Save
        # Clean symbol for filename
        safe_symbol = symbol.replace('=X', '').replace('-USD', '').replace('^', '').replace('=F', '')
        
        save_dir = os.path.join(r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed', category)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{safe_symbol}_{interval}.csv"
        save_path = os.path.join(save_dir, filename)
        
        df.to_csv(save_path, index=False)
        print(f"  Saved {len(df)} rows to {save_path}")
        
        time.sleep(1) # Rate limit
        
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")

def main():
    # Asset List
    assets = {
        'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD'],
        'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'CHFUSD=X'],
        'Index': ['^GSPC', '^NDX', '^DJI', '^RUT'],
        'Commodity': ['GC=F', 'CL=F', 'SI=F', 'HG=F', 'NG=F']
    }
    
    intervals = ['1h'] # Focus on 1h for now as it's the main timeframe for KHAOS-KAN
    
    for cat, symbols in assets.items():
        for sym in symbols:
            for interval in intervals:
                fetch_data(sym, cat, interval=interval)

if __name__ == "__main__":
    main()
