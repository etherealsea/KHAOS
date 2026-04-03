from openbb import obb
import pandas as pd
from datetime import datetime, timedelta

symbol = 'BTC-USD'
provider = 'yfinance'
start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
interval = '1d'

print(f"Testing fetch for {symbol} via {provider}...")
try:
    df = obb.crypto.price.historical(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        provider=provider
    ).to_dataframe()
    
    print("Fetch returned object.")
    if df is None:
        print("df is None")
    elif df.empty:
        print("df is empty")
    else:
        print(f"df shape: {df.shape}")
        print(df.head())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
