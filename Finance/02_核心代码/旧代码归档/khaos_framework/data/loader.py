import os
import pandas as pd
from datetime import datetime
from src.khaos_framework.objects import BarData, Exchange, Interval

class DataLoader:
    @staticmethod
    def load_histdata_csv(file_path, symbol, exchange=Exchange.LOCAL):
        """
        Load HistData.com generic ASCII format:
        20230102 180000;1826.837000;1827.337000;1826.617000;1826.637000;0
        """
        bars = []
        try:
            df = pd.read_csv(
                file_path, 
                sep=';', 
                header=None, 
                names=['dt', 'open', 'high', 'low', 'close', 'volume']
            )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
            
        df['datetime'] = pd.to_datetime(df['dt'], format='%Y%m%d %H%M%S')
        
        for row in df.itertuples():
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row.datetime,
                interval=Interval.MINUTE,
                open_price=row.open,
                high_price=row.high,
                low_price=row.low,
                close_price=row.close,
                volume=row.volume
            )
            bars.append(bar)
        return bars

    @staticmethod
    def load_investing_csv(file_path, symbol, exchange=Exchange.LOCAL):
        """
        Load Investing.com / Yahoo Finance style CSV
        Date,Open,High,Low,Close,Volume
        """
        bars = []
        try:
            df = pd.read_csv(file_path)
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                return []
                
            for row in df.itertuples():
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=row.datetime,
                    interval=Interval.MINUTE, # Assuming minute data
                    open_price=row.open,
                    high_price=row.high,
                    low_price=row.low,
                    close_price=row.close,
                    volume=row.volume if hasattr(row, 'volume') else 0
                )
                bars.append(bar)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        return bars

    @staticmethod
    def load_all_years(base_dir, symbol_path_map, years_filter=None):
        """
        Strictly load ONLY existing local files. No downloading.
        """
        all_data = {}
        
        for symbol, rel_path in symbol_path_map.items():
            full_dir = os.path.join(base_dir, rel_path)
            bars = []
            
            if not os.path.exists(full_dir):
                print(f"Directory not found: {full_dir} - Skipping {symbol}")
                continue
                
            files = sorted([f for f in os.listdir(full_dir) if f.endswith('.csv')])
            
            for f in files:
                if years_filter:
                    match = False
                    for y in years_filter:
                        if y in f:
                            match = True
                            break
                    if not match:
                        continue
                        
                print(f"Loading {symbol} from {f}...")
                file_path = os.path.join(full_dir, f)
                
                # Detect format based on filename or content
                if "DAT_ASCII" in f:
                    year_bars = DataLoader.load_histdata_csv(file_path, symbol)
                else:
                    year_bars = DataLoader.load_investing_csv(file_path, symbol)
                    
                bars.extend(year_bars)
                
            if bars:
                all_data[symbol] = bars
            else:
                print(f"No data loaded for {symbol}")
            
        return all_data
