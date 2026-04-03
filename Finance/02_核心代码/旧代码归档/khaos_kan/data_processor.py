import pandas as pd
import os

def resample_data(df, interval):
    """
    Resample 1m or higher frequency data to target interval.
    interval: '5min', '15min', '1h', '4h', '1d'
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
    # Logic
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample
    resampled = df.resample(interval).agg(agg_dict).dropna()
    return resampled.reset_index()

def process_multi_timeframe(file_path, output_dir):
    """
    Load a CSV (ideally 1m), and generate 5m, 15m, 1h, 4h, 1d versions.
    If input is e.g. 1h, can only generate 4h, 1d.
    """
    try:
        df = pd.read_csv(file_path)
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        
        # Determine source interval estimate
        # Check time diff between first two rows
        if 'date' in df.columns:
            t1 = pd.to_datetime(df['date'].iloc[0])
            t2 = pd.to_datetime(df['date'].iloc[1])
        elif 'datetime' in df.columns:
            t1 = pd.to_datetime(df['datetime'].iloc[0])
            t2 = pd.to_datetime(df['datetime'].iloc[1])
        else:
            print(f"Skipping {file_path}: No date column")
            return []

        diff_min = (t2 - t1).total_seconds() / 60
        
        targets = {
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        generated_files = []
        base_name = os.path.basename(file_path).replace('.csv', '')
        # Remove existing interval suffix if any (e.g. _1h)
        for suffix in ['_1m', '_5m', '_15m', '_1h', '_4h', '_1d']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
                
        for label, minutes in targets.items():
            if diff_min <= minutes:
                # Can resample
                # Pandas offset alias
                alias = label.replace('m', 'min').replace('d', 'D').replace('h', 'H')
                
                # If source is same as target, just copy?
                # Resampling ensures regularity
                resampled = resample_data(df.copy(), alias)
                
                if len(resampled) > 100: # Min length check
                    out_name = f"{base_name}_{label}.csv"
                    out_path = os.path.join(output_dir, out_name)
                    resampled.to_csv(out_path, index=False)
                    generated_files.append(out_path)
                    
        return generated_files
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []
