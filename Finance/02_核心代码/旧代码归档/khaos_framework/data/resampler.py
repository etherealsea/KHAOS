import pandas as pd
from src.khaos_framework.objects import BarData, Interval

class Resampler:
    @staticmethod
    def resample_bars(bars, interval_minutes):
        """
        将 1分钟 BarData 列表重采样为指定周期
        """
        if not bars:
            return []
            
        # 转换为 DataFrame 加速处理
        df = pd.DataFrame([b.__dict__ for b in bars])
        df.set_index('datetime', inplace=True)
        
        # Resample 规则
        ohlc_dict = {
            'open_price': 'first',
            'high_price': 'max',
            'low_price': 'min',
            'close_price': 'last',
            'volume': 'sum',
            'symbol': 'first',
            'exchange': 'first'
        }
        
        rule = f"{interval_minutes}T"
        if interval_minutes >= 1440:
            rule = f"{interval_minutes//1440}D"
        elif interval_minutes >= 60:
            rule = f"{interval_minutes//60}H"
            
        resampled_df = df.resample(rule).agg(ohlc_dict).dropna()
        
        # 转回 BarData 列表
        new_bars = []
        for dt, row in resampled_df.iterrows():
            bar = BarData(
                symbol=row['symbol'],
                exchange=row['exchange'],
                datetime=dt,
                interval=Interval.MINUTE, # 暂时标记，实际代表新周期
                open_price=row['open_price'],
                high_price=row['high_price'],
                low_price=row['low_price'],
                close_price=row['close_price'],
                volume=row['volume']
            )
            new_bars.append(bar)
            
        return new_bars
