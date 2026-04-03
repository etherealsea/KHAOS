import backtrader as bt
import numpy as np
import torch
from src.khaos_framework.models.kan_robust import RobustKAN
from src.khaos_framework.models.kan_model import KhaosKanPredictor # Fallback if needed
# Fix import path since khaos_ekf_core was moved to src/khaos_quant_engine/khaos_ekf_core.py 
# Wait, I need to check where khaos_ekf_core.py is. It was moved to src/khaos_quant_engine/
from src.khaos_quant_engine.khaos_ekf_core import ExtendedKalmanFilter

class KhaosBacktraderStrategy(bt.Strategy):
    params = (
        ('hurst_window', 20),
        ('kan_model', None),  # Pass the trained model instance
        ('print_log', False),
    )

    def log(self, txt, dt=None):
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        # Indicators
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        
        # KHAOS Components
        self.ekf = ExtendedKalmanFilter()
        self.kan = self.params.kan_model
        
        # State buffers for calculation
        self.prices = [] 
        
        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        # 1. Collect Data
        price = self.dataclose[0]
        self.prices.append(price)
        
        if len(self.prices) > 100:
            self.prices.pop(0)
            
        if len(self.prices) < self.params.hurst_window + 2:
            return

        # 2. Feature Engineering (On-the-fly)
        # Note: In a super optimized engine, this would be pre-calculated or incremental.
        # Here we mirror the logic from run_comprehensive_pipeline.py for consistency.
        
        window = self.params.hurst_window
        closes = np.array(self.prices[-window:])
        log_prices = np.log(closes)
        returns = np.diff(log_prices)
        
        # Volatility
        vol = np.std(returns) * np.sqrt(252 * 1440) # Assuming minute data
        
        # Slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0] / closes[0] * 10000
        
        # RSI Proxy
        gains = returns[returns > 0]
        losses = -returns[returns < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Hurst Proxy
        hurst = 0.5 + np.tanh(slope) * 0.3
        
        # 3. KAN Prediction
        if self.kan:
            features = torch.tensor([[hurst, vol, rsi/100.0, slope]], dtype=torch.float32)
            with torch.no_grad():
                # Assuming RobustKAN output
                preds = self.kan(features).numpy()[0]
            
            q_scale = preds[0] * 0.01
            r_scale = preds[1] * 0.1
            
            # Dynamic EKF Tuning
            self.ekf.Q[0,0] = abs(q_scale)
            self.ekf.R[0,0] = abs(r_scale)

        # 4. EKF Update
        log_price = np.log(price)
        self.ekf.predict(hurst=hurst)
        self.ekf.update(log_price)
        
        smooth_price, velocity = self.ekf.get_state()
        
        # 5. Trading Logic
        residual = log_price - smooth_price
        # Adaptive threshold
        threshold = 2.0 * np.std(self.prices[-20:]) / price if price > 0 else 0.001
        
        if self.order:
            return

        if not self.position:
            if residual < -threshold:
                self.log(f'BUY CREATE, {price:.2f}')
                self.order = self.buy()
            elif residual > threshold:
                self.log(f'SELL CREATE, {price:.2f}')
                self.order = self.sell()
        else:
            # Simple Mean Reversion Exit
            if self.position.size > 0:
                if residual > 0:
                    self.log(f'CLOSE LONG, {price:.2f}')
                    self.order = self.close()
            elif self.position.size < 0:
                if residual < 0:
                    self.log(f'CLOSE SHORT, {price:.2f}')
                    self.order = self.close()
