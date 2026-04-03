import numpy as np
import torch
from khaos_framework.backtesting.engine import CtaTemplate, BarData, OrderData, TradeData
from khaos_quant_engine.khaos_ekf_core import ExtendedKalmanFilter

class KhaosEkfKanStrategy(CtaTemplate):
    """
    KHAOS + EKF + KAN 深度融合策略
    """
    author = "Trae AI"
    
    # 策略参数
    hurst_window = 20
    fixed_size = 1
    
    # KAN 模型实例 (将被注入)
    kan_model_instance = None
    
    parameters = ["hurst_window", "fixed_size"]
    variables = ["ekf_price", "regime_pred"]
    
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.ekf = ExtendedKalmanFilter()
        
        # 使用注入的模型或默认
        self.kan = self.kan_model_instance
            
        self.prices = []
        
    def on_init(self):
        self.write_log("策略初始化")
        
    def on_start(self):
        self.write_log("策略启动")
        
    def on_bar(self, bar: BarData):
        # 1. 数据收集
        self.prices.append(bar.close_price)
        if len(self.prices) > 100:
            self.prices.pop(0)
            
        if len(self.prices) < self.hurst_window + 2:
            return
            
        # 2. 特征工程 (与训练时保持一致)
        # [Hurst, Vol, RSI, Slope]
        
        # Slice last window
        window = self.hurst_window
        closes = np.array(self.prices[-window:])
        log_prices = np.log(closes)
        returns = np.diff(log_prices)
        
        # 1. Volatility
        vol = np.std(returns) * np.sqrt(252 * 1440)
        
        # 2. Slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0] / closes[0] * 10000
        
        # 3. RSI
        gains = returns[returns > 0]
        losses = -returns[returns < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 4. Hurst Proxy
        hurst = 0.5 + np.tanh(slope) * 0.3
        
        # 3. KAN 预测
        if self.kan:
            features = torch.tensor([[hurst, vol, rsi/100.0, slope]], dtype=torch.float32)
            with torch.no_grad():
                # RobustKAN returns tensor
                preds = self.kan(features).numpy()[0]
                
            # Preds: [Q_scale, R_scale, Momentum_Factor]
            q_scale = preds[0] * 0.01
            r_scale = preds[1] * 0.1
            # momentum = preds[2] 
            
            # Update EKF
            self.ekf.Q[0,0] = abs(q_scale)
            self.ekf.R[0,0] = abs(r_scale)
        
        # 4. EKF Update
        log_price = np.log(bar.close_price)
        self.ekf.predict(hurst=hurst)
        self.ekf.update(log_price)
        
        smooth_price, velocity = self.ekf.get_state()
        
        # 5. Trading Logic
        residual = log_price - smooth_price
        threshold = 2.0 * np.std(self.prices[-20:]) / bar.close_price if bar.close_price > 0 else 0.001
        
        if self.pos == 0:
            if residual < -threshold:
                self.buy(bar.close_price, self.fixed_size)
            elif residual > threshold:
                self.short(bar.close_price, self.fixed_size)
        elif self.pos > 0:
            if residual > 0:
                self.sell(bar.close_price, self.pos)
        elif self.pos < 0:
            if residual < 0:
                self.cover(bar.close_price, abs(self.pos))

    def calculate_hurst(self):
        return 0.5
