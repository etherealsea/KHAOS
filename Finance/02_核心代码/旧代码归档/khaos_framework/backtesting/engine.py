"""
VeighNa-Lite 回测引擎
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable
from ..objects import BarData, TickData, OrderData, TradeData, Direction, Offset, Status, Exchange, Interval

class BacktestingEngine:
    def __init__(self):
        self.symbol = ""
        self.exchange = Exchange.LOCAL
        self.start = None
        self.end = None
        self.rate = 0
        self.slippage = 0
        self.size = 1
        self.pricetick = 0
        self.capital = 1_000_000
        self.mode = "BAR" # BAR or TICK

        self.strategy_class = None
        self.strategy = None
        self.history_data = []
        
        self.limit_orders = {}
        self.active_limit_orders = {}
        self.limit_order_count = 0
        
        self.trades = {}
        self.trade_count = 0
        
        self.logs = []
        
        self.daily_results = {}
        self.daily_df = None

    def set_parameters(self, vt_symbol: str, interval: str, start: datetime, end: datetime, rate: float, slippage: float, size: float, pricetick: float, capital: int):
        self.symbol = vt_symbol
        self.interval = interval
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start
        self.end = end
        self.capital = capital

    def add_strategy(self, strategy_class, setting: dict):
        self.strategy_class = strategy_class
        self.strategy_setting = setting

    def load_data(self):
        # 这里的逻辑被简化，实际应从数据库或CSV加载
        pass
        
    def set_data(self, data: list):
        """直接注入数据"""
        self.history_data = data

    def run_backtesting(self):
        if not self.history_data:
            self.output("数据为空，无法回测")
            return

        self.strategy = self.strategy_class(self, self.strategy_class.__name__, self.symbol, self.strategy_setting)
        self.strategy.on_init()
        self.strategy.on_start()
        
        self.output("开始回测")
        
        for data in self.history_data:
            if isinstance(data, BarData):
                self.strategy.on_bar(data)
                self.cross_limit_order(data)
            elif isinstance(data, TickData):
                self.strategy.on_tick(data)
                self.cross_limit_order(data)
                
        self.strategy.on_stop()
        self.output("回测结束")

    def cross_limit_order(self, data):
        """
        撮合限价单
        """
        if isinstance(data, BarData):
            buy_cross_price = data.low_price
            sell_cross_price = data.high_price
            buy_best_price = data.open_price
            sell_best_price = data.open_price
        else:
            buy_cross_price = data.ask_price_1
            sell_cross_price = data.bid_price_1
            buy_best_price = data.ask_price_1
            sell_best_price = data.bid_price_1
            
        # 遍历活动订单
        for orderid, order in list(self.active_limit_orders.items()):
            # 检查是否成交
            traded = False
            
            # 买单
            if order.direction == Direction.LONG:
                if order.price >= buy_cross_price:
                    traded = True
                    trade_price = min(order.price, buy_best_price) # 以对手价成交或限价
            # 卖单
            else:
                if order.price <= sell_cross_price:
                    traded = True
                    trade_price = max(order.price, sell_best_price)
            
            if traded:
                trade_price = round(trade_price, 8) # 简单修约
                
                self.trade_count += 1
                tradeid = str(self.trade_count)
                trade = TradeData(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    orderid=order.orderid,
                    tradeid=tradeid,
                    direction=order.direction,
                    offset=order.offset,
                    price=trade_price,
                    volume=order.volume,
                    datetime=data.datetime
                )
                
                self.trades[tradeid] = trade
                
                # 推送成交
                self.strategy.on_trade(trade)
                
                # 订单完成
                order.traded = order.volume
                order.status = Status.ALLTRADED
                if orderid in self.active_limit_orders:
                    del self.active_limit_orders[orderid]
                    
                self.strategy.on_order(order)

    def send_order(self, strategy, direction: Direction, offset: Offset, price: float, volume: float, stop: bool = False, lock: bool = False, net: bool = False):
        self.limit_order_count += 1
        orderid = str(self.limit_order_count)
        
        order = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.NOTTRADED,
            datetime=datetime.now() # 模拟时间
        )
        
        self.active_limit_orders[orderid] = order
        self.limit_orders[orderid] = order
        
        return [orderid]

    def cancel_order(self, strategy, vt_orderid):
        if vt_orderid in self.active_limit_orders:
            order = self.active_limit_orders[vt_orderid]
            order.status = Status.CANCELLED
            del self.active_limit_orders[vt_orderid]
            self.strategy.on_order(order)

    def output(self, msg):
        print(f"{datetime.now()} \t {msg}")
        self.logs.append(msg)
        
    def calculate_result(self):
        """
        计算回测结果 (简化版)
        """
        if not self.trades:
            return pd.DataFrame()
            
        trade_df = pd.DataFrame([t.__dict__ for t in self.trades.values()])
        return trade_df

    def calculate_statistics(self, df=None):
        if df is None:
            df = self.calculate_result()
            
        if df.empty:
            self.output("无成交记录，无法计算统计")
            return {}
            
        # 简单统计
        total_trades = len(df)
        
        # 这里的逻辑非常简化，仅为了演示
        # 实际需要计算逐日盈亏
        
        return {"total_trades": total_trades}

class CtaTemplate:
    """
    策略基类
    """
    author = ""
    parameters = []
    variables = []

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        self.cta_engine = cta_engine
        self.strategy_name = strategy_name
        self.vt_symbol = vt_symbol
        self.setting = setting
        
        self.pos = 0 # 持仓
        
        # 设置参数
        for key in self.parameters:
            if key in setting:
                setattr(self, key, setting[key])

    def on_init(self):
        """初始化"""
        pass

    def on_start(self):
        """启动"""
        pass

    def on_stop(self):
        """停止"""
        pass

    def on_tick(self, tick: TickData):
        """Tick更新"""
        pass

    def on_bar(self, bar: BarData):
        """K线更新"""
        pass
        
    def on_trade(self, trade: TradeData):
        """成交推送"""
        # 更新持仓
        if trade.direction == Direction.LONG:
            if trade.offset == Offset.OPEN:
                self.pos += trade.volume
            else:
                self.pos -= trade.volume
        else:
            if trade.offset == Offset.OPEN:
                self.pos -= trade.volume
            else:
                self.pos += trade.volume
                
        self.put_event()

    def on_order(self, order: OrderData):
        """订单推送"""
        pass

    def buy(self, price, volume, stop=False, lock=False, net=False):
        return self.cta_engine.send_order(self, Direction.LONG, Offset.OPEN, price, volume, stop, lock, net)

    def sell(self, price, volume, stop=False, lock=False, net=False):
        return self.cta_engine.send_order(self, Direction.SHORT, Offset.CLOSE, price, volume, stop, lock, net)

    def short(self, price, volume, stop=False, lock=False, net=False):
        return self.cta_engine.send_order(self, Direction.SHORT, Offset.OPEN, price, volume, stop, lock, net)

    def cover(self, price, volume, stop=False, lock=False, net=False):
        return self.cta_engine.send_order(self, Direction.LONG, Offset.CLOSE, price, volume, stop, lock, net)

    def put_event(self):
        """推送UI更新"""
        pass

    def write_log(self, msg):
        self.cta_engine.output(msg)
        
    def load_bar(self, days):
        """加载历史数据"""
        pass
