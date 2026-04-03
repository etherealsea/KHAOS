"""
VeighNa-Lite 核心对象定义
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class Direction(Enum):
    LONG = "多"
    SHORT = "空"
    NET = "净"

class Offset(Enum):
    OPEN = "开"
    CLOSE = "平"
    CLOSETODAY = "平今"
    CLOSEYESTERDAY = "平昨"

class Status(Enum):
    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"

class Exchange(Enum):
    BINANCE = "BINANCE"
    LOCAL = "LOCAL"

class Interval(Enum):
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    TICK = "tick"

@dataclass
class TickData:
    symbol: str
    exchange: Exchange
    datetime: datetime
    name: str = ""
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    last_price: float = 0
    last_volume: float = 0
    limit_up: float = 0
    limit_down: float = 0
    
    bid_price_1: float = 0
    bid_price_2: float = 0
    bid_price_3: float = 0
    bid_price_4: float = 0
    bid_price_5: float = 0
    
    ask_price_1: float = 0
    ask_price_2: float = 0
    ask_price_3: float = 0
    ask_price_4: float = 0
    ask_price_5: float = 0
    
    bid_volume_1: float = 0
    bid_volume_2: float = 0
    bid_volume_3: float = 0
    bid_volume_4: float = 0
    bid_volume_5: float = 0

    ask_volume_1: float = 0
    ask_volume_2: float = 0
    ask_volume_3: float = 0
    ask_volume_4: float = 0
    ask_volume_5: float = 0

@dataclass
class BarData:
    symbol: str
    exchange: Exchange
    datetime: datetime
    interval: Interval = None
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0

@dataclass
class OrderData:
    symbol: str
    exchange: Exchange
    orderid: str
    type: str = ""
    direction: Direction = None
    offset: Offset = Offset.OPEN
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: datetime = None
    reference: str = ""

@dataclass
class TradeData:
    symbol: str
    exchange: Exchange
    orderid: str
    tradeid: str
    direction: Direction = None
    offset: Offset = Offset.OPEN
    price: float = 0
    volume: float = 0
    datetime: datetime = None
