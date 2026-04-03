# run_custom.py
# 这是一个自定义的 VeighNa 启动脚本，专门用于解决 CtaBacktester 模块未加载的问题

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

# 导入 CTA 回测模块
from vnpy_ctabacktester import CtaBacktesterApp
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_datamanager import DataManagerApp

def main():
    qapp = create_qapp()

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # --- 关键步骤：手动添加应用模块 ---
    main_engine.add_app(CtaBacktesterApp) # 加载 CTA 回测
    main_engine.add_app(CtaStrategyApp)   # 加载 CTA 策略 (实盘用)
    main_engine.add_app(DataManagerApp)   # 加载 数据管理

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()

if __name__ == "__main__":
    main()
