
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from khaos.核心引擎.indicator import KHAOSIndicator

def visualize_asset(file_path, output_dir, indicator):
    asset_name = os.path.basename(file_path).replace('.csv', '')
    print(f"Processing {asset_name}...")
    
    df = pd.read_csv(file_path)
    
    # Process
    try:
        df = indicator.process(df)
    except Exception as e:
        print(f"Error processing {asset_name}: {e}")
        return

    # Slice last 500 candles for better visualization
    df_vis = df.iloc[-500:].copy()
    
    # Create Subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f'{asset_name} Price & KHAOS Regime', 'KHAOS Instability (Trend Health)', 'KHAOS Anomaly (Energy/Vol)'))

    # 1. Price Candlestick
    fig.add_trace(go.Candlestick(x=df_vis.index,
                                 open=df_vis['open'], high=df_vis['high'],
                                 low=df_vis['low'], close=df_vis['close'],
                                 name='Price'), row=1, col=1)

    # Add Regime Background Colors
    # We can't easily add background shapes for every candle efficiently.
    # Alternative: Add a "Ribbon" trace at the bottom of the price chart or colored markers.
    # Let's add a "Regime Bar" at the bottom of the price subplot.
    
    # Define Regime Colors
    # Instability < 30: Green (Stable)
    # Instability > 80: Red (Chaos)
    # Else: Grey
    
    colors = []
    for val in df_vis['khaos_instability']:
        if val < 30:
            colors.append('green')
        elif val > 80:
            colors.append('red')
        else:
            colors.append('grey')
            
    # Add a Scatter trace acting as a Ribbon
    ribbon_y = df_vis['low'].min() * 0.999
    fig.add_trace(go.Scatter(x=df_vis.index, y=[ribbon_y]*len(df_vis),
                             mode='markers',
                             marker=dict(color=colors, symbol='square', size=5),
                             name='Regime Ribbon'), row=1, col=1)

    # Add Signals
    # Exhaustion (Triangle Down)
    exhaustion_mask = df_vis['signal_exhaustion'] == 1
    if exhaustion_mask.any():
        fig.add_trace(go.Scatter(x=df_vis[exhaustion_mask].index, 
                                 y=df_vis[exhaustion_mask]['high'] * 1.001,
                                 mode='markers',
                                 marker=dict(symbol='triangle-down', size=10, color='purple'),
                                 name='Trend Exhaustion'), row=1, col=1)
        
    # Vol Alert (Star)
    vol_mask = df_vis['signal_vol_alert'] == 1
    if vol_mask.any():
        fig.add_trace(go.Scatter(x=df_vis[vol_mask].index, 
                                 y=df_vis[vol_mask]['low'] * 0.998,
                                 mode='markers',
                                 marker=dict(symbol='star', size=10, color='orange'),
                                 name='Vol Explosion'), row=1, col=1)

    # 2. Instability
    fig.add_trace(go.Scatter(x=df_vis.index, y=df_vis['khaos_instability'],
                             mode='lines', line=dict(color='blue'),
                             name='Instability'), row=2, col=1)
    # Thresholds
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)

    # 3. Anomaly
    fig.add_trace(go.Bar(x=df_vis.index, y=df_vis['khaos_anomaly'],
                         marker_color='orange',
                         name='Anomaly (Energy)'), row=3, col=1)
    fig.add_hline(y=3.0, line_dash="dash", line_color="red", row=3, col=1)

    # Layout
    fig.update_layout(title_text=f"KHAOS Indicator Analysis: {asset_name}",
                      height=900, template="plotly_dark")
    
    # Save
    out_path = os.path.join(output_dir, f'khaos_{asset_name}.html')
    fig.write_html(out_path)
    print(f"Saved chart to {out_path}")

def main():
    data_dir = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed'
    output_dir = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\04_项目文档\实验报告\charts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Assets to visualize
    assets = [
        os.path.join(data_dir, 'Index', 'SPXUSD_1h.csv'),
        os.path.join(data_dir, 'Crypto', 'BTCUSD_1h.csv'),
        os.path.join(data_dir, 'Forex', 'EURUSD_1h.csv')
    ]
    
    indicator = KHAOSIndicator()
    
    for asset in assets:
        if os.path.exists(asset):
            visualize_asset(asset, output_dir, indicator)
        else:
            print(f"File not found: {asset}")

if __name__ == "__main__":
    main()
