import pandas as pd
import numpy as np
from datetime import datetime
import os

class ReportGenerator:
    @staticmethod
    def generate_html_report(trades_df, equity_curve, metrics, filename="backtest_report.html"):
        """
        Generate a standalone HTML report.
        """
        
        # 1. Prepare Metrics HTML
        metrics_html = ""
        for k, v in metrics.items():
            metrics_html += f"""
            <div class="metric-card">
                <div class="metric-title">{k}</div>
                <div class="metric-value">{v}</div>
            </div>
            """
            
        # 2. Prepare Trade Table HTML (Top 100)
        trades_html = trades_df.head(100).to_html(classes="table table-striped", index=False)
        
        # 3. Simple SVG Chart (Equity Curve)
        # Normalize equity to 0-100 for SVG
        if not equity_curve.empty:
            eq_values = equity_curve['equity'].values
            dates = equity_curve['datetime'].dt.strftime('%Y-%m-%d').values
            
            min_val = np.min(eq_values)
            max_val = np.max(eq_values)
            range_val = max_val - min_val if max_val != min_val else 1
            
            svg_points = ""
            width = 800
            height = 300
            n_points = len(eq_values)
            step = width / (n_points - 1) if n_points > 1 else 0
            
            for i, val in enumerate(eq_values):
                x = i * step
                y = height - ((val - min_val) / range_val * height)
                svg_points += f"{x},{y} "
                
            svg_chart = f"""
            <svg width="100%" height="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="none">
                <polyline points="{svg_points}" fill="none" stroke="#007bff" stroke-width="2" />
            </svg>
            """
        else:
            svg_chart = "No Data"

        # 4. Full HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>KHAOS Quant Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f6f9; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }}
                .metric-title {{ font-size: 0.9em; color: #6c757d; }}
                .metric-value {{ font-size: 1.4em; font-weight: bold; color: #2c3e50; }}
                .chart-container {{ height: 300px; background: #fff; border: 1px solid #eee; margin-bottom: 30px; position: relative; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
                th {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #f1f1f1; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>KHAOS Quant Strategy Report</h1>
                <p>Generated on: {datetime.now()}</p>
                
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    {metrics_html}
                </div>
                
                <h2>Equity Curve</h2>
                <div class="chart-container">
                    {svg_chart}
                </div>
                
                <h2>Trade History (First 100)</h2>
                <div style="overflow-x: auto;">
                    {trades_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report saved to {filename}")
