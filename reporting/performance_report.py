"""
Performance reporting module for trading bot.
Generates detailed performance reports at configurable intervals.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
import jinja2
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceReporter:
    """Handles generation of performance reports at various intervals."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the performance reporter.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.report_dir = config.get('report_directory', 'reports/performance')
        self.template_dir = Path(__file__).parent / 'templates'
        
        # Create directories
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Create report template if it doesn't exist
        self._ensure_template_exists()
        
        # Track report generation times
        self.last_daily_report = None
        self.last_weekly_report = None
        self.last_monthly_report = None

    def _ensure_template_exists(self) -> None:
        """Create the HTML template if it doesn't exist."""
        template_path = self.template_dir / 'performance_report.html'
        if not template_path.exists():
            template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Report Period: {{ period_start }} to {{ period_end }}</p>
        </div>

        <div class="section">
            <h2>Performance Summary</h2>
            <div class="metric-grid">
                {% for metric in summary_metrics %}
                <div class="metric-card">
                    <div class="metric-title">{{ metric.name }}</div>
                    <div class="metric-value {% if metric.is_monetary %}{{ 'positive' if metric.value > 0 else 'negative' }}{% endif %}">
                        {{ metric.formatted_value }}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Trade Statistics</h2>
            <div class="metric-grid">
                {% for stat in trade_stats %}
                <div class="metric-card">
                    <div class="metric-title">{{ stat.name }}</div>
                    <div class="metric-value">{{ stat.value }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Risk Metrics</h2>
            <div class="metric-grid">
                {% for metric in risk_metrics %}
                <div class="metric-card">
                    <div class="metric-title">{{ metric.name }}</div>
                    <div class="metric-value">{{ metric.value }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        {% if charts %}
        <div class="section">
            <h2>Performance Charts</h2>
            {% for chart in charts %}
            <div class="chart-container">
                <h3>{{ chart.title }}</h3>
                <iframe src="{{ chart.path }}" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if recent_trades %}
        <div class="section">
            <h2>Recent Trades</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in recent_trades %}
                    <tr>
                        <td>{{ trade.timestamp }}</td>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ trade.side }}</td>
                        <td>{{ trade.entry_price }}</td>
                        <td>{{ trade.exit_price }}</td>
                        <td class="{{ 'positive' if trade.pnl > 0 else 'negative' }}">
                            {{ trade.pnl }}
                        </td>
                        <td>{{ trade.duration }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
            template_path.write_text(template_content)

    async def generate_reports(self, performance_data: Dict[str, Any]) -> None:
        """
        Generate reports based on configuration and timing.
        
        Args:
            performance_data: Current performance data
        """
        try:
            current_time = datetime.now()
            
            # Check and generate daily report
            if self.config.get('generate_daily_report', True):
                if (not self.last_daily_report or 
                    current_time.date() > self.last_daily_report.date()):
                    await self.generate_daily_report(performance_data)
                    self.last_daily_report = current_time
            
            # Check and generate weekly report
            if self.config.get('generate_weekly_report', True):
                if (not self.last_weekly_report or 
                    (current_time - self.last_weekly_report).days >= 7):
                    await self.generate_weekly_report(performance_data)
                    self.last_weekly_report = current_time
            
            # Check and generate monthly report
            if self.config.get('generate_monthly_report', True):
                if (not self.last_monthly_report or 
                    current_time.month != self.last_monthly_report.month):
                    await self.generate_monthly_report(performance_data)
                    self.last_monthly_report = current_time
                    
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")

    async def generate_daily_report(self, performance_data: Dict[str, Any]) -> None:
        """Generate daily performance report."""
        try:
            today = datetime.now().date()
            report_data = self._prepare_report_data(
                performance_data,
                title="Daily Performance Report",
                period_start=today,
                period_end=today
            )
            
            await self._generate_report(
                report_data,
                f"daily_report_{today.strftime('%Y%m%d')}.html"
            )
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")

    async def generate_weekly_report(self, performance_data: Dict[str, Any]) -> None:
        """Generate weekly performance report."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            report_data = self._prepare_report_data(
                performance_data,
                title="Weekly Performance Report",
                period_start=start_date,
                period_end=end_date
            )
            
            await self._generate_report(
                report_data,
                f"weekly_report_{end_date.strftime('%Y%m%d')}.html"
            )
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {str(e)}")

    async def generate_monthly_report(self, performance_data: Dict[str, Any]) -> None:
        """Generate monthly performance report."""
        try:
            end_date = datetime.now().date()
            start_date = end_date.replace(day=1)
            report_data = self._prepare_report_data(
                performance_data,
                title="Monthly Performance Report",
                period_start=start_date,
                period_end=end_date
            )
            
            await self._generate_report(
                report_data,
                f"monthly_report_{end_date.strftime('%Y%m')}.html"
            )
            
        except Exception as e:
            logger.error(f"Error generating monthly report: {str(e)}")

    def _prepare_report_data(self, performance_data: Dict[str, Any],
                           title: str, period_start: datetime.date,
                           period_end: datetime.date) -> Dict[str, Any]:
        """Prepare data for report template."""
        metrics = performance_data.get('metrics', {})
        trades = performance_data.get('trades', [])
        
        # Filter trades for the period
        period_trades = [
            t for t in trades
            if period_start <= datetime.fromisoformat(t['timestamp']).date() <= period_end
        ]
        
        # Prepare summary metrics
        summary_metrics = [
            {
                'name': 'Total P&L',
                'value': sum(t.get('pnl', 0) for t in period_trades),
                'formatted_value': f"${sum(t.get('pnl', 0) for t in period_trades):,.2f}",
                'is_monetary': True
            },
            {
                'name': 'Win Rate',
                'value': metrics.get('win_rate', 0),
                'formatted_value': f"{metrics.get('win_rate', 0)*100:.1f}%",
                'is_monetary': False
            },
            {
                'name': 'Profit Factor',
                'value': metrics.get('profit_factor', 0),
                'formatted_value': f"{metrics.get('profit_factor', 0):.2f}",
                'is_monetary': False
            }
        ]
        
        # Prepare trade statistics
        trade_stats = [
            {
                'name': 'Total Trades',
                'value': len(period_trades)
            },
            {
                'name': 'Average Trade',
                'value': f"${metrics.get('average_trade', 0):,.2f}"
            },
            {
                'name': 'Best Trade',
                'value': f"${metrics.get('best_trade', 0):,.2f}"
            },
            {
                'name': 'Worst Trade',
                'value': f"${metrics.get('worst_trade', 0):,.2f}"
            }
        ]
        
        # Prepare risk metrics
        risk_metrics = [
            {
                'name': 'Sharpe Ratio',
                'value': f"{metrics.get('sharpe_ratio', 0):.2f}"
            },
            {
                'name': 'Max Drawdown',
                'value': f"{metrics.get('max_drawdown', 0):.1f}%"
            },
            {
                'name': 'Average Hold Time',
                'value': f"{metrics.get('average_hold_time', 0):.1f}h"
            }
        ]
        
        # Get chart paths
        charts = [
            {'title': chart_type.replace('_', ' ').title(), 'path': path}
            for chart_type, path in performance_data.get('charts', {}).items()
        ]
        
        # Prepare recent trades
        recent_trades = []
        for trade in sorted(period_trades, key=lambda x: x['timestamp'], reverse=True)[:10]:
            recent_trades.append({
                'timestamp': datetime.fromisoformat(trade['timestamp']).strftime('%Y-%m-%d %H:%M'),
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': f"${float(trade['entry_price']):,.2f}",
                'exit_price': f"${float(trade.get('exit_price', 0)):,.2f}",
                'pnl': f"${float(trade.get('pnl', 0)):,.2f}",
                'duration': f"{float(trade.get('duration_hours', 0)):.1f}h"
            })
        
        return {
            'title': title,
            'period_start': period_start.strftime('%Y-%m-%d'),
            'period_end': period_end.strftime('%Y-%m-%d'),
            'summary_metrics': summary_metrics,
            'trade_stats': trade_stats,
            'risk_metrics': risk_metrics,
            'charts': charts,
            'recent_trades': recent_trades
        }

    async def _generate_report(self, report_data: Dict[str, Any], filename: str) -> None:
        """Generate HTML report from template."""
        try:
            template = self.jinja_env.get_template('performance_report.html')
            report_html = template.render(**report_data)
            
            report_path = os.path.join(self.report_dir, filename)
            with open(report_path, 'w') as f:
                f.write(report_html)
                
            logger.info(f"Generated report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")

    def get_latest_reports(self) -> Dict[str, str]:
        """
        Get paths to latest generated reports.
        
        Returns:
            dict: Report type to file path mapping
        """
        reports = {}
        
        # Find latest daily report
        daily_reports = sorted(
            [f for f in os.listdir(self.report_dir) if f.startswith('daily_report_')],
            reverse=True
        )
        if daily_reports:
            reports['daily'] = os.path.join(self.report_dir, daily_reports[0])
        
        # Find latest weekly report
        weekly_reports = sorted(
            [f for f in os.listdir(self.report_dir) if f.startswith('weekly_report_')],
            reverse=True
        )
        if weekly_reports:
            reports['weekly'] = os.path.join(self.report_dir, weekly_reports[0])
        
        # Find latest monthly report
        monthly_reports = sorted(
            [f for f in os.listdir(self.report_dir) if f.startswith('monthly_report_')],
            reverse=True
        )
        if monthly_reports:
            reports['monthly'] = os.path.join(self.report_dir, monthly_reports[0])
        
        return reports 