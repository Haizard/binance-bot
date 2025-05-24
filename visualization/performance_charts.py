"""
Interactive performance visualization system using Plotly.
"""
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import os

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """
    Handles creation and updating of interactive performance charts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.charts = {}  # Store chart figures
        self.data = {
            'trades': [],
            'equity_curve': pd.DataFrame(),
            'daily_stats': {},
            'metrics': {}
        }
        self.chart_dir = "charts"
        os.makedirs(self.chart_dir, exist_ok=True)
        
    async def start(self) -> None:
        """Start the visualization update loop."""
        update_interval = self.config.get('visualization', {}).get('update_interval', 300)
        
        while True:
            try:
                await self.update_charts()
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in visualization update loop: {str(e)}")
                await asyncio.sleep(10)  # Short delay on error

    async def update_data(self, new_data: Dict[str, Any]) -> None:
        """
        Update the underlying data for visualization.
        
        Args:
            new_data: New trading and performance data
        """
        try:
            # Update stored data
            self.data.update(new_data)
            
            # Convert trades to DataFrame if needed
            if isinstance(self.data['trades'], list):
                self.data['trades'] = pd.DataFrame(self.data['trades'])
            
            # Update equity curve
            if not self.data['equity_curve'].empty:
                self._update_equity_curve()
                
        except Exception as e:
            logger.error(f"Error updating visualization data: {str(e)}")

    async def update_charts(self) -> None:
        """Update all enabled chart types."""
        if not self.config.get('visualization', {}).get('enabled', True):
            return
            
        try:
            chart_types = self.config.get('visualization', {}).get('chart_types', [])
            
            for chart_type in chart_types:
                if chart_type == 'equity_curve':
                    await self._create_equity_curve()
                elif chart_type == 'drawdown':
                    await self._create_drawdown_chart()
                elif chart_type == 'win_rate':
                    await self._create_win_rate_chart()
                elif chart_type == 'profit_distribution':
                    await self._create_profit_distribution()
                elif chart_type == 'trade_duration':
                    await self._create_trade_duration_chart()
                elif chart_type == 'position_size':
                    await self._create_position_size_chart()
                    
            # Save updated charts
            await self._save_charts()
            
        except Exception as e:
            logger.error(f"Error updating charts: {str(e)}")

    async def _create_equity_curve(self) -> None:
        """Create equity curve chart."""
        try:
            if self.data['trades'].empty:
                return
                
            # Calculate cumulative equity
            df = self.data['trades'].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Create figure
            fig = go.Figure()
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_pnl'],
                name='Equity Curve',
                line=dict(color='blue', width=2)
            ))
            
            # Add benchmark if enabled
            if self.config.get('visualization', {}).get('include_benchmarks', True):
                # Add market benchmark (e.g., S&P 500) comparison
                pass
            
            # Customize layout
            fig.update_layout(
                title='Trading Account Equity Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L',
                hovermode='x unified',
                showlegend=True
            )
            
            self.charts['equity_curve'] = fig
            
        except Exception as e:
            logger.error(f"Error creating equity curve: {str(e)}")

    async def _create_drawdown_chart(self) -> None:
        """Create drawdown analysis chart."""
        try:
            if self.data['trades'].empty:
                return
                
            # Calculate drawdown
            df = self.data['trades'].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Calculate running maximum and drawdown
            df['running_max'] = df['cumulative_pnl'].expanding().max()
            df['drawdown'] = (df['cumulative_pnl'] - df['running_max']) / df['running_max'] * 100
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Equity Curve with Running Maximum', 'Drawdown (%)')
            )
            
            # Add equity curve and running maximum
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_pnl'],
                    name='Equity',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['running_max'],
                    name='Running Maximum',
                    line=dict(color='green', dash='dash')
                ),
                row=1, col=1
            )
            
            # Add drawdown
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['drawdown'],
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Drawdown Analysis',
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            self.charts['drawdown'] = fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {str(e)}")

    async def _create_win_rate_chart(self) -> None:
        """Create win rate analysis chart."""
        try:
            if self.data['trades'].empty:
                return
                
            df = self.data['trades'].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate rolling win rate
            window = self.config.get('metrics', {}).get('rolling_window_days', 30)
            df['is_win'] = df['pnl'] > 0
            df['rolling_wins'] = df['is_win'].rolling(window=window).sum()
            df['rolling_total'] = df['is_win'].rolling(window=window).count()
            df['win_rate'] = (df['rolling_wins'] / df['rolling_total']) * 100
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Win/Loss Distribution', 'Rolling Win Rate (%)')
            )
            
            # Add win/loss distribution
            win_loss = df['is_win'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=['Wins', 'Losses'],
                    y=[win_loss.get(True, 0), win_loss.get(False, 0)],
                    name='Trade Outcomes',
                    marker_color=['green', 'red']
                ),
                row=1, col=1
            )
            
            # Add rolling win rate
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['win_rate'],
                    name=f'{window}-Trade Rolling Win Rate',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Win Rate Analysis',
                height=800,
                showlegend=True
            )
            
            self.charts['win_rate'] = fig
            
        except Exception as e:
            logger.error(f"Error creating win rate chart: {str(e)}")

    async def _create_profit_distribution(self) -> None:
        """Create profit distribution chart."""
        try:
            if self.data['trades'].empty:
                return
                
            df = self.data['trades'].copy()
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('P&L Distribution', 'P&L by Trade')
            )
            
            # Add P&L distribution histogram
            fig.add_trace(
                go.Histogram(
                    x=df['pnl'],
                    name='P&L Distribution',
                    nbinsx=50,
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            # Add P&L by trade scatter
            fig.add_trace(
                go.Scatter(
                    x=range(len(df)),
                    y=df['pnl'],
                    mode='markers',
                    name='P&L by Trade',
                    marker=dict(
                        color=df['pnl'],
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Profit Distribution Analysis',
                height=800,
                showlegend=True
            )
            
            self.charts['profit_distribution'] = fig
            
        except Exception as e:
            logger.error(f"Error creating profit distribution chart: {str(e)}")

    async def _create_trade_duration_chart(self) -> None:
        """Create trade duration analysis chart."""
        try:
            if self.data['trades'].empty:
                return
                
            df = self.data['trades'].copy()
            
            # Calculate trade durations
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600  # Hours
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Trade Duration Distribution', 'Duration vs P&L')
            )
            
            # Add duration distribution
            fig.add_trace(
                go.Histogram(
                    x=df['duration'],
                    name='Duration Distribution',
                    nbinsx=30,
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            # Add duration vs P&L scatter
            fig.add_trace(
                go.Scatter(
                    x=df['duration'],
                    y=df['pnl'],
                    mode='markers',
                    name='Duration vs P&L',
                    marker=dict(
                        color=df['pnl'],
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Trade Duration Analysis',
                height=800,
                showlegend=True,
                xaxis2_title='Duration (hours)',
                yaxis2_title='P&L'
            )
            
            self.charts['trade_duration'] = fig
            
        except Exception as e:
            logger.error(f"Error creating trade duration chart: {str(e)}")

    async def _create_position_size_chart(self) -> None:
        """Create position size analysis chart."""
        try:
            if self.data['trades'].empty:
                return
                
            df = self.data['trades'].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Position Size Distribution', 'Position Size vs P&L')
            )
            
            # Add position size distribution
            fig.add_trace(
                go.Histogram(
                    x=df['size'],
                    name='Size Distribution',
                    nbinsx=30,
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            # Add size vs P&L scatter
            fig.add_trace(
                go.Scatter(
                    x=df['size'],
                    y=df['pnl'],
                    mode='markers',
                    name='Size vs P&L',
                    marker=dict(
                        color=df['pnl'],
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Position Size Analysis',
                height=800,
                showlegend=True,
                xaxis2_title='Position Size',
                yaxis2_title='P&L'
            )
            
            self.charts['position_size'] = fig
            
        except Exception as e:
            logger.error(f"Error creating position size chart: {str(e)}")

    def _update_equity_curve(self) -> None:
        """Update the equity curve data."""
        try:
            df = self.data['trades'].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate cumulative equity
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            self.data['equity_curve'] = df[['timestamp', 'cumulative_pnl']]
            
        except Exception as e:
            logger.error(f"Error updating equity curve data: {str(e)}")

    async def _save_charts(self) -> None:
        """Save all charts to HTML files."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for chart_type, fig in self.charts.items():
                filepath = os.path.join(self.chart_dir, f'{chart_type}_{timestamp}.html')
                fig.write_html(filepath)
                
        except Exception as e:
            logger.error(f"Error saving charts: {str(e)}")

    def get_latest_charts(self) -> Dict[str, str]:
        """
        Get paths to the latest chart files.
        
        Returns:
            dict: Mapping of chart types to their file paths
        """
        latest_charts = {}
        
        try:
            for chart_type in self.charts.keys():
                # Find latest file for this chart type
                files = [f for f in os.listdir(self.chart_dir) if f.startswith(chart_type)]
                if files:
                    latest_file = max(files)
                    latest_charts[chart_type] = os.path.join(self.chart_dir, latest_file)
                    
        except Exception as e:
            logger.error(f"Error getting latest charts: {str(e)}")
            
        return latest_charts 