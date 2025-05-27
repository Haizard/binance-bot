"""
Performance tracking and analytics agent implementation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from visualization.performance_charts import PerformanceVisualizer
from reporting.performance_report import PerformanceReporter
import os
import shutil
import asyncio
from status_manager import StatusManager

logger = logging.getLogger(__name__)

class PerformanceAgent(BaseAgent):
    """
    Agent responsible for tracking and analyzing trading performance.
    Handles performance metrics, risk monitoring, and analytics.
    """
    
    def __init__(self, message_broker: Any = None):
        """Initialize the PerformanceAgent."""
        super().__init__("Performance", message_broker)
        self.trades = []  # List of all trades
        self.positions = {}  # Current positions
        self.daily_stats = {}  # Daily performance statistics
        self.metrics = {}  # Current performance metrics
        self.config = {}  # Performance tracking configuration
        self.visualizer = None  # Performance visualizer instance

    async def setup(self) -> None:
        """Set up the performance agent."""
        # Subscribe to necessary topics
        await self.subscribe("config.update")
        await self.subscribe("trade.executed")
        await self.subscribe("trade.closed")
        await self.subscribe("position.update")
        
        # Request initial configuration
        await self.send_message("config.get.request", {
            'sender': self.name,
            'include_keys': False
        })
        
        StatusManager().update("Performance", {"message": "Setup completed, waiting for trades"})
        logger.info("PerformanceAgent setup completed")
        # Start heartbeat
        asyncio.create_task(self._heartbeat())

    async def _heartbeat(self):
        while True:
            # Show summary of trades, PnL, and open positions
            total_trades = self.metrics.get('total_trades', 0)
            total_profit = self.metrics.get('total_profit', 0)
            open_positions = []
            for symbol, pos in self.positions.items():
                open_positions.append(f"{symbol}({pos['side']})")
            msg = f"PerformanceAgent alive | Trades: {total_trades}, PnL: {total_profit:.2f}"
            if open_positions:
                msg += " | Open: " + ", ".join(open_positions)
            StatusManager().update("Performance", {"message": msg})
            await asyncio.sleep(5)

    async def cleanup(self) -> None:
        """Clean up the performance agent."""
        # Save current performance data
        await self._save_performance_data()
        
        # Generate final performance report
        await self._generate_performance_report()
        
        # Unsubscribe from all topics
        for topic in list(self.subscriptions):
            await self.unsubscribe(topic)
            
        logger.info("PerformanceAgent cleanup completed")

    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming messages.
        
        Args:
            message (dict): Message to process
        """
        topic = message['topic']
        data = message['message'].get('data', {})
        
        if topic == "config.update":
            await self._handle_config_update(data)
        elif topic == "trade.executed":
            await self._handle_trade_execution(data)
        elif topic == "trade.closed":
            await self._handle_trade_closure(data)
        elif topic == "position.update":
            await self._handle_position_update(data)

    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        config = data.get('config', {})
        
        # Update performance tracking configuration
        performance_config = config.get('Performance', {})
        if performance_config:
            self.config = performance_config
            
            # Update tracking parameters
            await self._update_tracking_parameters()
            
            # Initialize or update visualizer
            if performance_config.get('enable_live_charts', True):
                if not self.visualizer:
                    self.visualizer = PerformanceVisualizer(performance_config)
                    asyncio.create_task(self.visualizer.start())
                else:
                    self.visualizer.config = performance_config

    async def _handle_trade_execution(self, data: Dict[str, Any]) -> None:
        """Handle trade execution updates."""
        trade = data.get('trade', {})
        if not trade:
            return
            
        # Add trade to history
        self.trades.append(trade)
        
        # Update positions
        symbol = trade['symbol']
        self.positions[symbol] = trade
        
        # Update performance metrics
        await self._update_metrics()
        
        # Check risk limits
        if await self._check_risk_limits():
            # Broadcast risk alert if limits exceeded
            await self._send_risk_alert()

    async def _handle_trade_closure(self, data: Dict[str, Any]) -> None:
        """Handle trade closure updates."""
        trade = data.get('trade', {})
        if not trade:
            return
            
        # Update trade history
        self._update_trade_history(trade)
        
        # Remove from active positions
        symbol = trade['symbol']
        if symbol in self.positions:
            del self.positions[symbol]
            
        # Update performance metrics
        await self._update_metrics()
        
        # Generate trade report
        await self._generate_trade_report(trade)

    async def _handle_position_update(self, data: Dict[str, Any]) -> None:
        """Handle position updates."""
        position = data.get('position', {})
        if not position:
            return
            
        # Update position tracking
        symbol = position['symbol']
        self.positions[symbol] = position
        
        # Update unrealized P&L
        await self._update_unrealized_pnl()
        
        # Check position risk
        if await self._check_position_risk(position):
            # Send position risk alert
            await self._send_position_alert(position)

    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        try:
            StatusManager().update("Performance", {"message": "Updating performance metrics..."})
            # Calculate basic metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trades if t.get('pnl', 0) < 0])
            
            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit metrics
            total_profit = sum(t.get('pnl', 0) for t in self.trades)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Calculate risk metrics
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Update metrics dictionary
            self.metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': avg_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast metrics update
            await self.send_message("performance.metrics.update", {
                'metrics': self.metrics
            })
            
            StatusManager().update("Performance", {"message": f"Metrics updated: {self.metrics.get('total_trades', 0)} trades, PnL: {self.metrics.get('total_profit', 0):.2f}"})
            StatusManager().update("Performance", {"message": "Performance metrics updated"})
            
        except Exception as e:
            StatusManager().update("Performance", {"message": f"Error updating metrics: {str(e)}"})
            logger.error(f"Error updating metrics: {str(e)}")

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if not self.trades:
                return 0.0
                
            # Create equity curve
            equity_curve = pd.DataFrame(self.trades)
            equity_curve['cumulative_pnl'] = equity_curve['pnl'].cumsum()
            
            # Calculate running maximum
            running_max = equity_curve['cumulative_pnl'].expanding().max()
            
            # Calculate drawdown
            drawdown = (equity_curve['cumulative_pnl'] - running_max) / running_max
            
            return abs(float(drawdown.min()))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not self.trades:
                return 0.0
                
            # Convert trades to daily returns
            df = pd.DataFrame(self.trades)
            df['date'] = pd.to_datetime(df['timestamp'])
            daily_returns = df.groupby(df['date'].dt.date)['pnl'].sum()
            
            # Calculate Sharpe ratio
            risk_free_rate = self.config.get('risk_free_rate', 0.02)
            trading_days = 252
            
            excess_returns = daily_returns - (risk_free_rate / trading_days)
            sharpe_ratio = np.sqrt(trading_days) * (excess_returns.mean() / excess_returns.std())
            
            return float(sharpe_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    async def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for open positions."""
        try:
            total_unrealized = 0.0
            
            for symbol, position in self.positions.items():
                entry_price = position['entry_price']
                current_price = position['current_price']
                size = position['size']
                side = position['side']
                
                # Calculate unrealized P&L
                if side == 'LONG':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # SHORT
                    unrealized_pnl = (entry_price - current_price) * size
                    
                position['unrealized_pnl'] = unrealized_pnl
                total_unrealized += unrealized_pnl
            
            # Broadcast unrealized P&L update
            await self.send_message("performance.unrealized_pnl.update", {
                'total_unrealized_pnl': total_unrealized,
                'positions': self.positions
            })
            
        except Exception as e:
            logger.error(f"Error updating unrealized P&L: {str(e)}")

    async def _check_risk_limits(self) -> bool:
        """
        Check if any risk limits are exceeded.
        
        Returns:
            bool: True if any limits are exceeded
        """
        try:
            # Get risk limits
            daily_loss_limit = self.config.get('daily_loss_limit', 0.05)
            max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)
            
            # Calculate current metrics
            daily_pnl = self._calculate_daily_pnl()
            current_drawdown = self._calculate_max_drawdown()
            
            # Check limits
            limits_exceeded = False
            
            if abs(daily_pnl) > daily_loss_limit:
                limits_exceeded = True
                logger.warning(f"Daily loss limit exceeded: {daily_pnl:.2%}")
                
            if current_drawdown > max_drawdown_limit:
                limits_exceeded = True
                logger.warning(f"Maximum drawdown limit exceeded: {current_drawdown:.2%}")
                
            return limits_exceeded
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False

    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        try:
            today = datetime.now().date()
            
            # Get today's trades
            today_trades = [
                t for t in self.trades
                if datetime.fromisoformat(t['timestamp']).date() == today
            ]
            
            # Calculate realized P&L
            realized_pnl = sum(t.get('pnl', 0) for t in today_trades)
            
            # Add unrealized P&L
            unrealized_pnl = sum(
                p.get('unrealized_pnl', 0)
                for p in self.positions.values()
            )
            
            return realized_pnl + unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {str(e)}")
            return 0.0

    async def _generate_trade_report(self, trade: Dict[str, Any]) -> None:
        """Generate report for a closed trade."""
        try:
            # Calculate trade metrics
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            size = trade['size']
            side = trade['side']
            pnl = trade['pnl']
            duration = (
                datetime.fromisoformat(trade['exit_time']) -
                datetime.fromisoformat(trade['entry_time'])
            )
            
            # Create trade report
            report = {
                'trade_id': trade.get('id'),
                'symbol': trade['symbol'],
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_percent': (pnl / (entry_price * size)) * 100,
                'duration': str(duration),
                'strategy': trade.get('strategy'),
                'entry_signal': trade.get('entry_signal'),
                'exit_signal': trade.get('exit_signal')
            }
            
            # Broadcast trade report
            await self.send_message("performance.trade.report", {
                'report': report
            })
            
        except Exception as e:
            logger.error(f"Error generating trade report: {str(e)}")

    async def _generate_performance_report(self) -> None:
        """Generate comprehensive performance report."""
        try:
            StatusManager().update("Performance", {"message": "Generating performance report..."})
            # Get current metrics
            current_metrics = self.metrics.copy()
            
            # Calculate additional metrics
            profit_factor = self._calculate_profit_factor()
            avg_trade_duration = self._calculate_avg_trade_duration()
            largest_winner = self._get_largest_trade(win=True)
            largest_loser = self._get_largest_trade(win=False)
            
            # Create report
            report = {
                'metrics': current_metrics,
                'additional_metrics': {
                    'profit_factor': profit_factor,
                    'avg_trade_duration': avg_trade_duration,
                    'largest_winner': largest_winner,
                    'largest_loser': largest_loser
                },
                'positions': self.positions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast report
            await self.send_message("performance.report", {
                'report': report
            })
            
            StatusManager().update("Performance", {"message": "Performance report generated."})
            
        except Exception as e:
            StatusManager().update("Performance", {"message": f"Error generating performance report: {str(e)}"})
            logger.error(f"Error generating performance report: {str(e)}")

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        try:
            gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
            
            return gross_profit / gross_loss if gross_loss != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {str(e)}")
            return 0.0

    def _calculate_avg_trade_duration(self) -> timedelta:
        """Calculate average trade duration."""
        try:
            durations = [
                datetime.fromisoformat(t['exit_time']) -
                datetime.fromisoformat(t['entry_time'])
                for t in self.trades
                if 'exit_time' in t and 'entry_time' in t
            ]
            
            return sum(durations, timedelta()) / len(durations) if durations else timedelta()
            
        except Exception as e:
            logger.error(f"Error calculating average trade duration: {str(e)}")
            return timedelta()

    def _get_largest_trade(self, win: bool = True) -> Dict[str, Any]:
        """Get details of largest winning or losing trade."""
        try:
            trades = [t for t in self.trades if (t.get('pnl', 0) > 0) == win]
            if not trades:
                return {}
                
            largest_trade = max(trades, key=lambda x: abs(x.get('pnl', 0)))
            return {
                'symbol': largest_trade['symbol'],
                'pnl': largest_trade['pnl'],
                'timestamp': largest_trade['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting largest trade: {str(e)}")
            return {}

    async def _update_tracking_parameters(self) -> None:
        """Update performance tracking parameters from configuration."""
        try:
            # Get tracking parameters
            self.tracking_params = {
                'save_interval': self.config.get('save_interval', 3600),  # Save every hour
                'max_trade_history': self.config.get('max_trade_history', 1000),
                'metrics_window': self.config.get('metrics_window', 30),  # 30 days
                'risk_free_rate': self.config.get('risk_free_rate', 0.02)  # 2% annual
            }
            
            logger.info("Performance tracking parameters updated")
            
        except Exception as e:
            logger.error(f"Error updating tracking parameters: {str(e)}")

    async def _save_performance_data(self) -> None:
        """Save performance data to persistent storage."""
        try:
            # Create data directory if it doesn't exist
            data_dir = self.config.get('data_directory', 'data/performance')
            os.makedirs(data_dir, exist_ok=True)
            
            # Convert trade history to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trades_df.to_csv(f'{data_dir}/performance_data_{timestamp}.csv', index=False)
            
            # Save performance metrics
            metrics_df = pd.DataFrame([self.metrics])
            metrics_df.to_csv(f'{data_dir}/performance_metrics_{timestamp}.csv', index=False)
            
            # Save daily statistics
            daily_stats_df = pd.DataFrame.from_dict(self.daily_stats, orient='index')
            daily_stats_df.to_csv(f'{data_dir}/daily_stats_{timestamp}.csv')
            
            # Get and save latest charts if visualization is enabled
            if self.visualizer:
                chart_paths = self.visualizer.get_latest_charts()
                for chart_type, path in chart_paths.items():
                    if os.path.exists(path):
                        new_path = f'{data_dir}/{chart_type}_{timestamp}.html'
                        shutil.copy2(path, new_path)
            
            logger.info("Performance data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current performance metrics.
        
        Returns:
            dict: Performance summary
        """
        summary = {
            'metrics': self.metrics,
            'daily_stats': self.daily_stats.get(datetime.now().date().isoformat(), {}),
            'positions': self.positions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add chart paths if visualization is enabled
        if self.visualizer:
            summary['charts'] = self.visualizer.get_latest_charts()
        
        return summary 