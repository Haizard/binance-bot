"""
Performance alerts module for trading bot.
Monitors performance metrics and triggers alerts based on configured thresholds.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of performance alerts."""
    DRAWDOWN = "drawdown"
    LOSS_STREAK = "loss_streak"
    DAILY_LOSS = "daily_loss"
    PROFIT_TARGET = "profit_target"
    VOLATILITY = "volatility"
    WIN_RATE = "win_rate"
    RISK_EXPOSURE = "risk_exposure"
    TRADE_FREQUENCY = "trade_frequency"

@dataclass
class Alert:
    """Alert data structure."""
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float
    symbol: Optional[str] = None

class PerformanceAlertMonitor:
    """Monitors trading performance and generates alerts based on thresholds."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert monitor.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_alert_times: Dict[AlertType, datetime] = {}
        self.alert_cooldown = timedelta(minutes=config.get('alert_cooldown_minutes', 15))

    async def process_performance_update(self, performance_data: Dict[str, Any]) -> List[Alert]:
        """
        Process performance update and generate alerts if thresholds are breached.
        
        Args:
            performance_data: Current performance metrics and trade data
            
        Returns:
            list: New alerts generated
        """
        try:
            new_alerts = []
            metrics = performance_data.get('metrics', {})
            daily_stats = performance_data.get('daily_stats', {})
            
            # Check drawdown
            if await self._should_alert(AlertType.DRAWDOWN):
                drawdown = abs(metrics.get('max_drawdown', 0))
                threshold = self.alert_thresholds.get('drawdown_threshold', 0.1)
                if drawdown >= threshold:
                    alert = Alert(
                        type=AlertType.DRAWDOWN,
                        severity=AlertSeverity.WARNING if drawdown < threshold * 1.5 else AlertSeverity.CRITICAL,
                        message=f"Drawdown alert: Current drawdown of {drawdown:.1%} exceeds threshold of {threshold:.1%}",
                        timestamp=datetime.now(),
                        metric_value=drawdown,
                        threshold=threshold
                    )
                    new_alerts.append(alert)
                    await self._record_alert(alert)

            # Check loss streak
            current_streak = self._calculate_loss_streak(performance_data.get('trades', []))
            if current_streak > self.consecutive_losses:
                self.consecutive_losses = current_streak
                streak_threshold = self.alert_thresholds.get('loss_streak', 5)
                if current_streak >= streak_threshold and await self._should_alert(AlertType.LOSS_STREAK):
                    alert = Alert(
                        type=AlertType.LOSS_STREAK,
                        severity=AlertSeverity.WARNING if current_streak < streak_threshold * 1.5 else AlertSeverity.CRITICAL,
                        message=f"Loss streak alert: {current_streak} consecutive losing trades",
                        timestamp=datetime.now(),
                        metric_value=current_streak,
                        threshold=streak_threshold
                    )
                    new_alerts.append(alert)
                    await self._record_alert(alert)

            # Check daily loss limit
            daily_pnl = daily_stats.get('total_pnl', 0)
            if daily_pnl < 0:
                loss_limit = self.alert_thresholds.get('daily_loss_limit', 1000)
                if abs(daily_pnl) >= loss_limit and await self._should_alert(AlertType.DAILY_LOSS):
                    alert = Alert(
                        type=AlertType.DAILY_LOSS,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Daily loss limit alert: Current loss ${abs(daily_pnl):.2f} exceeds limit ${loss_limit:.2f}",
                        timestamp=datetime.now(),
                        metric_value=abs(daily_pnl),
                        threshold=loss_limit
                    )
                    new_alerts.append(alert)
                    await self._record_alert(alert)

            # Check profit target
            elif daily_pnl > 0:
                profit_target = self.alert_thresholds.get('profit_target', 5000)
                if daily_pnl >= profit_target and await self._should_alert(AlertType.PROFIT_TARGET):
                    alert = Alert(
                        type=AlertType.PROFIT_TARGET,
                        severity=AlertSeverity.INFO,
                        message=f"Profit target reached: ${daily_pnl:.2f}",
                        timestamp=datetime.now(),
                        metric_value=daily_pnl,
                        threshold=profit_target
                    )
                    new_alerts.append(alert)
                    await self._record_alert(alert)

            # Check win rate
            win_rate = metrics.get('win_rate', 0)
            min_win_rate = self.alert_thresholds.get('min_win_rate', 0.4)
            if win_rate < min_win_rate and await self._should_alert(AlertType.WIN_RATE):
                alert = Alert(
                    type=AlertType.WIN_RATE,
                    severity=AlertSeverity.WARNING,
                    message=f"Low win rate alert: Current win rate {win_rate:.1%} below minimum {min_win_rate:.1%}",
                    timestamp=datetime.now(),
                    metric_value=win_rate,
                    threshold=min_win_rate
                )
                new_alerts.append(alert)
                await self._record_alert(alert)

            # Check risk exposure
            risk_exposure = self._calculate_risk_exposure(performance_data)
            max_risk = self.alert_thresholds.get('max_risk_exposure', 0.15)
            if risk_exposure > max_risk and await self._should_alert(AlertType.RISK_EXPOSURE):
                alert = Alert(
                    type=AlertType.RISK_EXPOSURE,
                    severity=AlertSeverity.WARNING,
                    message=f"High risk exposure: Current exposure {risk_exposure:.1%} exceeds maximum {max_risk:.1%}",
                    timestamp=datetime.now(),
                    metric_value=risk_exposure,
                    threshold=max_risk
                )
                new_alerts.append(alert)
                await self._record_alert(alert)

            return new_alerts

        except Exception as e:
            logger.error(f"Error processing performance update for alerts: {str(e)}")
            return []

    def _calculate_loss_streak(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate current consecutive loss streak."""
        streak = 0
        for trade in reversed(trades):
            if trade.get('pnl', 0) < 0:
                streak += 1
            else:
                break
        return streak

    def _calculate_risk_exposure(self, performance_data: Dict[str, Any]) -> float:
        """Calculate current risk exposure based on open positions."""
        try:
            active_trades = performance_data.get('active_trades', {})
            total_exposure = sum(
                float(trade.get('quantity', 0)) * float(trade.get('entry_price', 0))
                for trade in active_trades.values()
            )
            account_value = performance_data.get('metrics', {}).get('account_value', 0)
            return total_exposure / account_value if account_value > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating risk exposure: {str(e)}")
            return 0.0

    async def _should_alert(self, alert_type: AlertType) -> bool:
        """Check if enough time has passed since the last alert of this type."""
        last_alert_time = self.last_alert_times.get(alert_type)
        if not last_alert_time:
            return True
            
        time_since_last = datetime.now() - last_alert_time
        return time_since_last >= self.alert_cooldown

    async def _record_alert(self, alert: Alert) -> None:
        """Record a new alert."""
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[alert.type] = alert.timestamp
        
        # Remove old alerts from active list
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > cutoff_time
        ]
        
        # Trim alert history if too long
        max_history = self.config.get('max_alert_history', 1000)
        if len(self.alert_history) > max_history:
            self.alert_history = self.alert_history[-max_history:]

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get current active alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            list: Active alerts
        """
        if severity:
            return [alert for alert in self.active_alerts if alert.severity == severity]
        return self.active_alerts.copy()

    def get_alert_history(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         alert_type: Optional[AlertType] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get historical alerts with optional filters.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            alert_type: Filter by alert type
            severity: Filter by severity
            
        Returns:
            list: Filtered alert history
        """
        alerts = self.alert_history.copy()
        
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return alerts

    def clear_alert(self, alert: Alert) -> None:
        """
        Clear an active alert.
        
        Args:
            alert: Alert to clear
        """
        if alert in self.active_alerts:
            self.active_alerts.remove(alert)

    def clear_all_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear() 