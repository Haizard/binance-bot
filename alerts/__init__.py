"""
Trading bot alerts system.
Provides real-time performance monitoring and multi-channel notifications.
"""

from .performance_alerts import (
    Alert,
    AlertType,
    AlertSeverity,
    PerformanceAlertMonitor
)

from .notification_handler import NotificationHandler

__all__ = [
    'Alert',
    'AlertType',
    'AlertSeverity',
    'PerformanceAlertMonitor',
    'NotificationHandler'
]

__version__ = '1.0.0' 