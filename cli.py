#!/usr/bin/env python3
"""
Command-line interface for managing the trading bot and alert system.
"""
import click
import yaml
import asyncio
from typing import Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import time
from alerts import (
    Alert,
    AlertType,
    AlertSeverity,
    PerformanceAlertMonitor,
    NotificationHandler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading config: {str(e)}[/red]")
        sys.exit(1)

class AlertMonitor:
    """Interactive alert monitoring interface."""
    
    def __init__(self, alert_monitor: PerformanceAlertMonitor,
                 notification_handler: NotificationHandler):
        """Initialize the alert monitor."""
        self.alert_monitor = alert_monitor
        self.notification_handler = notification_handler
        self.layout = Layout()
        
        # Configure layout
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="alerts", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
    def generate_alert_table(self) -> Table:
        """Generate table of active alerts."""
        table = Table(title="Active Alerts")
        
        table.add_column("Time", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Severity", style="magenta")
        table.add_column("Message", style="white")
        table.add_column("Value", justify="right", style="green")
        table.add_column("Threshold", justify="right", style="yellow")
        
        for alert in self.alert_monitor.get_active_alerts():
            severity_style = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.CRITICAL: "red"
            }.get(alert.severity, "white")
            
            table.add_row(
                alert.timestamp.strftime("%H:%M:%S"),
                alert.type.value.title(),
                Text(alert.severity.value.title(), style=severity_style),
                alert.message,
                f"{alert.metric_value:.2f}",
                f"{alert.threshold:.2f}"
            )
            
        return table
        
    def generate_stats_panel(self) -> Panel:
        """Generate statistics panel."""
        active_alerts = self.alert_monitor.get_active_alerts()
        stats = {
            "Total Active": len(active_alerts),
            "Critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "Warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            "Info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
        }
        
        content = "\n".join([
            f"{key}: {value}"
            for key, value in stats.items()
        ])
        
        return Panel(
            content,
            title="Alert Statistics",
            border_style="blue"
        )
        
    def update_display(self) -> None:
        """Update the display with current data."""
        # Update header
        self.layout["header"].update(
            Panel(
                f"Trading Bot Alert Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style="bold white"
            )
        )
        
        # Update main content
        self.layout["alerts"].update(self.generate_alert_table())
        self.layout["stats"].update(self.generate_stats_panel())
        
        # Update footer
        notification_status = {
            "Email": self.notification_handler.email_config.get('enabled', False),
            "Discord": self.notification_handler.discord_config.get('enabled', False),
            "Telegram": self.notification_handler.telegram_config.get('enabled', False)
        }
        
        status_text = " | ".join([
            f"{channel}: {'[green]✓[/green]' if enabled else '[red]✗[/red]'}"
            for channel, enabled in notification_status.items()
        ])
        
        self.layout["footer"].update(
            Panel(status_text, title="Notification Channels")
        )

@click.group()
def cli():
    """Trading bot command-line interface."""
    pass

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config file')
def start(config):
    """Start the trading bot."""
    try:
        cfg = load_config(config)
        console.print("[green]Starting trading bot...[/green]")
        # Initialize bot components here
        console.print("[green]Trading bot started successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error starting bot: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config file')
def monitor(config):
    """Monitor alerts in real-time."""
    try:
        cfg = load_config(config)
        
        # Initialize components
        alert_monitor = PerformanceAlertMonitor(cfg)
        notification_handler = NotificationHandler(cfg)
        
        # Create monitor interface
        monitor = AlertMonitor(alert_monitor, notification_handler)
        
        console.print("[green]Starting alert monitor...[/green]")
        
        with Live(monitor.layout, refresh_per_second=1) as live:
            while True:
                monitor.update_display()
                time.sleep(1)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping alert monitor...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in alert monitor: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config file')
@click.option('--days', '-d', default=7, help='Number of days of history to show')
def history(config, days):
    """Show alert history."""
    try:
        cfg = load_config(config)
        alert_monitor = PerformanceAlertMonitor(cfg)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        alerts = alert_monitor.get_alert_history(
            start_time=start_time,
            end_time=end_time
        )
        
        table = Table(title=f"Alert History (Last {days} days)")
        table.add_column("Time", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Severity", style="magenta")
        table.add_column("Message", style="white")
        table.add_column("Value", justify="right", style="green")
        table.add_column("Threshold", justify="right", style="yellow")
        
        for alert in alerts:
            severity_style = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.CRITICAL: "red"
            }.get(alert.severity, "white")
            
            table.add_row(
                alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                alert.type.value.title(),
                Text(alert.severity.value.title(), style=severity_style),
                alert.message,
                f"{alert.metric_value:.2f}",
                f"{alert.threshold:.2f}"
            )
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error showing history: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config file')
def status(config):
    """Show bot and alert system status."""
    try:
        cfg = load_config(config)
        alert_monitor = PerformanceAlertMonitor(cfg)
        notification_handler = NotificationHandler(cfg)
        
        # Alert statistics
        active_alerts = alert_monitor.get_active_alerts()
        stats_table = Table(title="Alert Statistics")
        stats_table.add_column("Metric", style="blue")
        stats_table.add_column("Value", style="cyan")
        
        stats = {
            "Total Active Alerts": len(active_alerts),
            "Critical Alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "Warning Alerts": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            "Info Alerts": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
        }
        
        for key, value in stats.items():
            stats_table.add_row(key, str(value))
            
        console.print(stats_table)
        
        # Notification channels
        channels_table = Table(title="Notification Channels")
        channels_table.add_column("Channel", style="blue")
        channels_table.add_column("Status", style="cyan")
        
        channels = {
            "Email": notification_handler.email_config.get('enabled', False),
            "Discord": notification_handler.discord_config.get('enabled', False),
            "Telegram": notification_handler.telegram_config.get('enabled', False)
        }
        
        for channel, enabled in channels.items():
            channels_table.add_row(
                channel,
                "[green]Active[/green]" if enabled else "[red]Inactive[/red]"
            )
            
        console.print(channels_table)
        
    except Exception as e:
        console.print(f"[red]Error showing status: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config file')
def clear(config):
    """Clear all active alerts."""
    try:
        cfg = load_config(config)
        alert_monitor = PerformanceAlertMonitor(cfg)
        
        count = len(alert_monitor.get_active_alerts())
        alert_monitor.clear_all_alerts()
        
        console.print(f"[green]Cleared {count} active alerts[/green]")
        
    except Exception as e:
        console.print(f"[red]Error clearing alerts: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config file')
def test_notifications(config):
    """Test notification channels."""
    try:
        cfg = load_config(config)
        notification_handler = NotificationHandler(cfg)
        
        # Create test alert
        test_alert = Alert(
            type=AlertType.INFO,
            severity=AlertSeverity.INFO,
            message="This is a test notification",
            timestamp=datetime.now(),
            metric_value=0.0,
            threshold=0.0
        )
        
        with console.status("[bold blue]Testing notification channels...") as status:
            asyncio.run(notification_handler.send_alert(test_alert))
            
        console.print("[green]Test notifications sent successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error testing notifications: {str(e)}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    cli() 