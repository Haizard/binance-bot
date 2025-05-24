"""
Notification handler for delivering trading alerts through various channels.
"""
import logging
import smtplib
import aiohttp
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime
from .performance_alerts import Alert, AlertSeverity

logger = logging.getLogger(__name__)

class NotificationHandler:
    """Handles delivery of alerts through configured notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification handler.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.email_config = config.get('email', {})
        self.discord_config = config.get('discord', {})
        self.telegram_config = config.get('telegram', {})
        
        # Initialize session for HTTP requests
        self.session = None
        
        # Track notification history
        self.notification_history: List[Dict[str, Any]] = []
        
    async def start(self):
        """Start the notification handler."""
        self.session = aiohttp.ClientSession()
        
    async def stop(self):
        """Stop the notification handler and cleanup."""
        if self.session:
            await self.session.close()
            self.session = None

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through all configured channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            bool: Whether alert was sent successfully through any channel
        """
        success = False
        notification_record = {
            'alert': alert,
            'timestamp': datetime.now(),
            'channels': {}
        }
        
        # Send through email if configured
        if self.email_config.get('enabled', False):
            email_success = await self._send_email_alert(alert)
            notification_record['channels']['email'] = email_success
            success = success or email_success
            
        # Send through Discord if configured
        if self.discord_config.get('enabled', False):
            discord_success = await self._send_discord_alert(alert)
            notification_record['channels']['discord'] = discord_success
            success = success or discord_success
            
        # Send through Telegram if configured
        if self.telegram_config.get('enabled', False):
            telegram_success = await self._send_telegram_alert(alert)
            notification_record['channels']['telegram'] = telegram_success
            success = success or telegram_success
            
        # Record notification attempt
        self.notification_history.append(notification_record)
        
        # Trim history if too long
        max_history = self.config.get('max_notification_history', 1000)
        if len(self.notification_history) > max_history:
            self.notification_history = self.notification_history[-max_history:]
            
        return success

    async def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert through email."""
        try:
            sender = self.email_config['sender']
            recipients = self.email_config['recipients']
            smtp_server = self.email_config['smtp_server']
            smtp_port = self.email_config['smtp_port']
            username = self.email_config['username']
            password = self.email_config['password']
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Trading Alert: {alert.type.value.title()} - {alert.severity.value.title()}"
            
            # Format message body
            body = f"""
Trading Alert

Type: {alert.type.value.title()}
Severity: {alert.severity.value.title()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Message: {alert.message}

Current Value: {alert.metric_value}
Threshold: {alert.threshold}
"""
            if alert.symbol:
                body += f"Symbol: {alert.symbol}\n"
                
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
                
            logger.info(f"Sent email alert: {alert.type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False

    async def _send_discord_alert(self, alert: Alert) -> bool:
        """Send alert through Discord webhook."""
        try:
            if not self.session:
                logger.error("HTTP session not initialized")
                return False
                
            webhook_url = self.discord_config['webhook_url']
            
            # Format message
            color = {
                AlertSeverity.INFO: 0x3498db,  # Blue
                AlertSeverity.WARNING: 0xf1c40f,  # Yellow
                AlertSeverity.CRITICAL: 0xe74c3c,  # Red
            }.get(alert.severity, 0x95a5a6)  # Gray default
            
            embed = {
                "title": f"Trading Alert: {alert.type.value.title()}",
                "description": alert.message,
                "color": color,
                "fields": [
                    {
                        "name": "Severity",
                        "value": alert.severity.value.title(),
                        "inline": True
                    },
                    {
                        "name": "Current Value",
                        "value": f"{alert.metric_value}",
                        "inline": True
                    },
                    {
                        "name": "Threshold",
                        "value": f"{alert.threshold}",
                        "inline": True
                    }
                ],
                "timestamp": alert.timestamp.isoformat()
            }
            
            if alert.symbol:
                embed["fields"].append({
                    "name": "Symbol",
                    "value": alert.symbol,
                    "inline": True
                })
            
            payload = {
                "embeds": [embed]
            }
            
            async with self.session.post(webhook_url, json=payload) as response:
                if response.status == 204:  # Discord returns 204 on success
                    logger.info(f"Sent Discord alert: {alert.type.value}")
                    return True
                else:
                    logger.error(f"Discord API error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Discord alert: {str(e)}")
            return False

    async def _send_telegram_alert(self, alert: Alert) -> bool:
        """Send alert through Telegram."""
        try:
            if not self.session:
                logger.error("HTTP session not initialized")
                return False
                
            bot_token = self.telegram_config['bot_token']
            chat_ids = self.telegram_config['chat_ids']
            
            # Format message
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.CRITICAL: "🚨"
            }.get(alert.severity, "")
            
            message = f"{severity_emoji} *Trading Alert*\n\n"
            message += f"*Type:* {alert.type.value.title()}\n"
            message += f"*Severity:* {alert.severity.value.title()}\n"
            message += f"*Message:* {alert.message}\n\n"
            message += f"*Current Value:* {alert.metric_value}\n"
            message += f"*Threshold:* {alert.threshold}\n"
            
            if alert.symbol:
                message += f"*Symbol:* {alert.symbol}\n"
                
            # Send to all configured chat IDs
            success = True
            for chat_id in chat_ids:
                params = {
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                async with self.session.post(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Telegram API error: {response.status}")
                        success = False
                        
            if success:
                logger.info(f"Sent Telegram alert: {alert.type.value}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {str(e)}")
            return False

    def get_notification_history(self,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               alert_type: Optional[str] = None,
                               severity: Optional[AlertSeverity] = None,
                               channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get notification history with optional filters.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            alert_type: Filter by alert type
            severity: Filter by alert severity
            channel: Filter by notification channel
            
        Returns:
            list: Filtered notification history
        """
        history = self.notification_history.copy()
        
        if start_time:
            history = [n for n in history if n['timestamp'] >= start_time]
        if end_time:
            history = [n for n in history if n['timestamp'] <= end_time]
        if alert_type:
            history = [n for n in history if n['alert'].type.value == alert_type]
        if severity:
            history = [n for n in history if n['alert'].severity == severity]
        if channel:
            history = [n for n in history if channel in n['channels']]
            
        return history 