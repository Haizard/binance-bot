"""
Configuration management agent implementation.
"""
import os
import json
import logging
import configparser
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ConfigAgent(BaseAgent):
    """
    Agent responsible for managing configuration settings and API keys.
    Handles loading, validation, and distribution of configuration to other agents.
    """
    def __init__(self, message_broker: Any = None, config_file: str = "config.ini"):
        """
        Initialize the ConfigAgent.
        
        Args:
            message_broker: Message broker instance for inter-agent communication
            config_file (str): Path to the configuration file
        """
        super().__init__("Config", message_broker)
        self.config_file = config_file
        self.config = {}
        self.api_keys = {}
        
    async def setup(self) -> None:
        """Set up the configuration agent."""
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        await self._load_config()
        
        # Subscribe to configuration update requests
        await self.subscribe("config.update.request")
        await self.subscribe("config.get.request")
        
        logger.info("ConfigAgent setup completed")

    async def cleanup(self) -> None:
        """Clean up the configuration agent."""
        # Unsubscribe from all topics
        for topic in list(self.subscriptions):
            await self.unsubscribe(topic)
        logger.info("ConfigAgent cleanup completed")

    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming messages.
        
        Args:
            message (dict): Message to process
        """
        topic = message['topic']
        data = message['message'].get('data', {})
        
        if topic == "config.update.request":
            await self._handle_config_update(data)
        elif topic == "config.get.request":
            await self._handle_config_get(data)

    async def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Load config file
            config = configparser.ConfigParser()
            config.read(self.config_file)
            
            # Convert config to dictionary
            self.config = {
                section: dict(config.items(section)) 
                for section in config.sections()
            }
            
            # Load API keys from environment or config
            self.api_keys = {
                'api_key': os.getenv('BINANCE_API_KEY', self.config.get('Binance', {}).get('api_key', '')),
                'api_secret': os.getenv('BINANCE_API_SECRET', self.config.get('Binance', {}).get('api_secret', ''))
            }
            
            # Validate configuration
            self._validate_config()
            
            # Notify other agents of configuration
            await self._broadcast_config()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = ['Binance', 'Trading', 'Strategies', 'Logging']
        required_trading_params = ['symbol', 'risk_per_trade', 'stop_loss', 'take_profit']
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check required trading parameters
        trading_config = self.config.get('Trading', {})
        for param in required_trading_params:
            if param not in trading_config:
                raise ValueError(f"Missing required trading parameter: {param}")
        
        # Validate API keys
        if not self.api_keys['api_key'] or not self.api_keys['api_secret']:
            raise ValueError("Missing Binance API credentials")

    async def _broadcast_config(self) -> None:
        """Broadcast configuration to all agents."""
        config_message = {
            'config': self.config,
            'api_keys': self.api_keys
        }
        await self.send_message("config.update", config_message)

    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """
        Handle configuration update requests.
        
        Args:
            data (dict): Update request data
        """
        try:
            section = data.get('section')
            updates = data.get('updates', {})
            
            if not section or not updates:
                logger.error("Invalid config update request")
                return
            
            # Update configuration
            if section not in self.config:
                self.config[section] = {}
            self.config[section].update(updates)
            
            # Save to file if needed
            # TODO: Implement config file update
            
            # Broadcast update
            await self._broadcast_config()
            
            logger.info(f"Configuration section {section} updated")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")

    async def _handle_config_get(self, data: Dict[str, Any]) -> None:
        """
        Handle configuration get requests.
        
        Args:
            data (dict): Get request data
        """
        try:
            section = data.get('section')
            sender = data.get('sender')
            
            if not sender:
                logger.error("Invalid config get request - missing sender")
                return
            
            response = {
                'config': self.config.get(section) if section else self.config,
                'api_keys': self.api_keys if data.get('include_keys') else None
            }
            
            await self.send_message(f"config.get.response.{sender}", response)
            
        except Exception as e:
            logger.error(f"Error handling config get request: {str(e)}")

    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(section, {}).get(key, default)

    def get_api_keys(self) -> Dict[str, str]:
        """
        Get API keys.
        
        Returns:
            dict: API key credentials
        """
        return self.api_keys.copy() 