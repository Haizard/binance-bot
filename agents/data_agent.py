"""
Market data collection and processing agent implementation.
"""
import logging
import pandas as pd
from typing import Any, Dict, Optional, List
from binance.client import Client
from binance.websockets import BinanceSocketManager
from datetime import datetime, timedelta
from .base_agent import BaseAgent
import asyncio

logger = logging.getLogger(__name__)

class DataAgent(BaseAgent):
    """
    Agent responsible for collecting and managing market data.
    Handles WebSocket connections, historical data, and real-time updates.
    """
    def __init__(self, message_broker: Any = None):
        """
        Initialize the DataAgent.
        
        Args:
            message_broker: Message broker instance for inter-agent communication
        """
        super().__init__("Data", message_broker)
        self.client: Optional[Client] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        self.websocket_connections = {}
        self.price_cache = {}
        self.historical_data = {}
        self.active_symbols = set()
        self.interval = '1h'  # Default interval
        self.api_keys = {}

    async def setup(self) -> None:
        """Set up the data agent."""
        # Subscribe to configuration updates
        await self.subscribe("config.update")
        await self.subscribe("data.subscribe.request")
        await self.subscribe("data.unsubscribe.request")
        await self.subscribe("data.historical.request")
        
        # Request initial configuration
        await self.send_message("config.get.request", {
            'sender': self.name,
            'include_keys': True
        })
        
        logger.info("DataAgent setup completed")

    async def cleanup(self) -> None:
        """Clean up the data agent."""
        # Close all WebSocket connections
        if self.socket_manager:
            for symbol in list(self.websocket_connections.keys()):
                await self._stop_symbol_stream(symbol)
            self.socket_manager.close()
        
        # Unsubscribe from all topics
        for topic in list(self.subscriptions):
            await self.unsubscribe(topic)
        
        logger.info("DataAgent cleanup completed")

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
        elif topic == "data.subscribe.request":
            await self._handle_subscribe_request(data)
        elif topic == "data.unsubscribe.request":
            await self._handle_unsubscribe_request(data)
        elif topic == "data.historical.request":
            await self._handle_historical_request(data)
        elif topic.startswith("config.get.response"):
            await self._handle_config_response(data)

    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        config = data.get('config', {})
        api_keys = data.get('api_keys', {})
        
        if api_keys:
            self.api_keys = api_keys
            # Initialize/reinitialize Binance client if API keys changed
            self._initialize_client()
        
        # Update trading parameters
        trading_config = config.get('Trading', {})
        if trading_config:
            new_interval = trading_config.get('interval')
            if new_interval and new_interval != self.interval:
                self.interval = new_interval
                # Restart streams with new interval if needed
                await self._restart_streams()

    async def _handle_subscribe_request(self, data: Dict[str, Any]) -> None:
        """Handle symbol subscription requests."""
        symbol = data.get('symbol')
        if not symbol:
            logger.error("Invalid subscribe request - missing symbol")
            return
        
        try:
            await self._start_symbol_stream(symbol)
            self.active_symbols.add(symbol)
            logger.info(f"Subscribed to {symbol} data stream")
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {str(e)}")

    async def _handle_unsubscribe_request(self, data: Dict[str, Any]) -> None:
        """Handle symbol unsubscription requests."""
        symbol = data.get('symbol')
        if not symbol or symbol not in self.active_symbols:
            return
        
        try:
            await self._stop_symbol_stream(symbol)
            self.active_symbols.remove(symbol)
            logger.info(f"Unsubscribed from {symbol} data stream")
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {str(e)}")

    async def _handle_historical_request(self, data: Dict[str, Any]) -> None:
        """Handle historical data requests."""
        symbol = data.get('symbol')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        limit = data.get('limit', 500)
        sender = data.get('sender')
        
        if not all([symbol, sender]):
            logger.error("Invalid historical data request")
            return
        
        try:
            historical_data = await self._fetch_historical_data(
                symbol, start_time, end_time, limit
            )
            
            response = {
                'symbol': symbol,
                'data': historical_data.to_dict('records') if not historical_data.empty else []
            }
            
            await self.send_message(f"data.historical.response.{sender}", response)
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            await self.send_message(f"data.historical.response.{sender}", {
                'symbol': symbol,
                'error': str(e)
            })

    async def _handle_config_response(self, data: Dict[str, Any]) -> None:
        """Handle configuration response."""
        config = data.get('config', {})
        api_keys = data.get('api_keys', {})
        
        if api_keys:
            self.api_keys = api_keys
            self._initialize_client()
        
        # Initialize with configured trading pair
        trading_config = config.get('Trading', {})
        if trading_config:
            symbol = trading_config.get('symbol')
            self.interval = trading_config.get('interval', self.interval)
            if symbol:
                await self._start_symbol_stream(symbol)
                self.active_symbols.add(symbol)

    def _initialize_client(self) -> None:
        """Initialize or reinitialize the Binance client."""
        if self.api_keys.get('api_key') and self.api_keys.get('api_secret'):
            self.client = Client(
                self.api_keys['api_key'],
                self.api_keys['api_secret']
            )
            self.socket_manager = BinanceSocketManager(self.client)
            logger.info("Binance client initialized")
        else:
            logger.error("Cannot initialize Binance client - missing API keys")

    async def _start_symbol_stream(self, symbol: str) -> None:
        """
        Start WebSocket stream for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
        """
        if not self.socket_manager:
            logger.error("Cannot start stream - socket manager not initialized")
            return
        
        if symbol in self.websocket_connections:
            return
        
        try:
            # Start kline/candlestick WebSocket
            conn_key = self.socket_manager.start_kline_socket(
                symbol,
                self._handle_socket_message,
                interval=self.interval
            )
            self.websocket_connections[symbol] = conn_key
            logger.info(f"Started WebSocket stream for {symbol}")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket for {symbol}: {str(e)}")
            raise

    async def _stop_symbol_stream(self, symbol: str) -> None:
        """
        Stop WebSocket stream for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
        """
        if symbol in self.websocket_connections:
            try:
                self.socket_manager.stop_socket(self.websocket_connections[symbol])
                del self.websocket_connections[symbol]
                logger.info(f"Stopped WebSocket stream for {symbol}")
            except Exception as e:
                logger.error(f"Error stopping WebSocket for {symbol}: {str(e)}")

    async def _restart_streams(self) -> None:
        """Restart all active WebSocket streams."""
        symbols = list(self.active_symbols)
        for symbol in symbols:
            await self._stop_symbol_stream(symbol)
            await self._start_symbol_stream(symbol)

    async def _fetch_historical_data(
        self, 
        symbol: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical kline/candlestick data.
        
        Args:
            symbol (str): Trading pair symbol
            start_time (str, optional): Start time in ISO format
            end_time (str, optional): End time in ISO format
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: Historical data
        """
        if not self.client:
            raise ValueError("Binance client not initialized")
        
        try:
            # Convert time strings to timestamps if provided
            start_ts = int(datetime.fromisoformat(start_time).timestamp() * 1000) if start_time else None
            end_ts = int(datetime.fromisoformat(end_time).timestamp() * 1000) if end_time else None
            
            # Fetch klines/candlesticks
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=self.interval,
                start_str=start_ts,
                end_str=end_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def _handle_socket_message(self, msg: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket messages.
        
        Args:
            msg (dict): WebSocket message
        """
        try:
            if msg.get('e') == 'error':
                logger.error(f"WebSocket error: {msg.get('m')}")
                return
            
            # Extract kline data
            kline = msg.get('k', {})
            symbol = kline.get('s')
            if not symbol:
                return
            
            # Create candle data
            candle = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'closed': kline['x']
            }
            
            # Update price cache
            self.price_cache[symbol] = candle
            
            # Broadcast update if candle is closed
            if candle['closed']:
                asyncio.create_task(
                    self.send_message(f"data.update.{symbol}", {
                        'symbol': symbol,
                        'data': candle
                    })
                )
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            float: Latest price or None if not available
        """
        candle = self.price_cache.get(symbol)
        return float(candle['close']) if candle else None 