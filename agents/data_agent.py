"""
Market data collection and processing agent implementation.
"""
import logging
import pandas as pd
from typing import Any, Dict, Optional, List
from binance.client import Client
from binance import BinanceSocketManager
from datetime import datetime, timedelta
from .base_agent import BaseAgent
import asyncio
from status_manager import StatusManager
import yaml
import os
from binance import AsyncClient

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
        self.websocket_tasks = {}
        self.price_cache = {}
        self.historical_data = {}
        self.active_symbols = set()
        self.interval = '1m'  # Changed default interval to 1m
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
        
        StatusManager().update("Data", {"message": "Setup completed, waiting for data requests"})
        logger.info("DataAgent setup completed")
        # Start heartbeat
        asyncio.create_task(self._heartbeat())
        # Try to load config.yaml directly if config not set
        if not hasattr(self, 'config') or not self.config:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
            config_path = os.path.abspath(config_path)
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.config = config
                api_cfg = config.get('api', {})
                trading_cfg = config.get('trading', {})
                if api_cfg:
                    self.api_keys = {
                        'api_key': api_cfg.get('api_key'),
                        'api_secret': api_cfg.get('api_secret')
                    }
                    await self._initialize_client()
                StatusManager().update("Data", {"message": f"Loaded config.yaml directly: api_keys={bool(self.api_keys)}, symbols={trading_cfg.get('symbols', [])}"})
                logger.info(f"Loaded config.yaml directly: api_keys={self.api_keys}, trading={trading_cfg}")
                # Auto-subscribe to symbols
                if trading_cfg:
                    for symbol in trading_cfg.get('symbols', []):
                        await self._handle_subscribe_request({'symbol': symbol})
            except Exception as e:
                StatusManager().update("Data", {"message": f"Error loading config.yaml: {str(e)}"})
                logger.error(f"Error loading config.yaml: {str(e)}")

    async def _heartbeat(self):
        while True:
            symbol_status = []
            debug_info = []
            # Log client state and active symbols
            if not self.client:
                debug_info.append("Binance client NOT initialized")
            else:
                debug_info.append("Binance client OK")
            debug_info.append(f"Active symbols: {list(self.active_symbols)}")
            
            if not self.active_symbols:
                msg = "DataAgent alive | No active symbols | " + ", ".join(debug_info)
                StatusManager().update("Data", {"message": msg})
                await asyncio.sleep(5)
                continue
            for symbol in self.active_symbols:
                price = self.price_cache.get(symbol)
                bids = asks = []
                volume = '?'
                spread = '?'
                error = None
                # Try to get order book and ticker info if client is available
                if self.client:
                    try:
                        ob = await self.client.get_order_book(symbol=symbol)
                        bids = ob['bids'][:5] if ob['bids'] else []
                        asks = ob['asks'][:5] if ob['asks'] else []
                        ticker = await self.client.get_ticker(symbol=symbol)
                        volume = ticker.get('volume', '?')
                        if bids and asks:
                            spread = f"{float(asks[0][0]) - float(bids[0][0]):.2g}"
                    except Exception as e:
                        error = str(e)
                bid_str = ','.join([f"{b[0]}({b[1]})" for b in bids]) if bids else '?'
                ask_str = ','.join([f"{a[0]}({a[1]})" for a in asks]) if asks else '?'
                status = (f"{symbol}: price={price if price is not None else '?'} "
                          f"spread={spread} vol={volume} "
                          f"bids=[{bid_str}] asks=[{ask_str}]")
                if error:
                    status += f" [ERROR: {error}]"
                symbol_status.append(status)
            msg = "DataAgent alive"
            if symbol_status:
                msg += " | " + " || ".join(symbol_status)
            if debug_info:
                 msg += " | " + ", ".join(debug_info)
            StatusManager().update("Data", {"message": msg})
            await asyncio.sleep(5)

    async def cleanup(self) -> None:
        """Clean up the data agent."""
        # Close all WebSocket connections
        if self.socket_manager:
            self.socket_manager.stop()
            for symbol in list(self.websocket_tasks.keys()):
                await self._stop_symbol_stream(symbol)
        
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

    async def _auto_subscribe_symbols(self, config):
        trading_cfg = config.get('trading') or config.get('Trading')
        if trading_cfg:
            symbols = trading_cfg.get('symbols', [])
            for symbol in symbols:
                await self._handle_subscribe_request({'symbol': symbol})
                StatusManager().update("Data", {"message": f"Auto-subscribed to {symbol} after config loaded"})

    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        config = data.get('config', {})
        api_keys = data.get('api_keys', {})

        StatusManager().update("Data", {"message": f"Received config.update: api_keys={bool(api_keys)}, config keys={list(config.keys())}"})
        logger.info(f"Received config.update: api_keys={api_keys}, config={config}")
        
        if api_keys:
            self.api_keys = api_keys
            # Initialize/reinitialize Binance client if API keys changed
            await self._initialize_client()
        
        # Update trading parameters
        trading_config = config.get('Trading', {}) or config.get('trading', {})
        if trading_config:
            new_interval = trading_config.get('interval')
            # Only update interval if explicitly provided in config and different
            if new_interval and new_interval != self.interval:
                self.interval = new_interval
                logger.info(f"DataAgent interval updated to: {self.interval}. Restarting streams...")
                await self._restart_streams()
            elif not new_interval:
                 # If no interval in config, use the default '1m'
                 self.interval = '1m'
                 logger.info(f"No interval specified in config, using default: {self.interval}. Restarting streams...")
                 await self._restart_streams()
        # Auto-subscribe after config update
        await self._auto_subscribe_symbols(config)

    async def _handle_subscribe_request(self, data: Dict[str, Any]) -> None:
        """Handle symbol subscription requests."""
        symbol = data.get('symbol')
        if not symbol:
            logger.error("Invalid subscribe request - missing symbol")
            return
        
        try:
            await self._start_symbol_stream(symbol)
            self.active_symbols.add(symbol)
            StatusManager().update("Data", {"message": f"Subscribed to {symbol} data stream"})
            logger.info(f"Subscribed to {symbol} data stream")
        except Exception as e:
            StatusManager().update("Data", {"message": f"Error subscribing to {symbol}: {str(e)}"})
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
            StatusManager().update("Data", {"message": f"Fetched historical data for {symbol}"})
            
            response = {
                'symbol': symbol,
                'data': historical_data.to_dict('records') if not historical_data.empty else []
            }
            
            await self.send_message(f"data.historical.response.{sender}", response)
            
        except Exception as e:
            StatusManager().update("Data", {"message": f"Error fetching historical data for {symbol}: {str(e)}"})
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            await self.send_message(f"data.historical.response.{sender}", {
                'symbol': symbol,
                'error': str(e)
            })

    async def _handle_config_response(self, data: Dict[str, Any]) -> None:
        """Handle configuration response."""
        config = data.get('config', {})
        api_keys = data.get('api_keys', {})

        StatusManager().update("Data", {"message": f"Received config.get.response: api_keys={bool(api_keys)}, config keys={list(config.keys())}"})
        logger.info(f"Received config.get.response: api_keys={api_keys}, config={config}")
        
        if api_keys:
            self.api_keys = api_keys
            await self._initialize_client()
        
        # Initialize with configured trading pair
        trading_config = config.get('Trading', {}) or config.get('trading', {})
        if trading_config:
            symbol = trading_config.get('symbol') # This seems wrong, should subscribe to all symbols in list
            symbols = trading_config.get('symbols', []) # Correctly get list of symbols
            new_interval = trading_config.get('interval')
            
            # Only update interval if explicitly provided in config
            if new_interval:
                 self.interval = new_interval
                 logger.info(f"DataAgent interval set from config response: {self.interval}")

            # Subscribe to all symbols from config
            for symbol in symbols:
                await self._start_symbol_stream(symbol)
                self.active_symbols.add(symbol)
                logger.info(f"DataAgent auto-subscribed to {symbol} from config response")

        # Auto-subscribe after config response (redundant with above, but keeping for now)
        # await self._auto_subscribe_symbols(config)

    async def _initialize_client(self) -> None:
        """Initialize or reinitialize the Binance client and socket manager (async)."""
        if self.api_keys.get('api_key') and self.api_keys.get('api_secret'):
            # Ensure AsyncClient is used for async operations
            if not isinstance(self.client, Client) or not hasattr(self.client, 'close_connection'): # Basic check for AsyncClient properties
                logger.info("Initializing AsyncClient")
                # Pass requests_params={} to explicitly not use a proxy
                self.client = await AsyncClient.create(
                    api_key=self.api_keys['api_key'],
                    api_secret=self.api_keys['api_secret'],
                    requests_params={} # Explicitly pass empty requests_params
                )
                if self.socket_manager:
                    # No explicit close for async manager, just drop reference
                    self.socket_manager = None
                self.socket_manager = BinanceSocketManager(self.client)
                logger.info("Binance client and async socket manager initialized")
            else:
                logger.info("Binance client already initialized")
        else:
            logger.error("Cannot initialize Binance client - missing API keys")

    async def _start_symbol_stream(self, symbol: str) -> None:
        """Start a WebSocket kline stream for a given symbol using the existing socket manager."""
        if symbol in self.websocket_tasks:
            logger.warning(f"Stream for {symbol} already exists.")
            return

        if not self.client or not self.socket_manager:
            logger.error(f"Binance client or socket manager not initialized, cannot start stream for {symbol}.")
            StatusManager().update("Data", {"message": f"Cannot start stream for {symbol}: Client/Socket Manager not initialized", "health": "ERROR"})
            return

        try:
            # Use the existing socket manager to start the kline stream
            # stream_name = f'{symbol.lower()}@kline_{self.interval}' # Construction moved into kline_socket call
            logger.info(f"Attempting to start kline stream for {symbol} with interval {self.interval}")
            
            async def kline_listener():
                # Use the stream context manager from the existing socket manager
                try:
                    async with self.socket_manager.kline_socket(symbol=symbol.lower(), interval=self.interval) as stream:
                        logger.info(f"WebSocket kline stream started for {symbol} with interval {self.interval}")
                        while True:
                            msg = await stream.recv()
                            logger.debug(f"Received message for {symbol}: {msg}") # Debugging: Log all incoming messages
                            if isinstance(msg, dict):
                                self._handle_socket_message(msg)
                            else:
                                logger.warning(f"Received non-dict message for {symbol}: {msg}")
                except Exception as e:
                    logger.error(f"Error within kline_listener for {symbol}: {str(e)}")
                    # Attempt to restart the stream after a delay
                    await asyncio.sleep(10) # Wait before attempting restart
                    logger.info(f"Attempting to restart kline stream for {symbol} after error.")
                    # Need a way to signal the outer function to restart, or handle restart internally
                    # For now, just log and the outer loop might try again depending on agent lifecycle
                    # A more robust approach would involve managing tasks and restarting them explicitly
                    pass # Allow the task to end after logging the error

            task = asyncio.create_task(kline_listener())
            self.websocket_tasks[symbol] = task
            logger.info(f"Started {symbol} stream (async) using existing socket manager")

        except Exception as e:
            logger.error(f"Error starting WebSocket stream for {symbol}: {str(e)}")
            StatusManager().update("Data", {"message": f"Error starting stream for {symbol}: {str(e)}", "health": "ERROR"})

    async def _stop_symbol_stream(self, symbol: str) -> None:
        """
        Stop async WebSocket stream for a symbol.
        """
        if symbol not in self.websocket_tasks:
            return
        try:
            task = self.websocket_tasks[symbol]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.websocket_tasks[symbol]
            StatusManager().update("Data", {"message": f"Stopped {symbol} stream (async)"})
            logger.info(f"Stopped {symbol} stream (async)")
        except Exception as e:
            StatusManager().update("Data", {"message": f"Error stopping {symbol} stream: {str(e)}"})
            logger.error(f"Error stopping {symbol} stream: {str(e)}")
            raise

    async def _restart_streams(self) -> None:
        """Restart all active streams with new settings."""
        active_symbols = list(self.active_symbols)
        for symbol in active_symbols:
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
        Currently only processes kline updates.
        """
        #logger.debug(f"Processing socket message: {msg}") # This can be noisy, use sparingly
        
        event_type = msg.get('e')
        
        if event_type == 'kline':
            kline = msg.get('k')
            if kline:
                symbol = kline.get('s')
                # We want to update the price cache on ANY kline update, not just closed ones
                # is_kline_closed = kline.get('x')
                
                if symbol:
                    # Process kline update (either open or closed)
                    # logger.info(f"Received kline update for {symbol}") # Can be noisy
                    try:
                        close_price = float(kline['c'])
                        # Update price cache with the latest closing price from the kline update
                        self.price_cache[symbol] = close_price
                        logger.debug(f"Updated price_cache for {symbol}: {self.price_cache[symbol]}") # Debugging: Log price update
                        
                        # Optionally, publish the raw kline data for other agents
                        # This is commented out to reduce message traffic unless needed
                        # processed_kline = {
                        #     'timestamp': int(kline['t']),
                        #     'open': float(kline['o']),
                        #     'high': float(kline['h']),
                        #     'low': float(kline['l']),
                        #     'close': close_price,
                        #     'volume': float(kline['v']),
                        #     'symbol': symbol
                        # }
                        # asyncio.create_task(
                        #     self.send_message(
                        #         f"data.update.{symbol}",
                        #         {'type': 'kline', 'data': processed_kline}
                        #     )
                        # )

                    except Exception as e:
                        logger.error(f"Error processing kline data for {symbol}: {str(e)}")
        # Add handling for other potential message types if necessary in the future
        # elif event_type == 'some_other_event':
        #     pass

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            float: Latest price or None if not available
        """
        return self.price_cache.get(symbol) 