"""
Trade execution and position management agent implementation.
"""
import logging
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime
from decimal import Decimal
from .base_agent import BaseAgent
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class TradeAgent(BaseAgent):
    """
    Agent responsible for trade execution and position management.
    Handles order placement, position sizing, and risk management.
    """
    
    def __init__(self, message_broker: Any = None):
        """Initialize the TradeAgent."""
        super().__init__("Trade", message_broker)
        self.client = None  # Binance client
        self.positions = {}  # Active positions
        self.orders = {}    # Active orders
        self.config = {}    # Trading configuration
        self.risk_limits = {}  # Risk management limits

    async def setup(self) -> None:
        """Set up the trade agent."""
        # Subscribe to necessary topics
        await self.subscribe("config.update")
        await self.subscribe("analysis.update.*")  # Subscribe to all analysis updates
        await self.subscribe("risk.limits.update")
        
        # Request initial configuration
        await self.send_message("config.get.request", {
            'sender': self.name,
            'include_keys': True  # Need API keys
        })
        
        logger.info("TradeAgent setup completed")

    async def cleanup(self) -> None:
        """Clean up the trade agent."""
        # Cancel all active orders
        await self._cancel_all_orders()
        
        # Close all positions if configured to do so
        if self.config.get('close_positions_on_shutdown', True):
            await self._close_all_positions()
        
        # Clear local state
        self.positions.clear()
        self.orders.clear()
        
        # Unsubscribe from all topics
        for topic in list(self.subscriptions):
            await self.unsubscribe(topic)
            
        logger.info("TradeAgent cleanup completed")

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
        elif topic.startswith("analysis.update."):
            await self._handle_analysis_update(topic.split('.')[-1], data)
        elif topic == "risk.limits.update":
            await self._handle_risk_update(data)

    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        config = data.get('config', {})
        
        # Update trading configuration
        trading_config = config.get('Trading', {})
        if trading_config:
            self.config = trading_config
            
            # Initialize or update Binance client if API keys provided
            api_key = trading_config.get('api_key')
            api_secret = trading_config.get('api_secret')
            
            if api_key and api_secret:
                self.client = Client(api_key, api_secret)
                logger.info("Binance client initialized")

    async def _handle_analysis_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Handle analysis updates and execute trades if signals are present.
        
        Args:
            symbol (str): Trading pair symbol
            data (dict): Analysis data
        """
        if not self.client or not self._is_trading_enabled():
            return
            
        analysis = data.get('analysis', {})
        if not analysis:
            return
            
        # Get the combined signal
        signal = analysis.get('combined_signal', {})
        if not signal:
            return
            
        # Check if signal meets execution criteria
        if await self._should_execute_trade(symbol, signal):
            # Determine position size
            position_size = await self._calculate_position_size(symbol)
            
            if signal['signal'] > 0:  # Buy signal
                await self._execute_long_entry(symbol, position_size, signal)
            elif signal['signal'] < 0:  # Sell signal
                await self._execute_short_entry(symbol, position_size, signal)

    async def _handle_risk_update(self, data: Dict[str, Any]) -> None:
        """Handle risk limit updates."""
        self.risk_limits = data.get('limits', {})
        
        # Check if any positions violate new limits
        for symbol, position in self.positions.items():
            if not await self._is_within_risk_limits(symbol, position):
                await self._reduce_position(symbol, position)

    async def _execute_long_entry(self, symbol: str, size: Decimal, signal: Dict[str, Any]) -> None:
        """Execute a long position entry."""
        try:
            # Calculate entry price and order parameters
            ticker = await self._get_ticker(symbol)
            entry_price = Decimal(ticker['ask'])
            
            # Calculate stop loss and take profit levels
            stop_loss = entry_price * (1 - self.config.get('stop_loss_percent', 0.02))
            take_profit = entry_price * (1 + self.config.get('take_profit_percent', 0.04))
            
            # Place the entry order
            order = await self._place_order(
                symbol=symbol,
                side="BUY",
                quantity=size,
                price=entry_price
            )
            
            if order:
                # Place stop loss and take profit orders
                await self._place_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=size,
                    stop_price=stop_loss,
                    order_type="STOP_LOSS_LIMIT"
                )
                
                await self._place_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=size,
                    price=take_profit,
                    order_type="LIMIT"
                )
                
                # Update position tracking
                self.positions[symbol] = {
                    'side': 'LONG',
                    'size': size,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Opened long position for {symbol} at {entry_price}")
                
        except Exception as e:
            logger.error(f"Error executing long entry for {symbol}: {str(e)}")

    async def _execute_short_entry(self, symbol: str, size: Decimal, signal: Dict[str, Any]) -> None:
        """Execute a short position entry."""
        try:
            # Calculate entry price and order parameters
            ticker = await self._get_ticker(symbol)
            entry_price = Decimal(ticker['bid'])
            
            # Calculate stop loss and take profit levels
            stop_loss = entry_price * (1 + self.config.get('stop_loss_percent', 0.02))
            take_profit = entry_price * (1 - self.config.get('take_profit_percent', 0.04))
            
            # Place the entry order
            order = await self._place_order(
                symbol=symbol,
                side="SELL",
                quantity=size,
                price=entry_price
            )
            
            if order:
                # Place stop loss and take profit orders
                await self._place_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=size,
                    stop_price=stop_loss,
                    order_type="STOP_LOSS_LIMIT"
                )
                
                await self._place_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=size,
                    price=take_profit,
                    order_type="LIMIT"
                )
                
                # Update position tracking
                self.positions[symbol] = {
                    'side': 'SHORT',
                    'size': size,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Opened short position for {symbol} at {entry_price}")
                
        except Exception as e:
            logger.error(f"Error executing short entry for {symbol}: {str(e)}")

    async def _calculate_position_size(self, symbol: str) -> Decimal:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Decimal: Position size in base currency
        """
        try:
            # Get account balance
            balance = await self._get_account_balance()
            
            # Get risk per trade
            risk_percent = self.config.get('risk_per_trade', 0.01)  # 1% default
            risk_amount = Decimal(balance) * Decimal(risk_percent)
            
            # Get current price
            ticker = await self._get_ticker(symbol)
            price = Decimal(ticker['last'])
            
            # Calculate size based on risk
            stop_loss_percent = self.config.get('stop_loss_percent', 0.02)
            risk_per_unit = price * Decimal(stop_loss_percent)
            
            size = (risk_amount / risk_per_unit).quantize(Decimal('0.00001'))
            
            # Apply position limits
            max_position_size = Decimal(self.config.get('max_position_size', float('inf')))
            size = min(size, max_position_size)
            
            return size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return Decimal('0')

    async def _should_execute_trade(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """
        Determine if a trade should be executed based on current conditions.
        
        Args:
            symbol (str): Trading pair symbol
            signal (dict): Signal information
            
        Returns:
            bool: True if trade should be executed
        """
        try:
            # Check if symbol is already in a position
            if symbol in self.positions:
                return False
                
            # Check signal strength
            if abs(signal.get('weighted_signal', 0)) < self.config.get('min_signal_strength', 0.5):
                return False
                
            # Check signal confidence
            if signal.get('confidence', 0) < self.config.get('min_confidence', 0.7):
                return False
                
            # Check risk limits
            if not await self._is_within_risk_limits(symbol, None):
                return False
                
            # Check trading hours if configured
            if not self._is_within_trading_hours():
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in trade execution check: {str(e)}")
            return False

    async def _is_within_risk_limits(self, symbol: str, position: Optional[Dict[str, Any]]) -> bool:
        """Check if a position is within risk limits."""
        try:
            # Check maximum positions
            max_positions = self.risk_limits.get('max_positions', 5)
            if len(self.positions) >= max_positions and symbol not in self.positions:
                return False
                
            # Check maximum drawdown
            current_drawdown = await self._calculate_drawdown()
            max_drawdown = self.risk_limits.get('max_drawdown', 0.1)
            if current_drawdown > max_drawdown:
                return False
                
            # Check position concentration
            if position:
                max_concentration = self.risk_limits.get('max_position_concentration', 0.2)
                concentration = position['size'] * Decimal(position['entry_price'])
                total_value = await self._get_portfolio_value()
                
                if concentration / total_value > max_concentration:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False

    async def _place_order(self, symbol: str, side: str, quantity: Decimal,
                          price: Optional[Decimal] = None, order_type: str = "LIMIT",
                          stop_price: Optional[Decimal] = None) -> Optional[Dict[str, Any]]:
        """Place an order on the exchange."""
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "quantity": float(quantity),
                "type": order_type
            }
            
            if price:
                params["price"] = float(price)
            if stop_price:
                params["stopPrice"] = float(stop_price)
                
            order = self.client.create_order(**params)
            
            # Track the order
            self.orders[order['orderId']] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'type': order_type,
                'status': order['status'],
                'timestamp': datetime.now().isoformat()
            }
            
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing order: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    async def _get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information."""
        return self.client.get_symbol_ticker(symbol=symbol)

    async def _get_account_balance(self) -> float:
        """Get account balance in quote currency."""
        account = self.client.get_account()
        quote_currency = self.config.get('quote_currency', 'USDT')
        
        for balance in account['balances']:
            if balance['asset'] == quote_currency:
                return float(balance['free'])
                
        return 0.0

    def _is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled."""
        return (
            self.config.get('trading_enabled', False) and
            self.client is not None
        )

    def _is_within_trading_hours(self) -> bool:
        """Check if current time is within configured trading hours."""
        trading_hours = self.config.get('trading_hours', {})
        if not trading_hours:
            return True
            
        now = datetime.now()
        start_hour = trading_hours.get('start', 0)
        end_hour = trading_hours.get('end', 24)
        
        return start_hour <= now.hour < end_hour 