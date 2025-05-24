"""
Dip detection and execution module for the TradeAgent.
Implements market dip detection and execution strategies.
"""
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
import numpy as np
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class DipExecutorModule:
    """
    Module responsible for detecting and executing trades on market dips.
    Integrates with TradeAgent for order execution and position management.
    """
    
    def __init__(self, trade_agent):
        """
        Initialize the DipExecutorModule.
        
        Args:
            trade_agent: Reference to the parent TradeAgent instance
        """
        self.trade_agent = trade_agent
        self.dip_config = {}
        self.dip_states = {}  # Tracks dip detection state per symbol
        self.price_history = {}  # Maintains recent price history for dip detection
        
    async def setup(self) -> None:
        """Set up the dip executor module."""
        # Subscribe to price updates via trade agent
        await self.trade_agent.subscribe("market.price.*")
        await self.trade_agent.subscribe("config.dip.update")
        
        # Initialize configuration with Decimal values
        self.dip_config = {
            'min_dip_percent': Decimal('2.0'),
            'recovery_percent': Decimal('0.5'),
            'volume_increase_factor': Decimal('1.5'),
            'max_position_size': Decimal('0.1'),
            'price_window': 24,
            'enabled_pairs': [],
            'cooldown_period': 4,
        }
    
    async def handle_message(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle incoming messages related to dip execution.
        
        Args:
            topic (str): Message topic
            data (dict): Message data
        """
        if topic.startswith("market.price."):
            symbol = topic.split('.')[-1]
            await self._update_price_history(symbol, data)
            await self._check_for_dip(symbol)
        elif topic == "config.dip.update":
            await self._handle_config_update(data)
    
    async def _update_price_history(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update price history for dip detection."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        price_data = {
            'timestamp': datetime.now(),
            'price': Decimal(str(data['price'])),
            'volume': Decimal(str(data['volume']))
        }
        
        self.price_history[symbol].append(price_data)
        
        # Remove old data outside the window
        window_start = datetime.now() - timedelta(hours=self.dip_config['price_window'])
        self.price_history[symbol] = [
            d for d in self.price_history[symbol]
            if d['timestamp'] > window_start
        ]
    
    async def _check_for_dip(self, symbol: str) -> None:
        """
        Check if current price action constitutes a dip worth trading.
        
        Args:
            symbol (str): Trading pair symbol
        """
        if not self._is_dip_trading_enabled(symbol):
            return
            
        prices = self.price_history[symbol]
        if not prices or len(prices) < 2:
            return
            
        current_price = prices[-1]['price']
        recent_max = max(p['price'] for p in prices[:-1])
        
        # Calculate price drop
        price_drop = ((recent_max - current_price) / recent_max) * Decimal('100')
        
        # Check if in cooldown period
        if self._is_in_cooldown(symbol):
            return
            
        # Check for significant dip
        if price_drop >= self.dip_config['min_dip_percent']:
            # Verify volume increase
            if await self._verify_volume_surge(symbol):
                # Check for potential recovery
                if await self._verify_recovery_signs(symbol):
                    await self._execute_dip_trade(symbol, current_price, float(price_drop))
    
    async def _verify_volume_surge(self, symbol: str) -> bool:
        """Verify if there's a volume surge during the dip."""
        recent_volumes = [p['volume'] for p in self.price_history[symbol][-5:]]
        avg_volume = sum(recent_volumes) / Decimal(str(len(recent_volumes)))
        current_volume = recent_volumes[-1]
        volume_threshold = avg_volume * Decimal(str(self.dip_config['volume_increase_factor']))
        
        return current_volume > volume_threshold
    
    async def _verify_recovery_signs(self, symbol: str) -> bool:
        """Check for signs of price recovery after the dip."""
        recent_prices = [p['price'] for p in self.price_history[symbol][-5:]]
        
        # Calculate price momentum
        price_changes = np.diff([float(p) for p in recent_prices])
        
        # Check if recent price changes show recovery
        if len(price_changes) >= 2:
            min_price = min(recent_prices)
            recovery_percent = (recent_prices[-1] - min_price) / min_price * Decimal('100')
            return recovery_percent >= self.dip_config['recovery_percent']
        
        return False
    
    async def _execute_dip_trade(self, symbol: str, price: Decimal, dip_percent: float) -> None:
        """
        Execute a trade based on detected dip.
        
        Args:
            symbol (str): Trading pair symbol
            price (Decimal): Current price
            dip_percent (float): Detected dip percentage
        """
        try:
            # Calculate position size based on dip severity
            account_balance = await self.trade_agent._get_account_balance()
            max_position = Decimal(str(account_balance)) * Decimal(str(self.dip_config['max_position_size']))
            
            # Adjust position size based on dip severity
            position_size = max_position * (dip_percent / self.dip_config['min_dip_percent'])
            position_size = min(position_size, max_position)
            
            # Create dip trade signal
            signal = {
                'signal': 1,  # Buy signal
                'strength': dip_percent / self.dip_config['min_dip_percent'],
                'type': 'dip',
                'metadata': {
                    'dip_percent': dip_percent,
                    'detection_time': datetime.now().isoformat()
                }
            }
            
            # Execute the trade through trade agent
            await self.trade_agent._execute_long_entry(symbol, position_size, signal)
            
            # Update dip state for cooldown
            self.dip_states[symbol] = {
                'last_trade_time': datetime.now(),
                'dip_percent': dip_percent
            }
            
            logger.info(f"Executed dip trade for {symbol} at {price} (Dip: {dip_percent:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error executing dip trade for {symbol}: {str(e)}")
    
    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """Handle dip configuration updates."""
        if 'dip_config' in data:
            self.dip_config.update(data['dip_config'])
            logger.info("Updated dip execution configuration")
    
    def _is_dip_trading_enabled(self, symbol: str) -> bool:
        """Check if dip trading is enabled for the symbol."""
        return (
            symbol in self.dip_config['enabled_pairs'] and
            self.trade_agent._is_trading_enabled() and
            self.trade_agent._is_within_trading_hours()
        )
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in post-trade cooldown period."""
        if symbol not in self.dip_states:
            return False
            
        last_trade = self.dip_states[symbol]['last_trade_time']
        cooldown_end = last_trade + timedelta(hours=self.dip_config['cooldown_period'])
        
        return datetime.now() < cooldown_end 