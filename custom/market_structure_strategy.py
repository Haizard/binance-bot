"""
Market Structure Analysis Strategy

This strategy uses ATR-based directional change to identify significant market structure points
and creates a hierarchical view of market structure at different timeframes. It identifies
support/resistance levels based on market structure and generates signals based on market
structure changes.

The strategy includes:
- ATR-based directional change detection
- Hierarchical market structure analysis
- Support/resistance level identification
- Signal generation based on market structure changes
- Proper stop-loss and take-profit calculation based on ATR
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
import sys

# Add the market-structure-main directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
market_structure_dir = os.path.join(current_dir, 'market-structure-main')
sys.path.append(market_structure_dir)

# Import market structure components
try:
    from atr_directional_change import ATRDirectionalChange
    from hierarchical_extremes import HierarchicalExtremes
    from local_extreme import LocalExtreme
except ImportError as e:
    logging.error(f"Error importing market structure components: {e}")
    raise ImportError(f"Failed to import market structure components: {e}")

# Setup logging
logger = logging.getLogger("Market Structure Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MarketStructureStrategy:
    """Market Structure Analysis Strategy

    This strategy uses ATR-based directional change to identify significant market structure points
    and creates a hierarchical view of market structure at different timeframes.
    """

    def __init__(self, name: str = "Market Structure Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "Market Structure Analysis Strategy that uses ATR-based directional change to identify "
            "significant market structure points and creates a hierarchical view of market structure "
            "at different timeframes. It identifies support/resistance levels based on market structure "
            "and generates signals based on market structure changes."
        )

        # Default parameters
        self.parameters = {
            'atr_lookback': 14,  # ATR lookback period
            'levels': 3,  # Number of hierarchical levels
            'breakout_threshold': 0.005,  # Threshold for breakout detection (0.5%)
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR
            'use_log_prices': True,  # Use log prices to handle scaling issues
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
        }

        # Initialize market structure components
        self.hierarchical_extremes = None
        self.last_update_index = -1
        self.structures = {}
        self.cached_levels = {}

    def _initialize_components(self, data: pd.DataFrame) -> None:
        """Initialize market structure components

        Args:
            data: DataFrame with OHLCV data
        """
        try:
            # Initialize hierarchical extremes
            self.hierarchical_extremes = HierarchicalExtremes(
                levels=self.parameters['levels'],
                atr_lookback=self.parameters['atr_lookback']
            )
            self.last_update_index = -1
            logger.info(f"Initialized hierarchical extremes with {self.parameters['levels']} levels")
        except Exception as e:
            logger.error(f"Error initializing market structure components: {e}")
            raise RuntimeError(f"Failed to initialize market structure components: {e}")

    def _update_market_structure(self, data: pd.DataFrame) -> None:
        """Update market structure with new data

        Args:
            data: DataFrame with OHLCV data
        """
        if self.hierarchical_extremes is None:
            self._initialize_components(data)

        try:
            # Get price data
            high = data['high'].to_numpy()
            low = data['low'].to_numpy()
            close = data['close'].to_numpy()
            time_index = data.index

            # Update hierarchical extremes
            for i in range(self.last_update_index + 1, len(data)):
                self.hierarchical_extremes.update(i, time_index, high, low, close)

            self.last_update_index = len(data) - 1
            logger.debug(f"Updated market structure to index {self.last_update_index}")
        except Exception as e:
            logger.error(f"Error updating market structure: {e}")
            # Continue with what we have rather than failing completely
            pass

    def _get_support_resistance_levels(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Get support and resistance levels from market structure

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with support and resistance levels
        """
        if self.hierarchical_extremes is None or len(data) == 0:
            return {'support_levels': [], 'resistance_levels': []}

        try:
            # Get current price
            current_price = data['close'].iloc[-1]

            # Get support and resistance levels from hierarchical extremes
            support_levels = []
            resistance_levels = []

            # Check each level
            for level in range(self.parameters['levels']):
                # Get the most recent high and low for this level
                high = self.hierarchical_extremes.get_level_high(level)
                low = self.hierarchical_extremes.get_level_low(level)

                # Add high to resistance levels if it's above current price
                if high is not None and high.price > current_price:
                    resistance_levels.append({
                        'price': float(high.price),
                        'level': level,
                        'strength': float(level + 1) / self.parameters['levels'],
                        'index': int(high.index)
                    })

                # Add low to support levels if it's below current price
                if low is not None and low.price < current_price:
                    support_levels.append({
                        'price': float(low.price),
                        'level': level,
                        'strength': float(level + 1) / self.parameters['levels'],
                        'index': int(low.index)
                    })

            # Sort levels by price
            support_levels.sort(key=lambda x: x['price'], reverse=True)
            resistance_levels.sort(key=lambda x: x['price'])

            return {'support_levels': support_levels, 'resistance_levels': resistance_levels}
        except Exception as e:
            logger.error(f"Error getting support/resistance levels: {e}")
            return {'support_levels': [], 'resistance_levels': []}

    def _detect_breakouts(self, data: pd.DataFrame, levels: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Detect breakouts of support and resistance levels

        Args:
            data: DataFrame with OHLCV data
            levels: Dictionary with support and resistance levels

        Returns:
            Dictionary with breakout information
        """
        if len(data) < 2:
            return {'breakout': False, 'direction': 'NEUTRAL', 'level': None, 'strength': 0.0}

        try:
            # Get current and previous prices
            current_price = data['close'].iloc[-1]
            previous_price = data['close'].iloc[-2]
            current_high = data['high'].iloc[-1]
            current_low = data['low'].iloc[-1]

            # Get support and resistance levels
            support_levels = levels['support_levels']
            resistance_levels = levels['resistance_levels']

            # Check for resistance breakout
            for level in resistance_levels:
                level_price = level['price']
                # Breakout if previous price was below level and current price is above level
                if previous_price < level_price and current_price > level_price:
                    # Calculate breakout strength based on how far price moved beyond the level
                    breakout_strength = (current_price - level_price) / level_price
                    # Normalize strength to 0-1 range
                    normalized_strength = min(1.0, breakout_strength / self.parameters['breakout_threshold'])
                    
                    if normalized_strength >= self.parameters['signal_threshold']:
                        return {
                            'breakout': True,
                            'direction': 'BUY',
                            'level': level,
                            'strength': float(normalized_strength)
                        }

            # Check for support breakout
            for level in support_levels:
                level_price = level['price']
                # Breakout if previous price was above level and current price is below level
                if previous_price > level_price and current_price < level_price:
                    # Calculate breakout strength based on how far price moved beyond the level
                    breakout_strength = (level_price - current_price) / level_price
                    # Normalize strength to 0-1 range
                    normalized_strength = min(1.0, breakout_strength / self.parameters['breakout_threshold'])
                    
                    if normalized_strength >= self.parameters['signal_threshold']:
                        return {
                            'breakout': True,
                            'direction': 'SELL',
                            'level': level,
                            'strength': float(normalized_strength)
                        }

            return {'breakout': False, 'direction': 'NEUTRAL', 'level': None, 'strength': 0.0}
        except Exception as e:
            logger.error(f"Error detecting breakouts: {e}")
            return {'breakout': False, 'direction': 'NEUTRAL', 'level': None, 'strength': 0.0}

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)

        Args:
            data: DataFrame with OHLCV data
            period: ATR period

        Returns:
            ATR value
        """
        try:
            if len(data) < period + 1:
                return 0.0

            # Calculate true range
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Calculate ATR
            atr = np.mean(tr[-period:])
            
            return float(atr)
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on market structure analysis

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['atr_lookback'] + 10:
            logger.warning("Insufficient data for market structure analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for market structure analysis'
                }
            }

        try:
            # Ensure data has required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'metadata': {
                        'reason': f"Missing required columns: {missing_columns}"
                    }
                }

            # Get symbol from data if available
            symbol = data.get('symbol', ['Unknown'])[0] if 'symbol' in data else 'Unknown'

            # Apply log transform if enabled
            if self.parameters['use_log_prices']:
                data_transformed = data.copy()
                for col in ['open', 'high', 'low', 'close']:
                    data_transformed[col] = np.log(data_transformed[col])
                analysis_data = data_transformed
            else:
                analysis_data = data

            # Update market structure
            self._update_market_structure(analysis_data)

            # Get support and resistance levels
            levels = self._get_support_resistance_levels(analysis_data)

            # Detect breakouts
            breakout = self._detect_breakouts(analysis_data, levels)

            # If no breakout, return neutral signal
            if not breakout['breakout']:
                return {
                    'symbol': symbol,
                    'direction': 'NEUTRAL',
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'metadata': {
                        'reason': 'No market structure breakout detected',
                        'support_levels': levels['support_levels'],
                        'resistance_levels': levels['resistance_levels']
                    }
                }

            # Calculate entry, stop loss, and take profit
            current_price = data['close'].iloc[-1]
            atr = self._calculate_atr(data, self.parameters['atr_period'])
            
            if breakout['direction'] == 'BUY':
                entry_price = current_price
                stop_loss = entry_price - (atr * self.parameters['atr_multiplier'])
                take_profit = entry_price + (atr * self.parameters['atr_multiplier'] * self.parameters['take_profit_ratio'])
                reason = f"Bullish breakout of level {breakout['level']['price']:.2f} (Level {breakout['level']['level']})"
            else:  # SELL
                entry_price = current_price
                stop_loss = entry_price + (atr * self.parameters['atr_multiplier'])
                take_profit = entry_price - (atr * self.parameters['atr_multiplier'] * self.parameters['take_profit_ratio'])
                reason = f"Bearish breakout of level {breakout['level']['price']:.2f} (Level {breakout['level']['level']})"

            # Return signal
            return {
                'symbol': symbol,
                'direction': breakout['direction'],
                'signal_strength': breakout['strength'],
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'metadata': {
                    'reason': reason,
                    'breakout_level': breakout['level'],
                    'support_levels': levels['support_levels'],
                    'resistance_levels': levels['resistance_levels'],
                    'atr': float(atr)
                }
            }
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': f"Error generating signals: {str(e)}"
                }
            }
