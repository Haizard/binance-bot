"""
🌙 Moon Dev's Market Profile Strategy
Strategy that uses market profile concepts to identify significant price levels and generate signals.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, List, Tuple, Optional
import logging

from src.strategies.python_base_strategy import PythonBaseStrategy
from src.strategies.custom.mp_support_resist import find_levels, support_resistance_levels, sr_penetration_signal

class MarketProfileStrategy(PythonBaseStrategy):
    """Market Profile Strategy
    
    This strategy uses kernel density estimation to identify significant price levels
    and generates signals when price breaks through these levels.
    
    The strategy:
    1. Uses kernel density estimation to create a market profile
    2. Identifies significant price levels (support/resistance) based on volume distribution
    3. Generates signals when price breaks through these levels
    
    Parameters:
    - lookback_period: Number of bars to look back for market profile calculation
    - first_weight: Weight for the first bar in the lookback period (0-1)
    - atr_multiplier: Multiplier for ATR to determine bandwidth for kernel density estimation
    - prominence_threshold: Threshold for peak prominence (0-1)
    - risk_per_trade: Percentage of account to risk per trade (0-1)
    - atr_period: Period for ATR calculation (for stop loss)
    - atr_stop_multiplier: Multiplier for ATR (for stop loss)
    - take_profit_ratio: Ratio of risk for take profit (e.g., 2 = 2:1 reward:risk)
    - signal_threshold: Minimum signal strength to generate a trade (0-1)
    """
    
    def __init__(self, name="Market Profile Strategy"):
        """Initialize the strategy
        
        Args:
            name: Name of the strategy
        """
        super().__init__(name)
        
        # Set default parameters
        self.set_parameters({
            # Market Profile parameters
            'lookback_period': 100,
            'first_weight': 0.1,
            'atr_multiplier': 3.0,
            'prominence_threshold': 0.25,
            
            # Risk management parameters
            'risk_per_trade': 0.02,  # 2% risk per trade
            'atr_period': 14,
            'atr_stop_multiplier': 2.0,
            'take_profit_ratio': 2.0,  # Risk:Reward ratio
            
            # Signal parameters
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            
            # Other settings
            'verbose': True,
            'cache_levels': True  # Cache levels to avoid recalculating for the same data
        })
        
        # Set description
        self.set_description(
            "Market Profile strategy that uses kernel density estimation to identify "
            "significant price levels and generates signals when price breaks through these levels. "
            "It identifies support and resistance based on volume distribution."
        )
        
        # Set supported timeframes
        for tf in ['1H', '4H', '1D']:
            self.add_timeframe(tf)
            
        # Set supported symbols
        for symbol in ['BTCUSD', 'ETHUSD']:
            self.add_symbol(symbol)
            
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self._levels_cache = {}
        
    def generate_signals(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate trading signals based on market profile levels
        
        Args:
            data: DataFrame with OHLCV data
            parameters: Strategy parameters (optional, uses default if not provided)
            
        Returns:
            Dictionary with signal information
        """
        # Use provided parameters or default parameters
        params = parameters if parameters is not None else self.parameters
        
        try:
            # Validate data
            self._validate_data(data)
            
            # Get parameters
            lookback = params['lookback_period']
            first_w = params['first_weight']
            atr_mult = params['atr_multiplier']
            prom_thresh = params['prominence_threshold']
            signal_threshold = params['signal_threshold']
            verbose = params['verbose']
            
            # Check if we have enough data
            if len(data) < lookback + 10:
                if verbose:
                    self.logger.warning(f"Not enough data for market profile calculation. Need at least {lookback + 10} bars.")
                return {'direction': 'NEUTRAL', 'signal_strength': 0.0}
            
            # Calculate market profile levels
            cache_key = f"{len(data)}_{lookback}_{first_w}_{atr_mult}_{prom_thresh}"
            if params['cache_levels'] and cache_key in self._levels_cache:
                levels = self._levels_cache[cache_key]
            else:
                try:
                    levels = support_resistance_levels(
                        data, lookback, first_w=first_w, atr_mult=atr_mult, prom_thresh=prom_thresh
                    )
                    if params['cache_levels']:
                        self._levels_cache[cache_key] = levels
                except Exception as e:
                    self.logger.error(f"Error calculating market profile levels: {str(e)}")
                    # Fallback to simpler calculation with default parameters
                    levels = support_resistance_levels(data, lookback)
            
            # Generate signals based on level penetration
            signal_value = sr_penetration_signal(data, levels)[-1]
            
            # Get current price and ATR
            current_price = data['close'].iloc[-1]
            atr = ta.atr(data['high'], data['low'], data['close'], params['atr_period']).iloc[-1]
            
            # If signal is below threshold, return neutral
            if abs(signal_value) < signal_threshold:
                return {'direction': 'NEUTRAL', 'signal_strength': 0.0}
            
            # Calculate stop loss and take profit
            if signal_value > 0:  # BUY signal
                stop_loss = current_price - (atr * params['atr_stop_multiplier'])
                risk = current_price - stop_loss
                take_profit = current_price + (risk * params['take_profit_ratio'])
                
                return {
                    'direction': 'BUY',
                    'signal_strength': abs(signal_value),
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Price broke above significant resistance level',
                        'levels': [level for level in levels[-1] if level is not None] if levels[-1] is not None else [],
                        'atr': atr
                    }
                }
            elif signal_value < 0:  # SELL signal
                stop_loss = current_price + (atr * params['atr_stop_multiplier'])
                risk = stop_loss - current_price
                take_profit = current_price - (risk * params['take_profit_ratio'])
                
                return {
                    'direction': 'SELL',
                    'signal_strength': abs(signal_value),
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Price broke below significant support level',
                        'levels': [level for level in levels[-1] if level is not None] if levels[-1] is not None else [],
                        'atr': atr
                    }
                }
            else:
                return {'direction': 'NEUTRAL', 'signal_strength': 0.0}
                
        except Exception as e:
            self.logger.error(f"Error generating market profile signals: {str(e)}")
            return {'direction': 'NEUTRAL', 'signal_strength': 0.0, 'error': str(e)}
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data
        
        Args:
            data: DataFrame with OHLCV data
            
        Raises:
            ValueError: If data is invalid
        """
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(data) < 10:
            raise ValueError(f"Not enough data points. Got {len(data)}, need at least 10.")
            
    def visualize(self, data: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Visualize market profile levels and signals
        
        Args:
            data: DataFrame with OHLCV data
            parameters: Strategy parameters (optional, uses default if not provided)
            
        Returns:
            Dictionary with visualization information
        """
        # This method would be implemented to create visualizations
        # For now, we'll return a placeholder
        return {'message': 'Visualization not implemented yet'}


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load sample data
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    
    # Create strategy
    strategy = MarketProfileStrategy()
    
    # Generate signals
    signal = strategy.generate_signals(data)
    
    print(f"Signal: {signal}")
