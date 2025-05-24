"""
Rolling Window Extremes Strategy

This strategy uses an efficient rolling window approach to identify local tops and bottoms
in price data. It provides a more efficient alternative to scipy's argrelextrema function
and can be used as a foundation for other pattern-based strategies.

The strategy generates trading signals based on the detection of local extremes and
their confirmation, with proper stop-loss and take-profit calculation based on ATR.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Any, Optional, Tuple
import logging

class RollingWindowStrategy:
    """
    Rolling Window Extremes Strategy
    
    This strategy uses an efficient rolling window approach to identify local tops and bottoms
    in price data. It provides a more efficient alternative to scipy's argrelextrema function
    and can be used as a foundation for other pattern-based strategies.
    
    Parameters:
    -----------
    order : int
        Order parameter for extremes detection (default: 5)
    use_log_prices : bool
        Whether to use log prices for scaling (default: True)
    risk_per_trade : float
        Risk per trade as a decimal (default: 0.02 or 2%)
    take_profit_ratio : float
        Risk:Reward ratio (default: 1.5)
    atr_period : int
        Period for ATR calculation (default: 14)
    atr_multiplier : float
        Multiplier for ATR (default: 2.0)
    signal_threshold : float
        Minimum signal strength to generate a trade (default: 0.5)
    """
    
    def __init__(self, name: str = "Rolling Window Extremes Strategy"):
        """Initialize the strategy
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        
        # Set default parameters
        self.parameters = {
            'order': 5,  # Order parameter for extremes detection
            'use_log_prices': True,  # Use log prices for scaling
            'risk_per_trade': 0.02,  # 2% risk per trade
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            'trend_lookback': 20,  # Lookback period for trend determination
            'verbose': True  # Print detailed logs
        }
        
        # Set description
        self.description = (
            "Rolling Window Extremes strategy that uses an efficient rolling window approach "
            "to identify local tops and bottoms in price data. It provides a more efficient "
            "alternative to scipy's argrelextrema function and can be used as a foundation "
            "for other pattern-based strategies."
        )
    
    def rw_top(self, data: np.ndarray, curr_index: int, order: int) -> bool:
        """Check if there is a local top at the current index
        
        Args:
            data: Array of price data
            curr_index: Current index to check
            order: Order parameter for extremes detection
            
        Returns:
            True if there is a local top, False otherwise
        """
        if curr_index < order * 2 + 1:
            return False

        top = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] > v or data[k - i] > v:
                top = False
                break
        
        return top
    
    def rw_bottom(self, data: np.ndarray, curr_index: int, order: int) -> bool:
        """Check if there is a local bottom at the current index
        
        Args:
            data: Array of price data
            curr_index: Current index to check
            order: Order parameter for extremes detection
            
        Returns:
            True if there is a local bottom, False otherwise
        """
        if curr_index < order * 2 + 1:
            return False

        bottom = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] < v or data[k - i] < v:
                bottom = False
                break
        
        return bottom
    
    def rw_extremes(self, data: np.ndarray, order: int) -> Tuple[List, List]:
        """Find local tops and bottoms using rolling window approach
        
        Args:
            data: Array of price data
            order: Order parameter for extremes detection
            
        Returns:
            Tuple of (tops, bottoms) where each element is a list of [confirmation_index, extreme_index, price]
        """
        # Rolling window local tops and bottoms
        tops = []
        bottoms = []
        for i in range(len(data)):
            if self.rw_top(data, i, order):
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, i - order, data[i - order]]
                tops.append(top)
            
            if self.rw_bottom(data, i, order):
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, i - order, data[i - order]]
                bottoms.append(bottom)
        
        return tops, bottoms
    
    def determine_trend(self, data: pd.DataFrame) -> str:
        """Determine the current trend using simple moving averages
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        lookback = self.parameters['trend_lookback']
        if len(data) < lookback * 2:
            return 'sideways'
        
        # Calculate short and long moving averages
        short_ma = data['close'].rolling(window=lookback).mean()
        long_ma = data['close'].rolling(window=lookback * 2).mean()
        
        # Get the most recent values
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        
        # Determine trend
        if current_short > current_long * 1.01:  # 1% buffer
            return 'uptrend'
        elif current_short < current_long * 0.99:  # 1% buffer
            return 'downtrend'
        else:
            return 'sideways'
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on rolling window extremes
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Ensure we have enough data
            if len(data) < self.parameters['order'] * 4:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'Not enough data for rolling window analysis'
                    }
                }
            
            # Get parameters
            order = self.parameters['order']
            use_log_prices = self.parameters['use_log_prices']
            
            # Extract close prices
            close_prices = data['close'].to_numpy()
            
            # Apply log transform if enabled
            if use_log_prices:
                close_prices = np.log(close_prices)
            
            # Find extremes
            tops, bottoms = self.rw_extremes(close_prices, order)
            
            # Calculate ATR for stop loss
            atr = ta.atr(
                data['high'], data['low'], data['close'], 
                self.parameters['atr_period']
            ).iloc[-1]
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Determine current trend
            trend = self.determine_trend(data)
            
            # Check if we have any recent extremes
            if not tops and not bottoms:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'No extremes detected',
                        'trend': trend
                    }
                }
            
            # Get the most recent extreme
            last_top = tops[-1] if tops else [-1, -1, -1]
            last_bottom = bottoms[-1] if bottoms else [-1, -1, -1]
            
            # Check if the extreme was confirmed in the last candle
            last_candle_index = len(close_prices) - 1
            
            # Generate signal based on the most recent extreme and trend
            if last_top[0] == last_candle_index and trend == 'downtrend':
                # Top confirmed in the last candle during a downtrend - SELL signal
                # Calculate stop loss and take profit
                stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                risk = stop_loss - current_price
                take_profit = current_price - (risk * self.parameters['take_profit_ratio'])
                
                return {
                    'direction': 'SELL',
                    'signal_strength': 0.7,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Local top confirmed in downtrend',
                        'top_price': np.exp(last_top[2]) if use_log_prices else last_top[2],
                        'top_index': last_top[1],
                        'trend': trend
                    }
                }
            elif last_bottom[0] == last_candle_index and trend == 'uptrend':
                # Bottom confirmed in the last candle during an uptrend - BUY signal
                # Calculate stop loss and take profit
                stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.parameters['take_profit_ratio'])
                
                return {
                    'direction': 'BUY',
                    'signal_strength': 0.7,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Local bottom confirmed in uptrend',
                        'bottom_price': np.exp(last_bottom[2]) if use_log_prices else last_bottom[2],
                        'bottom_index': last_bottom[1],
                        'trend': trend
                    }
                }
            else:
                # No signal
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'No extreme confirmed in the last candle with matching trend',
                        'last_top_confirmation': last_top[0],
                        'last_bottom_confirmation': last_bottom[0],
                        'current_index': last_candle_index,
                        'trend': trend
                    }
                }
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            if self.parameters['verbose']:
                logging.error(f"Error in rolling window signal generation: {str(e)}")
                logging.error(error_details)
            
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': f'Error: {str(e)}',
                    'error_details': error_details
                }
            }
