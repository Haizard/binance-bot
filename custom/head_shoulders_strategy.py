"""
Head and Shoulders Strategy Implementation

This module contains the implementation of the Head and Shoulders strategy,
which identifies head and shoulders chart patterns (both regular and inverted)
and generates trading signals based on pattern completion.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Union, List, Optional
from collections import deque

# Import head and shoulders functions
from src.strategies.custom.head_shoulders import (
    find_hs_patterns,
    HSPattern,
    rw_top,
    rw_bottom,
    check_hs_pattern,
    check_ihs_pattern
)

class HeadShouldersStrategy:
    """Head and Shoulders Strategy

    This strategy identifies head and shoulders chart patterns (both regular and inverted)
    and generates trading signals based on pattern completion.
    """

    def __init__(self, name="Head and Shoulders Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name

        # Set parameters
        self.parameters = {
            'order': 6,  # Order parameter for extrema detection
            'r2_threshold': 0.5,  # R-squared threshold for pattern quality
            'early_find': False,  # Whether to find patterns early (before breakout)
            'risk_per_trade': 0.02,  # 2% risk per trade
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'use_log_prices': True,  # Use log prices to handle scaling issues
            'verbose': True  # Print detailed logs
        }

        # Set description
        self.description = (
            "Head and Shoulders strategy that identifies head and shoulders chart patterns "
            "(both regular and inverted) and generates trading signals based on pattern completion. "
            "It uses a rolling window approach to detect extrema points and validates patterns "
            "based on geometric properties and quality metrics."
        )

    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for pattern detection

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Numpy array with preprocessed price data
        """
        # Use close prices
        prices = data['close'].values

        # Apply log transformation if enabled
        if self.parameters['use_log_prices']:
            prices = np.log(prices)

        return prices

    def _find_patterns(self, data: pd.DataFrame) -> Tuple[List[HSPattern], List[HSPattern]]:
        """Find head and shoulders patterns in the data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (regular patterns, inverted patterns)
        """
        # Preprocess data
        prices = self._preprocess_data(data)

        # Find patterns
        hs_patterns, ihs_patterns = find_hs_patterns(
            prices,
            order=self.parameters['order'],
            early_find=self.parameters['early_find']
        )

        # Filter patterns by quality
        if self.parameters['r2_threshold'] > 0:
            hs_patterns = [p for p in hs_patterns if p.pattern_r2 >= self.parameters['r2_threshold']]
            ihs_patterns = [p for p in ihs_patterns if p.pattern_r2 >= self.parameters['r2_threshold']]

        return hs_patterns, ihs_patterns

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on the provided data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        try:
            # Ensure we have enough data
            if len(data) < 30:  # Need at least 30 bars for pattern detection
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'Not enough data for analysis'
                    }
                }

            # Calculate ATR for stop loss
            if 'atr' in data.columns:
                atr = data['atr'].iloc[-1]
            else:
                atr = ta.atr(data['high'], data['low'], data['close'], length=self.parameters['atr_period']).iloc[-1]

            # Find patterns
            hs_patterns, ihs_patterns = self._find_patterns(data)

            # Check if we have any patterns
            if not hs_patterns and not ihs_patterns:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'No patterns detected'
                    }
                }

            # Get the most recent pattern
            current_price = data['close'].iloc[-1]
            
            # Check for regular head and shoulders (bearish)
            if hs_patterns:
                latest_hs = hs_patterns[-1]
                
                # Check if the pattern is recent (breakout happened in the last 3 bars)
                if len(data) - latest_hs.break_i <= 3:
                    # Calculate stop loss and take profit
                    stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * self.parameters['take_profit_ratio'])
                    
                    return {
                        'direction': 'SELL',
                        'signal_strength': min(0.9, latest_hs.pattern_r2),  # Use pattern quality as signal strength
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            'reason': 'Regular Head and Shoulders pattern (bearish)',
                            'pattern_r2': float(latest_hs.pattern_r2),
                            'pattern_width': int(latest_hs.head_width),
                            'pattern_height': float(latest_hs.head_height)
                        }
                    }
            
            # Check for inverted head and shoulders (bullish)
            if ihs_patterns:
                latest_ihs = ihs_patterns[-1]
                
                # Check if the pattern is recent (breakout happened in the last 3 bars)
                if len(data) - latest_ihs.break_i <= 3:
                    # Calculate stop loss and take profit
                    stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * self.parameters['take_profit_ratio'])
                    
                    return {
                        'direction': 'BUY',
                        'signal_strength': min(0.9, latest_ihs.pattern_r2),  # Use pattern quality as signal strength
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            'reason': 'Inverted Head and Shoulders pattern (bullish)',
                            'pattern_r2': float(latest_ihs.pattern_r2),
                            'pattern_width': int(latest_ihs.head_width),
                            'pattern_height': float(latest_ihs.head_height)
                        }
                    }
            
            # No recent patterns
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': 'No recent patterns detected',
                    'total_patterns': len(hs_patterns) + len(ihs_patterns)
                }
            }
            
        except Exception as e:
            # Handle any exceptions
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': f'Error in pattern detection: {str(e)}'
                }
            }
