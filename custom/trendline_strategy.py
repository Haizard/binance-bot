"""
TrendLine Strategy Implementation

This module contains the implementation of the TrendLine strategy,
which uses gradient descent to find optimal trendlines and generates
trading signals based on trendline breakouts.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Union, List

# Import trendline functions
from src.strategies.custom.trendline_automation import (
    fit_trendlines_single,
    fit_trendlines_high_low,
    check_trend_line,
    optimize_slope
)

class TrendLineStrategy:
    """TrendLine Strategy

    This strategy uses gradient descent to find optimal trendlines and generates
    trading signals based on trendline breakouts.
    """

    def __init__(self, name="TrendLine Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name

        # Set parameters
        self.parameters = {
            'lookback': 30,  # Lookback period for trendline calculation
            'breakout_threshold': 0.005,  # Threshold for breakout detection (0.5%)
            'risk_per_trade': 0.02,  # 2% risk per trade
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'use_log_prices': True,  # Use log prices to handle scaling issues
            'verbose': True  # Print detailed logs
        }

        # Set description
        self.description = (
            "TrendLine strategy that uses gradient descent to find optimal trendlines "
            "and generates trading signals based on trendline breakouts."
        )

    def _calculate_trendlines(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate trendlines for the given data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (support_slope, support_intercept, resist_slope, resist_intercept)
        """
        # Initialize arrays for storing trendline coefficients
        lookback = self.parameters['lookback']
        n = len(data)

        support_slope = np.zeros(n)
        support_intercept = np.zeros(n)
        resist_slope = np.zeros(n)
        resist_intercept = np.zeros(n)

        # Fill with NaN for the initial lookback period
        support_slope[:lookback-1] = np.nan
        support_intercept[:lookback-1] = np.nan
        resist_slope[:lookback-1] = np.nan
        resist_intercept[:lookback-1] = np.nan

        # Apply log transformation if specified
        if self.parameters['use_log_prices']:
            high = np.log(data['high'])
            low = np.log(data['low'])
            close = np.log(data['close'])
        else:
            high = data['high']
            low = data['low']
            close = data['close']

        # Calculate trendlines for each window
        for i in range(lookback - 1, n):
            window = slice(i - lookback + 1, i + 1)
            support_coefs, resist_coefs = fit_trendlines_high_low(
                high.iloc[window].values,
                low.iloc[window].values,
                close.iloc[window].values
            )

            support_slope[i] = support_coefs[0]
            support_intercept[i] = support_coefs[1]
            resist_slope[i] = resist_coefs[0]
            resist_intercept[i] = resist_coefs[1]

        # Convert to Series
        support_slope_series = pd.Series(support_slope, index=data.index)
        support_intercept_series = pd.Series(support_intercept, index=data.index)
        resist_slope_series = pd.Series(resist_slope, index=data.index)
        resist_intercept_series = pd.Series(resist_intercept, index=data.index)

        return (
            support_slope_series,
            support_intercept_series,
            resist_slope_series,
            resist_intercept_series
        )

    def _detect_breakouts(
        self,
        data: pd.DataFrame,
        support_slope: pd.Series,
        support_intercept: pd.Series,
        resist_slope: pd.Series,
        resist_intercept: pd.Series
    ) -> pd.Series:
        """Detect breakouts from trendlines

        Args:
            data: DataFrame with OHLCV data
            support_slope: Series with support slope values
            support_intercept: Series with support intercept values
            resist_slope: Series with resistance slope values
            resist_intercept: Series with resistance intercept values

        Returns:
            Series with signal values (1 for buy, -1 for sell, 0 for neutral)
        """
        n = len(data)
        signals = np.zeros(n)

        # Apply log transformation if specified
        if self.parameters['use_log_prices']:
            close = np.log(data['close']).values
            high = np.log(data['high']).values
            low = np.log(data['low']).values
        else:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values

        # Convert Series to numpy arrays for faster access
        support_slope_arr = support_slope.values
        support_intercept_arr = support_intercept.values
        resist_slope_arr = resist_slope.values
        resist_intercept_arr = resist_intercept.values

        # Calculate breakout threshold
        threshold = self.parameters['breakout_threshold']

        # Detect breakouts
        for i in range(1, n):
            if np.isnan(support_slope_arr[i]) or np.isnan(resist_slope_arr[i]):
                continue

            # Calculate current trendline values
            x_val = i
            support_val = support_slope_arr[i] * x_val + support_intercept_arr[i]
            resist_val = resist_slope_arr[i] * x_val + resist_intercept_arr[i]

            # Check for breakouts
            if close[i] > resist_val * (1 + threshold) and close[i-1] <= resist_val:
                # Breakout above resistance line
                signals[i] = 1
            elif close[i] < support_val * (1 - threshold) and close[i-1] >= support_val:
                # Breakout below support line
                signals[i] = -1

        return pd.Series(signals, index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on the provided data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        try:
            # Ensure we have enough data
            if len(data) < self.parameters['lookback'] + 10:
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

            # Calculate trendlines
            support_slope, support_intercept, resist_slope, resist_intercept = self._calculate_trendlines(data)

            # Detect breakouts
            signals = self._detect_breakouts(
                data,
                support_slope,
                support_intercept,
                resist_slope,
                resist_intercept
            )

            # Get the most recent signal
            current_signal = signals.iloc[-1]
            current_price = data['close'].iloc[-1]

            # Generate trading signal based on the breakout
            if current_signal > 0:
                # Calculate stop loss and take profit
                stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.parameters['take_profit_ratio'])

                return {
                    'direction': 'BUY',
                    'signal_strength': 0.8,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Breakout above resistance trendline',
                        'support_slope': float(support_slope.iloc[-1]),
                        'resist_slope': float(resist_slope.iloc[-1])
                    }
                }
            elif current_signal < 0:
                # Calculate stop loss and take profit
                stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                risk = stop_loss - current_price
                take_profit = current_price - (risk * self.parameters['take_profit_ratio'])

                return {
                    'direction': 'SELL',
                    'signal_strength': 0.8,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Breakout below support trendline',
                        'support_slope': float(support_slope.iloc[-1]),
                        'resist_slope': float(resist_slope.iloc[-1])
                    }
                }
            else:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'No trendline breakout detected',
                        'support_slope': float(support_slope.iloc[-1]) if not np.isnan(support_slope.iloc[-1]) else 0,
                        'resist_slope': float(resist_slope.iloc[-1]) if not np.isnan(resist_slope.iloc[-1]) else 0
                    }
                }
        except Exception as e:
            # Handle any errors
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': f'Error generating signals: {str(e)}'
                }
            }
