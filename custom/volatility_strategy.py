"""
Volatility Strategy using ATR and Bollinger Bands

This strategy uses ATR to identify periods of expanding volatility and Bollinger Bands
to identify price extremes. It generates buy signals when volatility is expanding and
price is near the lower Bollinger Band, and sell signals when volatility is expanding
and price is near the upper Bollinger Band. It implements proper stop loss and take
profit calculation based on ATR and risk-reward ratio.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas_ta as ta

# Setup logging
logger = logging.getLogger("Volatility Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class VolatilityStrategy:
    """Volatility Strategy using ATR and Bollinger Bands

    This strategy uses ATR to identify periods of expanding volatility and Bollinger Bands
    to identify price extremes.
    """

    def __init__(self, name: str = "Volatility Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "Volatility Strategy that uses ATR to identify periods of expanding volatility "
            "and Bollinger Bands to identify price extremes. It generates buy signals when "
            "volatility is expanding and price is near the lower Bollinger Band, and sell "
            "signals when volatility is expanding and price is near the upper Bollinger Band."
        )

        # Default parameters
        self.parameters = {
            'bb_period': 20,  # Bollinger Bands period
            'bb_std': 2.0,  # Bollinger Bands standard deviation
            'atr_period': 14,  # ATR period
            'atr_lookback': 5,  # Period to check if ATR is increasing
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'lower_band_threshold': 0.2,  # Threshold for price near lower band (0-1)
            'upper_band_threshold': 0.8,  # Threshold for price near upper band (0-1)
            'atr_change_threshold': 0.02,  # Minimum ATR percent change to consider volatility expanding
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
        }

        # Initialize cache for indicators
        self.indicator_cache = {}

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for the strategy

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators
        """
        try:
            # Create a copy of the data
            df = data.copy()

            # Get parameters
            bb_period = self.parameters['bb_period']
            bb_std = self.parameters['bb_std']
            atr_period = self.parameters['atr_period']
            atr_lookback = self.parameters['atr_lookback']

            # Calculate Bollinger Bands
            df['sma'] = ta.sma(df['close'], length=bb_period)
            df['std'] = ta.stdev(df['close'], length=bb_period)
            df['upper_band'] = df['sma'] + (df['std'] * bb_std)
            df['lower_band'] = df['sma'] - (df['std'] * bb_std)
            df['middle_band'] = df['sma']

            # Calculate ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

            # Calculate if ATR is increasing (volatility expansion)
            df['atr_change'] = df['atr'].pct_change(periods=atr_lookback)
            df['volatility_expanding'] = df['atr_change'] > self.parameters['atr_change_threshold']

            # Calculate price position relative to Bollinger Bands
            df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

            # Calculate if price is near the bands
            df['near_lower_band'] = df['bb_position'] < self.parameters['lower_band_threshold']
            df['near_upper_band'] = df['bb_position'] > self.parameters['upper_band_threshold']

            # Calculate Bollinger Bands width (normalized)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return original data with NaN indicators
            df = data.copy()
            for col in ['sma', 'std', 'upper_band', 'lower_band', 'middle_band', 'atr', 
                       'atr_change', 'volatility_expanding', 'bb_position', 
                       'near_lower_band', 'near_upper_band', 'bb_width']:
                df[col] = np.nan
            return df

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on volatility and Bollinger Bands

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['bb_period'] + 10:
            logger.warning("Insufficient data for volatility analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for volatility analysis'
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

            # Calculate indicators
            df = self._calculate_indicators(data)

            # Get current values
            current_close = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            current_atr_change = df['atr_change'].iloc[-1]
            current_bb_position = df['bb_position'].iloc[-1]
            current_bb_width = df['bb_width'].iloc[-1]
            volatility_expanding = df['volatility_expanding'].iloc[-1]
            near_lower_band = df['near_lower_band'].iloc[-1]
            near_upper_band = df['near_upper_band'].iloc[-1]

            # Initialize signal
            signal = {
                'symbol': symbol,
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'atr': current_atr,
                    'atr_change': current_atr_change,
                    'volatility_expanding': volatility_expanding,
                    'bb_position': current_bb_position,
                    'bb_width': current_bb_width,
                    'upper_band': df['upper_band'].iloc[-1],
                    'middle_band': df['middle_band'].iloc[-1],
                    'lower_band': df['lower_band'].iloc[-1],
                    'reason': 'No signal'
                }
            }

            # Check for buy signal
            if volatility_expanding and near_lower_band:
                # Calculate signal strength based on ATR change and BB position
                atr_strength = min(1.0, current_atr_change / (self.parameters['atr_change_threshold'] * 2))
                position_strength = min(1.0, (self.parameters['lower_band_threshold'] - current_bb_position) / self.parameters['lower_band_threshold'])
                signal_strength = (atr_strength + position_strength) / 2

                if signal_strength >= self.parameters['signal_threshold']:
                    # Calculate stop loss and take profit
                    stop_loss = current_close - (current_atr * self.parameters['atr_multiplier'])
                    risk = current_close - stop_loss
                    take_profit = current_close + (risk * self.parameters['take_profit_ratio'])

                    # Update signal
                    signal.update({
                        'direction': 'BUY',
                        'signal_strength': signal_strength,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            **signal['metadata'],
                            'reason': 'Volatility expanding and price near lower Bollinger Band',
                            'atr_strength': atr_strength,
                            'position_strength': position_strength,
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })

            # Check for sell signal
            elif volatility_expanding and near_upper_band:
                # Calculate signal strength based on ATR change and BB position
                atr_strength = min(1.0, current_atr_change / (self.parameters['atr_change_threshold'] * 2))
                position_strength = min(1.0, (current_bb_position - self.parameters['upper_band_threshold']) / (1 - self.parameters['upper_band_threshold']))
                signal_strength = (atr_strength + position_strength) / 2

                if signal_strength >= self.parameters['signal_threshold']:
                    # Calculate stop loss and take profit
                    stop_loss = current_close + (current_atr * self.parameters['atr_multiplier'])
                    risk = stop_loss - current_close
                    take_profit = current_close - (risk * self.parameters['take_profit_ratio'])

                    # Update signal
                    signal.update({
                        'direction': 'SELL',
                        'signal_strength': signal_strength,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            **signal['metadata'],
                            'reason': 'Volatility expanding and price near upper Bollinger Band',
                            'atr_strength': atr_strength,
                            'position_strength': position_strength,
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })

            return signal
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
