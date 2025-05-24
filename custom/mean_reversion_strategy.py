"""
Mean Reversion Strategy using Bollinger Bands and RSI

This strategy uses Bollinger Bands to identify overbought and oversold conditions
and RSI as a confirmation indicator. It generates buy signals when price is below
the lower Bollinger Band and RSI is below 30, and sell signals when price is above
the upper Bollinger Band and RSI is above 70. It implements proper stop loss and
take profit calculation based on ATR and risk-reward ratio.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas_ta as ta

# Setup logging
logger = logging.getLogger("Mean Reversion Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MeanReversionStrategy:
    """Mean Reversion Strategy using Bollinger Bands and RSI

    This strategy uses Bollinger Bands to identify overbought and oversold conditions
    and RSI as a confirmation indicator.
    """

    def __init__(self, name: str = "Mean Reversion Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "Mean Reversion Strategy that uses Bollinger Bands to identify overbought and oversold "
            "conditions and RSI as a confirmation indicator. It generates buy signals when price "
            "is below the lower Bollinger Band and RSI is below 30, and sell signals when price "
            "is above the upper Bollinger Band and RSI is above 70."
        )

        # Default parameters
        self.parameters = {
            'bb_period': 20,  # Bollinger Bands period
            'bb_std': 2.0,  # Bollinger Bands standard deviation
            'rsi_period': 14,  # RSI period
            'rsi_oversold': 30,  # RSI oversold threshold
            'rsi_overbought': 70,  # RSI overbought threshold
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR
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
            rsi_period = self.parameters['rsi_period']
            atr_period = self.parameters['atr_period']

            # Calculate Bollinger Bands
            df['sma'] = ta.sma(df['close'], length=bb_period)
            df['std'] = ta.stdev(df['close'], length=bb_period)
            df['upper_band'] = df['sma'] + (df['std'] * bb_std)
            df['lower_band'] = df['sma'] - (df['std'] * bb_std)
            df['middle_band'] = df['sma']

            # Calculate RSI
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)

            # Calculate ATR for stop loss
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

            # Calculate additional indicators for analysis
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']  # Normalized BB width
            df['bb_percent'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])  # Position within BB

            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return original data with NaN indicators
            df = data.copy()
            for col in ['sma', 'std', 'upper_band', 'lower_band', 'middle_band', 'rsi', 'atr', 'bb_width', 'bb_percent']:
                df[col] = np.nan
            return df

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on Bollinger Bands and RSI

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['bb_period'] + 10:
            logger.warning("Insufficient data for mean reversion analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for mean reversion analysis'
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
            current_rsi = df['rsi'].iloc[-1]
            current_upper_band = df['upper_band'].iloc[-1]
            current_lower_band = df['lower_band'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            current_bb_percent = df['bb_percent'].iloc[-1]

            # Initialize signal
            signal = {
                'symbol': symbol,
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'rsi': current_rsi,
                    'upper_band': current_upper_band,
                    'lower_band': current_lower_band,
                    'middle_band': df['middle_band'].iloc[-1],
                    'atr': current_atr,
                    'bb_percent': current_bb_percent,
                    'reason': 'No signal'
                }
            }

            # Check for buy signal
            if current_close < current_lower_band and current_rsi < self.parameters['rsi_oversold']:
                # Calculate signal strength based on how far price is below lower band and RSI is below oversold
                price_strength = min(1.0, (current_lower_band - current_close) / current_atr)
                rsi_strength = min(1.0, (self.parameters['rsi_oversold'] - current_rsi) / 10)
                signal_strength = (price_strength + rsi_strength) / 2

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
                            'reason': 'Price below lower Bollinger Band and RSI oversold',
                            'price_strength': price_strength,
                            'rsi_strength': rsi_strength,
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })

            # Check for sell signal
            elif current_close > current_upper_band and current_rsi > self.parameters['rsi_overbought']:
                # Calculate signal strength based on how far price is above upper band and RSI is above overbought
                price_strength = min(1.0, (current_close - current_upper_band) / current_atr)
                rsi_strength = min(1.0, (current_rsi - self.parameters['rsi_overbought']) / 10)
                signal_strength = (price_strength + rsi_strength) / 2

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
                            'reason': 'Price above upper Bollinger Band and RSI overbought',
                            'price_strength': price_strength,
                            'rsi_strength': rsi_strength,
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
