"""
Trend Following Strategy using Moving Averages and MACD

This strategy uses fast and slow moving averages to identify trends and MACD as a
confirmation indicator. It generates buy signals when fast MA crosses above slow MA
and MACD is positive, and sell signals when fast MA crosses below slow MA and MACD
is negative. It implements proper stop loss and take profit calculation based on ATR
and risk-reward ratio.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas_ta as ta

# Setup logging
logger = logging.getLogger("Trend Following Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TrendFollowingStrategy:
    """Trend Following Strategy using Moving Averages and MACD

    This strategy uses fast and slow moving averages to identify trends and MACD as a
    confirmation indicator.
    """

    def __init__(self, name: str = "Trend Following Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "Trend Following Strategy that uses fast and slow moving averages to identify "
            "trends and MACD as a confirmation indicator. It generates buy signals when "
            "fast MA crosses above slow MA and MACD is positive, and sell signals when "
            "fast MA crosses below slow MA and MACD is negative."
        )

        # Default parameters
        self.parameters = {
            'fast_ma_period': 20,  # Fast moving average period
            'slow_ma_period': 50,  # Slow moving average period
            'ma_type': 'ema',  # Type of moving average ('sma', 'ema', 'wma')
            'macd_fast': 12,  # MACD fast period
            'macd_slow': 26,  # MACD slow period
            'macd_signal': 9,  # MACD signal period
            'atr_period': 14,  # ATR period
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
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
            fast_ma_period = self.parameters['fast_ma_period']
            slow_ma_period = self.parameters['slow_ma_period']
            ma_type = self.parameters['ma_type']
            macd_fast = self.parameters['macd_fast']
            macd_slow = self.parameters['macd_slow']
            macd_signal = self.parameters['macd_signal']
            atr_period = self.parameters['atr_period']

            # Calculate Moving Averages based on type
            if ma_type == 'sma':
                df['fast_ma'] = ta.sma(df['close'], length=fast_ma_period)
                df['slow_ma'] = ta.sma(df['close'], length=slow_ma_period)
            elif ma_type == 'ema':
                df['fast_ma'] = ta.ema(df['close'], length=fast_ma_period)
                df['slow_ma'] = ta.ema(df['close'], length=slow_ma_period)
            elif ma_type == 'wma':
                df['fast_ma'] = ta.wma(df['close'], length=fast_ma_period)
                df['slow_ma'] = ta.wma(df['close'], length=slow_ma_period)
            else:
                # Default to EMA
                df['fast_ma'] = ta.ema(df['close'], length=fast_ma_period)
                df['slow_ma'] = ta.ema(df['close'], length=slow_ma_period)

            # Calculate MACD
            macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            df['macd_line'] = macd[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
            df['macd_signal'] = macd[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']
            df['macd_histogram'] = macd[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}']

            # Calculate ATR for stop loss
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

            # Calculate MA crossover
            df['ma_crossover'] = 0
            df.loc[(df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)), 'ma_crossover'] = 1
            df.loc[(df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)), 'ma_crossover'] = -1

            # Calculate MACD crossover
            df['macd_crossover'] = 0
            df.loc[(df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)), 'macd_crossover'] = 1
            df.loc[(df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)), 'macd_crossover'] = -1

            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return original data with NaN indicators
            df = data.copy()
            for col in ['fast_ma', 'slow_ma', 'macd_line', 'macd_signal', 'macd_histogram', 'atr', 'ma_crossover', 'macd_crossover']:
                df[col] = np.nan
            return df

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on Moving Averages and MACD

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['slow_ma_period'] + 10:
            logger.warning("Insufficient data for trend following analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for trend following analysis'
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
            current_fast_ma = df['fast_ma'].iloc[-1]
            current_slow_ma = df['slow_ma'].iloc[-1]
            current_macd_line = df['macd_line'].iloc[-1]
            current_macd_signal = df['macd_signal'].iloc[-1]
            current_macd_histogram = df['macd_histogram'].iloc[-1]
            current_ma_crossover = df['ma_crossover'].iloc[-1]
            current_macd_crossover = df['macd_crossover'].iloc[-1]

            # Initialize signal
            signal = {
                'symbol': symbol,
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'fast_ma': current_fast_ma,
                    'slow_ma': current_slow_ma,
                    'macd_line': current_macd_line,
                    'macd_signal': current_macd_signal,
                    'macd_histogram': current_macd_histogram,
                    'ma_crossover': current_ma_crossover,
                    'macd_crossover': current_macd_crossover,
                    'atr': current_atr,
                    'reason': 'No signal'
                }
            }

            # Check for buy signal
            if current_ma_crossover == 1 and current_macd_line > 0:
                # Calculate signal strength based on MACD value and histogram
                macd_strength = min(1.0, current_macd_line / (current_atr * 0.1))
                histogram_strength = min(1.0, current_macd_histogram / (current_atr * 0.05))
                signal_strength = (macd_strength + histogram_strength) / 2

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
                            'reason': 'Fast MA crossed above Slow MA and MACD is positive',
                            'macd_strength': macd_strength,
                            'histogram_strength': histogram_strength,
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })

            # Check for sell signal
            elif current_ma_crossover == -1 and current_macd_line < 0:
                # Calculate signal strength based on MACD value and histogram
                macd_strength = min(1.0, abs(current_macd_line) / (current_atr * 0.1))
                histogram_strength = min(1.0, abs(current_macd_histogram) / (current_atr * 0.05))
                signal_strength = (macd_strength + histogram_strength) / 2

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
                            'reason': 'Fast MA crossed below Slow MA and MACD is negative',
                            'macd_strength': macd_strength,
                            'histogram_strength': histogram_strength,
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
