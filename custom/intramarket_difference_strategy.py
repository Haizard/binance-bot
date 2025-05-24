"""
Intramarket Difference Strategy

This strategy compares price movements between related markets (e.g., BTC and ETH)
and identifies divergences that may lead to mean reversion. It uses the CMMA
(Close Minus Moving Average) indicator to normalize price movements and generates
signals when the difference between two assets exceeds a threshold and then reverts.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas_ta as ta

# Setup logging
logger = logging.getLogger("Intramarket Difference Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class IntramarketDifferenceStrategy:
    """Intramarket Difference Strategy

    This strategy compares price movements between related markets (e.g., BTC and ETH)
    and identifies divergences that may lead to mean reversion.
    """

    def __init__(self, name: str = "Intramarket Difference Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "Intramarket Difference Strategy that compares price movements between related "
            "markets (e.g., BTC and ETH) and identifies divergences that may lead to mean "
            "reversion. It uses the CMMA (Close Minus Moving Average) indicator to normalize "
            "price movements and generates signals when the difference between two assets "
            "exceeds a threshold and then reverts."
        )

        # Default parameters
        self.parameters = {
            'lookback': 24,  # Lookback period for moving average
            'threshold': 0.25,  # Threshold for signal generation
            'atr_lookback': 168,  # ATR lookback period for normalization
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            'primary_symbol': 'ETH',  # Primary symbol (e.g., ETH)
            'reference_symbol': 'BTC',  # Reference symbol (e.g., BTC)
            'use_log_returns': True,  # Whether to use log returns for calculation
        }

        # Initialize cache for indicators
        self.indicator_cache = {}

    def _calculate_cmma(self, data: pd.DataFrame, symbol: str, lookback: int, atr_lookback: int) -> pd.Series:
        """Calculate CMMA (Close Minus Moving Average) indicator

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to calculate CMMA for
            lookback: Lookback period for moving average
            atr_lookback: ATR lookback period for normalization

        Returns:
            Series with CMMA values
        """
        try:
            # Filter data for the specified symbol
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.Series(np.nan, index=data.index)
            
            # Calculate ATR
            atr = ta.atr(symbol_data['high'], symbol_data['low'], symbol_data['close'], atr_lookback)
            
            # Calculate moving average
            ma = symbol_data['close'].rolling(lookback).mean()
            
            # Calculate CMMA
            cmma = (symbol_data['close'] - ma) / (atr * np.sqrt(lookback))
            
            return cmma
        except Exception as e:
            logger.error(f"Error calculating CMMA for {symbol}: {e}")
            return pd.Series(np.nan, index=data.index)

    def _threshold_revert_signal(self, ind: pd.Series, threshold: float) -> np.ndarray:
        """Generate signals based on threshold crossings and reversals

        Args:
            ind: Series with indicator values
            threshold: Threshold for signal generation

        Returns:
            Array with signal values (1 for buy, -1 for sell, 0 for neutral)
        """
        try:
            # Initialize signal array
            signal = np.zeros(len(ind))
            position = 0
            
            # Generate signals
            for i in range(len(ind)):
                if ind.iloc[i] > threshold:
                    position = 1
                if ind.iloc[i] < -threshold:
                    position = -1

                if position == 1 and ind.iloc[i] <= 0:
                    position = 0
                
                if position == -1 and ind.iloc[i] >= 0:
                    position = 0

                signal[i] = position
            
            return signal
        except Exception as e:
            logger.error(f"Error generating threshold revert signal: {e}")
            return np.zeros(len(ind))

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on intramarket differences

        Args:
            data: DataFrame with OHLCV data for multiple symbols

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['lookback'] + 10:
            logger.warning("Insufficient data for intramarket difference analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for intramarket difference analysis'
                }
            }

        try:
            # Ensure data has required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
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

            # Get parameters
            lookback = self.parameters['lookback']
            threshold = self.parameters['threshold']
            atr_lookback = self.parameters['atr_lookback']
            primary_symbol = self.parameters['primary_symbol']
            reference_symbol = self.parameters['reference_symbol']
            
            # Check if both symbols are present in the data
            symbols = data['symbol'].unique()
            if primary_symbol not in symbols or reference_symbol not in symbols:
                logger.warning(f"Missing required symbols: {primary_symbol} or {reference_symbol}")
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'metadata': {
                        'reason': f"Missing required symbols: {primary_symbol} or {reference_symbol}"
                    }
                }
            
            # Calculate CMMA for both symbols
            primary_cmma = self._calculate_cmma(data, primary_symbol, lookback, atr_lookback)
            reference_cmma = self._calculate_cmma(data, reference_symbol, lookback, atr_lookback)
            
            # Calculate intermarket difference
            intermarket_diff = primary_cmma - reference_cmma
            
            # Generate signals
            signal_values = self._threshold_revert_signal(intermarket_diff, threshold)
            
            # Get current values
            primary_data = data[data['symbol'] == primary_symbol]
            current_close = primary_data['close'].iloc[-1]
            current_atr = ta.atr(primary_data['high'], primary_data['low'], primary_data['close'], atr_lookback).iloc[-1]
            current_signal = signal_values[-1]
            current_diff = intermarket_diff.iloc[-1]
            
            # Initialize signal
            signal = {
                'symbol': primary_symbol,
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'primary_cmma': primary_cmma.iloc[-1],
                    'reference_cmma': reference_cmma.iloc[-1],
                    'intermarket_diff': current_diff,
                    'atr': current_atr,
                    'reason': 'No signal'
                }
            }
            
            # Check for buy signal
            if current_signal == 1:
                # Calculate signal strength based on difference value
                signal_strength = min(1.0, abs(current_diff / threshold))
                
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
                            'reason': f'{primary_symbol} is undervalued compared to {reference_symbol}',
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })
            
            # Check for sell signal
            elif current_signal == -1:
                # Calculate signal strength based on difference value
                signal_strength = min(1.0, abs(current_diff / threshold))
                
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
                            'reason': f'{primary_symbol} is overvalued compared to {reference_symbol}',
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
