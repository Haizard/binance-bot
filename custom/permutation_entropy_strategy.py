"""
Permutation Entropy Strategy

This strategy calculates permutation entropy to measure time series complexity
and identify regime changes in the market. It generates trading signals based on
changes in entropy values, providing insights into market predictability.
"""

import pandas as pd
import numpy as np
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
import pandas_ta as ta

# Setup logging
logger = logging.getLogger("Permutation Entropy Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class PermutationEntropyStrategy:
    """Permutation Entropy Strategy

    This strategy calculates permutation entropy to measure time series complexity
    and identify regime changes in the market.
    """

    def __init__(self, name: str = "Permutation Entropy Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "Permutation Entropy Strategy that calculates permutation entropy to measure "
            "time series complexity and identify regime changes in the market. It generates "
            "trading signals based on changes in entropy values, providing insights into "
            "market predictability."
        )

        # Default parameters
        self.parameters = {
            'dimension': 3,  # Dimension of the ordinal patterns (d)
            'mult': 28,  # Multiplier for lookback window (lookback = factorial(d) * mult)
            'entropy_ma_period': 10,  # Moving average period for smoothing entropy
            'high_entropy_threshold': 0.8,  # Threshold for high entropy (unpredictable market)
            'low_entropy_threshold': 0.4,  # Threshold for low entropy (predictable market)
            'entropy_change_threshold': 0.1,  # Threshold for significant entropy change
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # ATR period for stop loss calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            'price_column': 'close',  # Column to use for entropy calculation
        }

        # Initialize cache for indicators
        self.indicator_cache = {}

    def _ordinal_patterns(self, arr: np.ndarray, d: int) -> np.ndarray:
        """Calculate ordinal patterns from time series data

        Args:
            arr: Input array
            d: Dimension of the ordinal patterns

        Returns:
            Array with ordinal patterns
        """
        try:
            assert(d >= 2)
            fac = math.factorial(d)
            d1 = d - 1
            mults = []
            for i in range(1, d):
                mult = fac / math.factorial(i + 1)
                mults.append(mult)
           
            # Create array to put ordinal pattern in
            ordinals = np.empty(len(arr))
            ordinals[:] = np.nan

            for i in range(d1, len(arr)):
                dat = arr[i - d1:  i+1] 
                pattern_ordinal = 0
                for l in range(1, d): 
                    count = 0
                    for r in range(l):
                        if dat[d1 - l] >= dat[d1 - r]:
                           count += 1
                     
                    pattern_ordinal += count * mults[l - 1]
                ordinals[i] = int(pattern_ordinal)
            
            return ordinals
        except Exception as e:
            logger.error(f"Error calculating ordinal patterns: {e}")
            return np.full(len(arr), np.nan)

    def _permutation_entropy(self, arr: np.ndarray, d: int, mult: int) -> np.ndarray:
        """Calculate permutation entropy from time series data

        Args:
            arr: Input array
            d: Dimension of the ordinal patterns
            mult: Multiplier for lookback window

        Returns:
            Array with permutation entropy values
        """
        try:
            fac = math.factorial(d)
            lookback = fac * mult
            
            ent = np.empty(len(arr))
            ent[:] = np.nan
            ordinals = self._ordinal_patterns(arr, d)
            
            for i in range(lookback + d - 1, len(arr)):
                window = ordinals[i - lookback + 1:i+1]
                
                # Create distribution
                freqs = pd.Series(window).value_counts().to_dict()
                for j in range(fac):
                    if j in freqs:
                        freqs[j] = freqs[j] / lookback
               
                # Calculate entropy
                perm_entropy = 0.0
                for k, v in freqs.items():
                    perm_entropy += v * math.log2(v)

                # Normalize to 0-1
                perm_entropy = -1. * (1. / math.log2(fac)) * perm_entropy
                ent[i] = perm_entropy
                
            return ent
        except Exception as e:
            logger.error(f"Error calculating permutation entropy: {e}")
            return np.full(len(arr), np.nan)

    def _calculate_entropy_change(self, entropy: np.ndarray, period: int) -> np.ndarray:
        """Calculate change in entropy over a period

        Args:
            entropy: Array with entropy values
            period: Period for change calculation

        Returns:
            Array with entropy change values
        """
        try:
            entropy_change = np.zeros_like(entropy)
            entropy_change[:] = np.nan
            
            for i in range(period, len(entropy)):
                entropy_change[i] = entropy[i] - entropy[i - period]
                
            return entropy_change
        except Exception as e:
            logger.error(f"Error calculating entropy change: {e}")
            return np.full(len(entropy), np.nan)

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on permutation entropy

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['dimension'] * 10:
            logger.warning("Insufficient data for permutation entropy analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for permutation entropy analysis'
                }
            }

        try:
            # Ensure data has required columns
            price_column = self.parameters['price_column']
            if price_column not in data.columns:
                logger.error(f"Missing required column: {price_column}")
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'metadata': {
                        'reason': f"Missing required column: {price_column}"
                    }
                }

            # Get parameters
            dimension = self.parameters['dimension']
            mult = self.parameters['mult']
            entropy_ma_period = self.parameters['entropy_ma_period']
            high_entropy_threshold = self.parameters['high_entropy_threshold']
            low_entropy_threshold = self.parameters['low_entropy_threshold']
            entropy_change_threshold = self.parameters['entropy_change_threshold']
            
            # Get symbol from data if available
            symbol = data.get('symbol', ['Unknown'])[0] if 'symbol' in data else 'Unknown'
            
            # Calculate permutation entropy
            prices = data[price_column].values
            entropy = self._permutation_entropy(prices, dimension, mult)
            
            # Calculate smoothed entropy
            entropy_ma = pd.Series(entropy).rolling(entropy_ma_period).mean().values
            
            # Calculate entropy change
            entropy_change = self._calculate_entropy_change(entropy_ma, entropy_ma_period)
            
            # Calculate ATR for stop loss
            atr = ta.atr(data['high'], data['low'], data['close'], self.parameters['atr_period'])
            
            # Get current values
            current_close = data['close'].iloc[-1]
            current_entropy = entropy_ma[-1]
            current_entropy_change = entropy_change[-1]
            current_atr = atr.iloc[-1]
            
            # Initialize signal
            signal = {
                'symbol': symbol,
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'entropy': current_entropy,
                    'entropy_change': current_entropy_change,
                    'atr': current_atr,
                    'reason': 'No significant entropy change'
                }
            }
            
            # Check for buy signal (entropy decreasing from high to low)
            if (current_entropy < high_entropy_threshold and 
                current_entropy > low_entropy_threshold and 
                current_entropy_change < -entropy_change_threshold):
                
                # Calculate signal strength based on entropy change
                signal_strength = min(1.0, abs(current_entropy_change) / (2 * entropy_change_threshold))
                
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
                            'reason': 'Entropy decreasing from high to low (market becoming more predictable)',
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })
            
            # Check for sell signal (entropy increasing from low to high)
            elif (current_entropy > low_entropy_threshold and 
                  current_entropy < high_entropy_threshold and 
                  current_entropy_change > entropy_change_threshold):
                
                # Calculate signal strength based on entropy change
                signal_strength = min(1.0, abs(current_entropy_change) / (2 * entropy_change_threshold))
                
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
                            'reason': 'Entropy increasing from low to high (market becoming less predictable)',
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
