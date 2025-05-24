"""
Perceptually Important Points (PIP) Strategy

This strategy identifies significant pivot points in price data using various distance measures
and generates trading signals based on pattern recognition. It reduces a price series to its
most significant points while preserving the overall shape, which can be used as a foundation
for pattern recognition strategies.

The strategy supports three distance measures:
1. Euclidean Distance
2. Perpendicular Distance
3. Vertical Distance
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Any, Optional, Tuple
import logging

class PIPStrategy:
    """
    Perceptually Important Points (PIP) Strategy
    
    This strategy identifies significant pivot points in price data using various distance measures
    and generates trading signals based on pattern recognition.
    
    Parameters:
    -----------
    n_pips : int
        Number of perceptually important points to identify (default: 5)
    lookback : int
        Lookback period for pattern identification (default: 30)
    dist_measure : int
        Distance measure for PIP calculation (default: 3)
        1 = Euclidean Distance
        2 = Perpendicular Distance
        3 = Vertical Distance
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
    
    def __init__(self, name: str = "Perceptually Important Points Strategy"):
        """Initialize the strategy
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        
        # Set default parameters
        self.parameters = {
            'n_pips': 5,  # Number of perceptually important points to identify
            'lookback': 30,  # Lookback period for pattern identification
            'dist_measure': 3,  # Distance measure for PIP calculation (3 = Vertical Distance)
            'use_log_prices': True,  # Use log prices for scaling
            'risk_per_trade': 0.02,  # 2% risk per trade
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            'pattern_library': {  # Pre-defined patterns to match against
                'double_top': [0, 1, 0, 1, 0],  # Normalized pattern shape
                'double_bottom': [0, -1, 0, -1, 0],
                'head_shoulders': [0, 1, 2, 1, 0],
                'inv_head_shoulders': [0, -1, -2, -1, 0],
                'ascending_triangle': [0, 1, 0, 1, 1],
                'descending_triangle': [0, -1, 0, -1, -1],
                'rising_wedge': [0, 1, 0.5, 1.5, 1],
                'falling_wedge': [0, -1, -0.5, -1.5, -1]
            },
            'pattern_thresholds': {  # Correlation thresholds for pattern matching
                'double_top': 0.85,
                'double_bottom': 0.85,
                'head_shoulders': 0.8,
                'inv_head_shoulders': 0.8,
                'ascending_triangle': 0.75,
                'descending_triangle': 0.75,
                'rising_wedge': 0.75,
                'falling_wedge': 0.75
            },
            'pattern_signals': {  # Signal direction for each pattern
                'double_top': 'SELL',
                'double_bottom': 'BUY',
                'head_shoulders': 'SELL',
                'inv_head_shoulders': 'BUY',
                'ascending_triangle': 'BUY',
                'descending_triangle': 'SELL',
                'rising_wedge': 'SELL',
                'falling_wedge': 'BUY'
            },
            'verbose': True  # Print detailed logs
        }
        
        # Set description
        self.description = (
            "Perceptually Important Points (PIP) strategy that identifies significant pivot points "
            "in price data using various distance measures and generates trading signals based on "
            "pattern recognition. It reduces a price series to its most significant points while "
            "preserving the overall shape, which can be used as a foundation for pattern recognition "
            "strategies."
        )
    
    def find_pips(self, data: np.ndarray, n_pips: int, dist_measure: int) -> Tuple[List[int], List[float]]:
        """Find perceptually important points in price data
        
        Args:
            data: Array of price data
            n_pips: Number of perceptually important points to identify
            dist_measure: Distance measure for PIP calculation
                1 = Euclidean Distance
                2 = Perpendicular Distance
                3 = Vertical Distance
            
        Returns:
            Tuple of (pip_indices, pip_values)
        """
        # Initialize with first and last points
        pips_x = [0, len(data) - 1]  # Index
        pips_y = [data[0], data[-1]]  # Price

        # Iteratively add points with maximum distance
        for curr_point in range(2, n_pips):
            md = 0.0  # Max distance
            md_i = -1  # Max distance index
            insert_index = -1

            # Check each segment between adjacent PIPs
            for k in range(0, curr_point - 1):
                # Left adjacent, right adjacent indices
                left_adj = k
                right_adj = k + 1

                # Calculate line parameters between adjacent PIPs
                time_diff = pips_x[right_adj] - pips_x[left_adj]
                price_diff = pips_y[right_adj] - pips_y[left_adj]
                slope = price_diff / time_diff
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope

                # Check each point between adjacent PIPs
                for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                    d = 0.0  # Distance
                    if dist_measure == 1:  # Euclidean distance
                        d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                        d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                    elif dist_measure == 2:  # Perpendicular distance
                        d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                    else:  # Vertical distance
                        d = abs((slope * i + intercept) - data[i])

                    # Update max distance if current distance is greater
                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj

            # Insert the point with maximum distance
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])

        return pips_x, pips_y
    
    def normalize_pattern(self, pattern: List[float]) -> List[float]:
        """Normalize a pattern to have zero mean and unit standard deviation
        
        Args:
            pattern: List of pattern values
            
        Returns:
            Normalized pattern
        """
        pattern_array = np.array(pattern)
        mean = np.mean(pattern_array)
        std = np.std(pattern_array)
        
        if std == 0:
            return [0] * len(pattern)
        
        return list((pattern_array - mean) / std)
    
    def match_pattern(self, pattern: List[float], library: Dict[str, List[float]], thresholds: Dict[str, float]) -> Tuple[str, float]:
        """Match a pattern against a library of patterns
        
        Args:
            pattern: Normalized pattern to match
            library: Dictionary of pattern names to normalized patterns
            thresholds: Dictionary of pattern names to correlation thresholds
            
        Returns:
            Tuple of (pattern_name, correlation)
        """
        best_match = None
        best_corr = 0
        
        for name, template in library.items():
            # Ensure same length
            if len(pattern) != len(template):
                continue
            
            # Calculate correlation
            corr = np.corrcoef(pattern, template)[0, 1]
            
            # Check if correlation exceeds threshold and is better than previous matches
            if not np.isnan(corr) and corr > thresholds.get(name, 0.7) and corr > best_corr:
                best_match = name
                best_corr = corr
        
        return best_match, best_corr
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on perceptually important points
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Ensure we have enough data
            if len(data) < self.parameters['lookback']:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'Not enough data for PIP analysis'
                    }
                }
            
            # Get parameters
            n_pips = self.parameters['n_pips']
            lookback = self.parameters['lookback']
            dist_measure = self.parameters['dist_measure']
            use_log_prices = self.parameters['use_log_prices']
            
            # Extract close prices
            close_prices = data['close'].to_numpy()
            
            # Apply log transform if enabled
            if use_log_prices:
                close_prices = np.log(close_prices)
            
            # Get the last window of data for pattern detection
            last_window = close_prices[-lookback:]
            
            # Find PIPs in the last window
            pips_x, pips_y = self.find_pips(last_window, n_pips, dist_measure)
            
            # Normalize the pattern
            norm_pattern = self.normalize_pattern(pips_y)
            
            # Match against pattern library
            pattern_name, correlation = self.match_pattern(
                norm_pattern, 
                self.parameters['pattern_library'],
                self.parameters['pattern_thresholds']
            )
            
            # Calculate ATR for stop loss
            atr = ta.atr(
                data['high'], data['low'], data['close'], 
                self.parameters['atr_period']
            ).iloc[-1]
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # If a pattern is matched, generate a signal
            if pattern_name:
                direction = self.parameters['pattern_signals'].get(pattern_name, 'NEUTRAL')
                
                if direction == 'BUY':
                    # Calculate stop loss and take profit
                    stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * self.parameters['take_profit_ratio'])
                    
                    return {
                        'direction': 'BUY',
                        'signal_strength': correlation,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            'reason': f'Matched {pattern_name} pattern with correlation {correlation:.2f}',
                            'pattern_name': pattern_name,
                            'correlation': correlation,
                            'pip_points': list(zip(pips_x, pips_y))
                        }
                    }
                elif direction == 'SELL':
                    # Calculate stop loss and take profit
                    stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * self.parameters['take_profit_ratio'])
                    
                    return {
                        'direction': 'SELL',
                        'signal_strength': correlation,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            'reason': f'Matched {pattern_name} pattern with correlation {correlation:.2f}',
                            'pattern_name': pattern_name,
                            'correlation': correlation,
                            'pip_points': list(zip(pips_x, pips_y))
                        }
                    }
            
            # If no pattern is matched, return neutral signal
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': 'No significant pattern detected',
                    'pip_points': list(zip(pips_x, pips_y))
                }
            }
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            if self.parameters['verbose']:
                logging.error(f"Error in PIP signal generation: {str(e)}")
                logging.error(error_details)
            
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': f'Error: {str(e)}',
                    'error_details': error_details
                }
            }
