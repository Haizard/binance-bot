"""
Volume Spread Analysis (VSA) Strategy

This strategy analyzes the relationship between price range and volume to identify
anomalies in the volume-price relationship. It generates signals based on specific
VSA patterns (e.g., effort vs. result).

The strategy uses the following approach:
1. Normalizes volume and price range using rolling windows
2. Fits a rolling linear model mapping normalized volume to normalized price range
3. Calculates the difference between actual range and predicted range
4. Generates signals based on this difference (VSA indicator)

Key concepts:
- Positive VSA indicator: Price range is larger than expected given the volume (bullish)
- Negative VSA indicator: Price range is smaller than expected given the volume (bearish)
- Zero VSA indicator: Price range is as expected given the volume (neutral)
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple
import logging

class VSAStrategy:
    """
    Volume Spread Analysis (VSA) Strategy
    
    This strategy analyzes the relationship between price range and volume to identify
    anomalies in the volume-price relationship. It generates signals based on specific
    VSA patterns (effort vs. result).
    
    Parameters:
    -----------
    norm_lookback : int
        Lookback period for normalization (default: 168)
    buy_threshold : float
        Threshold for buy signals (default: 0.8)
    sell_threshold : float
        Threshold for sell signals (default: -0.8)
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
    
    def __init__(self, name: str = "VSA Strategy"):
        """Initialize the strategy
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        
        # Set default parameters
        self.parameters = {
            'norm_lookback': 168,  # Lookback period for normalization (1 week of hourly data)
            'buy_threshold': 0.8,  # Threshold for buy signals
            'sell_threshold': -0.8,  # Threshold for sell signals
            'risk_per_trade': 0.02,  # 2% risk per trade
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            'verbose': True  # Print detailed logs
        }
        
        # Set description
        self.description = (
            "Volume Spread Analysis (VSA) strategy that analyzes the relationship between "
            "price range and volume to identify anomalies in the volume-price relationship. "
            "It generates signals based on specific VSA patterns (effort vs. result)."
        )
    
    def vsa_indicator(self, data: pd.DataFrame, norm_lookback: int = 168) -> pd.Series:
        """Calculate the VSA indicator
        
        Args:
            data: DataFrame with OHLCV data
            norm_lookback: Lookback period for normalization
            
        Returns:
            Series with VSA indicator values
        """
        # Ensure we have volume data
        if 'volume' not in data.columns:
            raise ValueError("Volume data is required for VSA indicator")
        
        # Calculate ATR for normalization
        atr = ta.atr(data['high'], data['low'], data['close'], norm_lookback)
        
        # Calculate median volume for normalization
        vol_med = data['volume'].rolling(norm_lookback).median()
        
        # Normalize range and volume
        norm_range = (data['high'] - data['low']) / atr
        norm_volume = data['volume'] / vol_med
        
        # Convert to numpy arrays for faster processing
        norm_vol = norm_volume.to_numpy()
        norm_range_arr = norm_range.to_numpy()
        
        # Initialize range deviation array
        range_dev = np.zeros(len(data))
        range_dev[:] = np.nan
        
        # Calculate range deviation for each point
        for i in range(norm_lookback * 2, len(data)):
            # Get window of data
            window_vol = norm_vol[i - norm_lookback + 1:i + 1]
            window_range = norm_range_arr[i - norm_lookback + 1:i + 1]
            
            # Filter out NaN values
            mask = ~np.isnan(window_vol) & ~np.isnan(window_range)
            if np.sum(mask) < 10:  # Need at least 10 valid points
                continue
                
            # Fit linear regression
            try:
                slope, intercept, r_val, _, _ = stats.linregress(
                    window_vol[mask], window_range[mask]
                )
                
                # Skip if slope is negative or correlation is weak
                if slope <= 0.0 or r_val < 0.2:
                    range_dev[i] = 0.0
                    continue
                
                # Calculate predicted range based on volume
                pred_range = intercept + slope * norm_vol[i]
                
                # Calculate deviation from predicted range
                range_dev[i] = norm_range_arr[i] - pred_range
            except Exception as e:
                if self.parameters['verbose']:
                    logging.warning(f"Error calculating VSA indicator: {str(e)}")
                range_dev[i] = 0.0
        
        return pd.Series(range_dev, index=data.index)
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on the VSA indicator
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Ensure we have enough data
            if len(data) < self.parameters['norm_lookback'] * 2 + 10:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'Not enough data for VSA analysis'
                    }
                }
            
            # Calculate VSA indicator
            vsa_indicator = self.vsa_indicator(
                data, self.parameters['norm_lookback']
            )
            
            # Get the most recent value
            current_vsa = vsa_indicator.iloc[-1]
            
            # Calculate ATR for stop loss
            atr = ta.atr(
                data['high'], data['low'], data['close'], 
                self.parameters['atr_period']
            ).iloc[-1]
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Generate signal based on VSA indicator
            if current_vsa > self.parameters['buy_threshold']:
                # Calculate stop loss and take profit
                stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.parameters['take_profit_ratio'])
                
                # Calculate signal strength (0.5 to 1.0)
                signal_strength = min(1.0, 0.5 + (current_vsa - self.parameters['buy_threshold']) / 2)
                
                return {
                    'direction': 'BUY',
                    'signal_strength': signal_strength,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Price range larger than expected given volume (bullish)',
                        'vsa_indicator': float(current_vsa),
                        'threshold': self.parameters['buy_threshold']
                    }
                }
            elif current_vsa < self.parameters['sell_threshold']:
                # Calculate stop loss and take profit
                stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                risk = stop_loss - current_price
                take_profit = current_price - (risk * self.parameters['take_profit_ratio'])
                
                # Calculate signal strength (0.5 to 1.0)
                signal_strength = min(1.0, 0.5 + abs(current_vsa - self.parameters['sell_threshold']) / 2)
                
                return {
                    'direction': 'SELL',
                    'signal_strength': signal_strength,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Price range smaller than expected given volume (bearish)',
                        'vsa_indicator': float(current_vsa),
                        'threshold': self.parameters['sell_threshold']
                    }
                }
            else:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'VSA indicator within thresholds',
                        'vsa_indicator': float(current_vsa),
                        'buy_threshold': self.parameters['buy_threshold'],
                        'sell_threshold': self.parameters['sell_threshold']
                    }
                }
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            if self.parameters['verbose']:
                logging.error(f"Error in VSA signal generation: {str(e)}")
                logging.error(error_details)
            
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': f'Error: {str(e)}',
                    'error_details': error_details
                }
            }
