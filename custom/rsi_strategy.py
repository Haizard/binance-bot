"""
🌙 Moon Dev's RSI Strategy
A mean reversion strategy based on RSI (Relative Strength Index)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import pandas_ta as ta
from termcolor import cprint

import sys
import os
# No project root path addition needed for this environment

# Define a base class directly instead of importing from src
class PythonBaseStrategy:
    def __init__(self, name):
        self.name = name
        self.parameters = {}

    def set_parameters(self, params):
        self.parameters = params

    def set_description(self, desc):
        self.description = desc

    def add_timeframe(self, tf):
        pass

    def add_symbol(self, symbol):
        pass

    def log(self, message, color):
        print(f"{self.name}: {message}")

class RSIStrategy(PythonBaseStrategy):
    """RSI Mean Reversion Strategy
    
    This strategy generates buy signals when RSI drops below the oversold threshold
    and sell signals when RSI rises above the overbought threshold.
    
    It uses ATR for stop loss calculation and implements a fixed risk-reward ratio
    for take profit levels.
    """
    
    def __init__(self, name="RSI Strategy"):
        """Initialize the strategy
        
        Args:
            name: Name of the strategy
        """
        super().__init__(name)
        
        # Set default parameters
        self.set_parameters({
            # RSI parameters
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            
            # Risk management parameters
            'risk_per_trade': 0.02,  # 2% risk per trade
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            
            # Other settings
            'verbose': True
        })
        
        # Set description
        self.set_description(
            "RSI Mean Reversion strategy that generates buy signals when RSI drops "
            "below the oversold threshold and sell signals when RSI rises above the "
            "overbought threshold. Uses ATR for stop loss calculation."
        )
        
        # Set supported timeframes
        for tf in ['15m', '1H', '4H', '1D']:
            self.add_timeframe(tf)
            
        # Set supported symbols
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'ETHUSD']:
            self.add_symbol(symbol)
            
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on the provided data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signal information
        """
        # Check if we have enough data
        min_bars = max(self.parameters['rsi_period'], self.parameters['atr_period']) + 10
        if len(data) < min_bars:
            return {
                'symbol': data.get('symbol', 'Unknown'),
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'metadata': {
                    'reason': 'Not enough data for analysis'
                }
            }
            
        # Calculate indicators
        indicators = self._calculate_indicators(data)
        
        # Get the latest values
        current_close = data['close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        previous_rsi = indicators['rsi'].iloc[-2]
        current_atr = indicators['atr'].iloc[-1]
        
        # Default signal (neutral)
        signal = {
            'symbol': data.get('symbol', 'Unknown'),
            'direction': 'NEUTRAL',
            'signal_strength': 0,
            'entry_price': current_close,
            'stop_loss': 0,
            'take_profit': 0,
            'metadata': {
                'rsi': current_rsi,
                'atr': current_atr,
                'indicators': indicators.iloc[-1].to_dict()
            }
        }
        
        # Buy signal: RSI crosses below oversold threshold
        if current_rsi < self.parameters['oversold_threshold'] and previous_rsi >= self.parameters['oversold_threshold']:
            # Calculate stop loss and take profit
            stop_loss = current_close - (current_atr * self.parameters['atr_multiplier'])
            risk = current_close - stop_loss
            take_profit = current_close + (risk * self.parameters['take_profit_ratio'])
            
            # Update signal
            signal.update({
                'direction': 'BUY',
                'signal_strength': 0.8,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'metadata': {
                    **signal['metadata'],
                    'reason': 'RSI crossed below oversold threshold',
                    'risk_reward_ratio': self.parameters['take_profit_ratio']
                }
            })
            
            if self.parameters['verbose']:
                self.log(f"BUY signal generated at {current_close:.5f} (RSI: {current_rsi:.2f})", "green")
                
        # Sell signal: RSI crosses above overbought threshold
        elif current_rsi > self.parameters['overbought_threshold'] and previous_rsi <= self.parameters['overbought_threshold']:
            # Calculate stop loss and take profit
            stop_loss = current_close + (current_atr * self.parameters['atr_multiplier'])
            risk = stop_loss - current_close
            take_profit = current_close - (risk * self.parameters['take_profit_ratio'])
            
            # Update signal
            signal.update({
                'direction': 'SELL',
                'signal_strength': 0.8,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'metadata': {
                    **signal['metadata'],
                    'reason': 'RSI crossed above overbought threshold',
                    'risk_reward_ratio': self.parameters['take_profit_ratio']
                }
            })
            
            if self.parameters['verbose']:
                self.log(f"SELL signal generated at {current_close:.5f} (RSI: {current_rsi:.2f})", "red")
                
        return signal
        
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for the strategy
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        # Create a copy of the data
        df = data.copy()
        
        # Get parameters
        rsi_period = self.parameters['rsi_period']
        atr_period = self.parameters['atr_period']
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['close'], length=rsi_period)
        
        # Calculate ATR for stop loss
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        
        # Calculate additional indicators for analysis
        df['rsi_ma'] = ta.sma(df['rsi'], length=5)  # Smoothed RSI
        df['rsi_trend'] = df['rsi'] - df['rsi'].shift(5)  # RSI trend
        
        return df
