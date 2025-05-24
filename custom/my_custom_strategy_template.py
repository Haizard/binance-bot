"""
🌙 Moon Dev's Custom Strategy Template
Use this template to create your own trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import pandas_ta as ta  # Optional, for technical indicators
from termcolor import cprint

import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.strategies.python_base_strategy import PythonBaseStrategy

class MyCustomStrategy(PythonBaseStrategy):
    """My Custom Trading Strategy
    
    Replace this with a description of your strategy.
    Explain the main idea, what indicators it uses, and when it generates signals.
    """
    
    def __init__(self, name="My Custom Strategy"):
        """Initialize the strategy
        
        Args:
            name: Name of the strategy
        """
        super().__init__(name)
        
        # Set default parameters - CUSTOMIZE THESE FOR YOUR STRATEGY
        self.set_parameters({
            # Technical parameters
            'param1': 20,           # Example: period for indicator 1
            'param2': 50,           # Example: period for indicator 2
            
            # Risk management parameters
            'risk_per_trade': 0.02,  # 2% risk per trade
            'stop_loss_atr': 2.0,    # ATR multiplier for stop loss
            'take_profit_ratio': 2.0, # Risk:Reward ratio
            
            # Other settings
            'verbose': True          # Print detailed logs
        })
        
        # Set description
        self.set_description(
            "Replace this with a detailed description of your strategy. "
            "Explain how it works, what indicators it uses, and when it "
            "generates buy and sell signals."
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
        min_bars = max(self.parameters['param1'], self.parameters['param2']) + 10
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
        
        # YOUR STRATEGY LOGIC GOES HERE
        # This is where you implement your trading rules
        # Example:
        #   - Check for indicator crossovers
        #   - Look for price patterns
        #   - Analyze volume or volatility
        #   - Combine multiple conditions
        
        # For this template, we'll use a simple example:
        # Buy when indicator1 crosses above indicator2
        # Sell when indicator1 crosses below indicator2
        
        indicator1_current = indicators['indicator1'].iloc[-1]
        indicator2_current = indicators['indicator2'].iloc[-1]
        indicator1_previous = indicators['indicator1'].iloc[-2]
        indicator2_previous = indicators['indicator2'].iloc[-2]
        
        # Default signal (neutral)
        signal = {
            'symbol': data.get('symbol', 'Unknown'),
            'direction': 'NEUTRAL',
            'signal_strength': 0,
            'entry_price': current_close,
            'stop_loss': 0,
            'take_profit': 0,
            'metadata': {
                'indicators': indicators.iloc[-1].to_dict()
            }
        }
        
        # Check for buy signal
        if (indicator1_current > indicator2_current and 
            indicator1_previous <= indicator2_previous):
            
            # Calculate stop loss and take profit
            atr = indicators['atr'].iloc[-1]
            stop_loss = current_close - (atr * self.parameters['stop_loss_atr'])
            risk = current_close - stop_loss
            take_profit = current_close + (risk * self.parameters['take_profit_ratio'])
            
            # Update signal
            signal.update({
                'direction': 'BUY',
                'signal_strength': 0.8,  # Confidence level (0-1)
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'metadata': {
                    **signal['metadata'],
                    'reason': 'Indicator1 crossed above Indicator2',
                    'risk_reward_ratio': self.parameters['take_profit_ratio']
                }
            })
            
            if self.parameters['verbose']:
                self.log(f"BUY signal generated at {current_close:.5f}", "green")
                
        # Check for sell signal
        elif (indicator1_current < indicator2_current and 
              indicator1_previous >= indicator2_previous):
            
            # Calculate stop loss and take profit
            atr = indicators['atr'].iloc[-1]
            stop_loss = current_close + (atr * self.parameters['stop_loss_atr'])
            risk = stop_loss - current_close
            take_profit = current_close - (risk * self.parameters['take_profit_ratio'])
            
            # Update signal
            signal.update({
                'direction': 'SELL',
                'signal_strength': 0.8,  # Confidence level (0-1)
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'metadata': {
                    **signal['metadata'],
                    'reason': 'Indicator1 crossed below Indicator2',
                    'risk_reward_ratio': self.parameters['take_profit_ratio']
                }
            })
            
            if self.parameters['verbose']:
                self.log(f"SELL signal generated at {current_close:.5f}", "red")
                
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
        param1 = self.parameters['param1']
        param2 = self.parameters['param2']
        
        # CALCULATE YOUR INDICATORS HERE
        # Examples:
        
        # Simple Moving Averages
        df['indicator1'] = ta.sma(df['close'], length=param1)
        df['indicator2'] = ta.sma(df['close'], length=param2)
        
        # ATR for stop loss calculation
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Additional indicators
        # df['rsi'] = ta.rsi(df['close'], length=14)
        # df['macd'], df['macd_signal'], df['macd_hist'] = ta.macd(df['close'])
        # df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.bbands(df['close'])
        
        return df
