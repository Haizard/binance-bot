"""
🌙 Moon Dev's SMA RSI Strategy
Built with love by Moon Dev 🚀

This strategy combines SMA crossover with RSI confirmation for trading signals.
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Tuple
from ..base_strategy import BaseStrategy

class SmaRsiStrategy(BaseStrategy):
    """
    A trading strategy that uses SMA crossover with RSI confirmation
    
    Parameters:
    - fast_sma: Fast SMA period (default: 20)
    - slow_sma: Slow SMA period (default: 50)
    - rsi_period: RSI period (default: 14)
    - rsi_overbought: RSI overbought level (default: 70)
    - rsi_oversold: RSI oversold level (default: 30)
    """
    
    def __init__(self, 
                 fast_sma: int = 20,
                 slow_sma: int = 50,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 **kwargs):
        super().__init__(**kwargs)
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy indicators"""
        # Calculate SMAs
        df['fast_sma'] = ta.sma(df['close'], length=self.fast_sma)
        df['slow_sma'] = ta.sma(df['close'], length=self.slow_sma)
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        return df
        
    def generate_signals(self, df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Generate trading signals based on indicator values"""
        if len(df) < self.slow_sma:
            return "NONE", {}
            
        # Get latest values
        current_fast_sma = df['fast_sma'].iloc[-1]
        current_slow_sma = df['slow_sma'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Previous values
        prev_fast_sma = df['fast_sma'].iloc[-2]
        prev_slow_sma = df['slow_sma'].iloc[-2]
        
        # Check for crossovers
        golden_cross = prev_fast_sma <= prev_slow_sma and current_fast_sma > current_slow_sma
        death_cross = prev_fast_sma >= prev_slow_sma and current_fast_sma < current_slow_sma
        
        # Generate signals with RSI confirmation
        if golden_cross and current_rsi < self.rsi_overbought:
            stop_loss = current_price * 0.99  # 1% stop loss
            take_profit = current_price * 1.02  # 2% take profit
            return "BUY", {
                "entry": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": f"Golden cross (Fast SMA {self.fast_sma} > Slow SMA {self.slow_sma}) with RSI {current_rsi:.2f} below overbought"
            }
            
        elif death_cross and current_rsi > self.rsi_oversold:
            stop_loss = current_price * 1.01  # 1% stop loss
            take_profit = current_price * 0.98  # 2% take profit
            return "SELL", {
                "entry": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": f"Death cross (Fast SMA {self.fast_sma} < Slow SMA {self.slow_sma}) with RSI {current_rsi:.2f} above oversold"
            }
            
        return "NONE", {} 