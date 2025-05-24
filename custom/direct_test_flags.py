"""
Direct test for the Flags and Pennants Strategy
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.strategies.custom.flags_pennants import FlagPattern
from src.strategies.custom.strategy_factory import StrategyFactory

def create_manual_signal():
    """Create a manual signal to test the strategy adapter"""
    print("Creating a manual signal to test the Flags and Pennants Strategy...")
    
    # Create strategy instance
    strategy = StrategyFactory.create_strategy("flags_pennants")
    
    # Print strategy info
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    
    # Create a simple dataset
    n_days = 50
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a simple price series
    price = np.ones(len(date_range)) * 100
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': price - 0.1,
        'high': price + 0.2,
        'low': price - 0.2,
        'close': price,
        'volume': np.random.uniform(1000, 5000, len(date_range))
    }, index=date_range)
    
    # Add symbol column
    data['symbol'] = 'BTCUSD'
    
    # Get the current price
    current_price = data['close'].iloc[-1]
    
    # Calculate ATR for stop loss
    atr = 2.0  # Simplified ATR calculation
    
    # Create a manual flag pattern
    flag = FlagPattern(
        base_x=20,  # Start of the trend
        base_y=100.0,  # Start price
        tip_x=30,  # Top of pole
        tip_y=130.0,  # Top price
        conf_x=49,  # Confirmation at last candle
        conf_y=130.0,  # Confirmation price
        pennant=False,  # It's a flag, not a pennant
        flag_width=10,
        flag_height=5.0,
        pole_width=10,
        pole_height=30.0,
        support_slope=-0.2,
        support_intercept=130.0,
        resist_slope=-0.1,
        resist_intercept=135.0
    )
    
    # Create a manual signal
    signal = {
        'direction': 'BUY',
        'signal_strength': 0.8,
        'entry_price': current_price,
        'stop_loss': current_price - (atr * strategy.parameters['atr_multiplier']),
        'take_profit': current_price + (atr * strategy.parameters['atr_multiplier'] * strategy.parameters['take_profit_ratio']),
        'metadata': {
            'reason': 'Confirmed bull_flag pattern',
            'pattern_metrics': {
                'pattern_type': 'bull_flag',
                'pole_height': flag.pole_height,
                'pole_width': flag.pole_width,
                'flag_height': flag.flag_height,
                'flag_width': flag.flag_width,
                'support_slope': flag.support_slope,
                'resist_slope': flag.resist_slope
            }
        }
    }
    
    # Print the signal
    print("\nManually created signal:")
    print(f"Direction: {signal['direction']}")
    print(f"Signal Strength: {signal['signal_strength']}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"Metadata: {signal['metadata']}")
    
    # Plot the data with the pattern
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'])
    
    # Mark the pattern points
    plt.plot(data.index[flag.base_x], flag.base_y, 'go', markersize=10, label='Base')
    plt.plot(data.index[flag.tip_x], flag.tip_y, 'ro', markersize=10, label='Tip')
    plt.plot(data.index[flag.conf_x], flag.conf_y, 'bo', markersize=10, label='Confirmation')
    
    # Draw the pole
    plt.plot([data.index[flag.base_x], data.index[flag.tip_x]], 
             [flag.base_y, flag.tip_y], 'g-', linewidth=2, label='Pole')
    
    # Draw the flag
    x_flag = np.arange(flag.tip_x, flag.conf_x + 1)
    y_support = flag.support_intercept + flag.support_slope * (x_flag - flag.tip_x)
    y_resist = flag.resist_intercept + flag.resist_slope * (x_flag - flag.tip_x)
    
    plt.plot(data.index[x_flag], y_support, 'b-', linewidth=2, label='Support')
    plt.plot(data.index[x_flag], y_resist, 'r-', linewidth=2, label='Resistance')
    
    plt.title('Manual Bull Flag Pattern')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('manual_flag_pattern.png')
    print("\nManual flag pattern plot saved to 'manual_flag_pattern.png'")
    
    return strategy, data, signal

if __name__ == "__main__":
    # Create a manual signal
    strategy, data, signal = create_manual_signal()
