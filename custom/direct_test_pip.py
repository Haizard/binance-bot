"""
Direct test for the Pip Pattern Miner Strategy
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

from src.strategies.custom.strategy_factory import StrategyFactory
from src.strategies.custom.perceptually_important import find_pips

def create_manual_signal():
    """Create a manual signal to test the Pip Pattern Miner Strategy"""
    print("Creating a manual signal to test the Pip Pattern Miner Strategy...")
    
    # Create strategy instance
    strategy = StrategyFactory.create_strategy("pip_pattern_miner")
    
    # Print strategy info
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    
    # Create a simple dataset with a strong uptrend
    n_days = 100
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a price series with a strong uptrend at the end
    price = np.ones(len(date_range)) * 100
    
    # Add a strong uptrend in the last 20 days
    for i in range(20):
        price[-20+i:] = 100 + i * 1.5
    
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
    
    # Force a BUY signal
    # We'll modify the strategy's signal generation function to always return a BUY signal
    original_signal_function = strategy.signal_function
    
    def forced_buy_signal(data):
        # Get the current price
        current_price = data['close'].iloc[-1]
        
        # Calculate ATR for stop loss
        atr = data['high'].rolling(window=strategy.parameters['atr_period']).max() - data['low'].rolling(window=strategy.parameters['atr_period']).min()
        atr = atr.iloc[-1]
        
        # Calculate stop loss and take profit
        stop_loss = current_price - (atr * strategy.parameters['atr_multiplier'])
        risk = current_price - stop_loss
        take_profit = current_price + (risk * strategy.parameters['take_profit_ratio'])
        
        # Extract close prices for PIP calculation
        close_prices = data['close'].to_numpy()
        
        # Find PIPs in the last window
        lookback = min(strategy.parameters['lookback'], len(close_prices))
        last_window = close_prices[-lookback:]
        
        pips_x, pips_y = find_pips(
            last_window, 
            strategy.parameters['n_pips'], 
            strategy.parameters['dist_measure']
        )
        
        # Create a forced BUY signal
        return {
            'direction': 'BUY',
            'signal_strength': 0.8,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'metadata': {
                'reason': 'Forced bullish PIP pattern for testing',
                'prediction_value': 0.8,
                'pip_points': list(zip(pips_x, pips_y))
            }
        }
    
    # Replace the signal function temporarily
    strategy.signal_function = forced_buy_signal
    
    # Generate the signal
    signal = strategy.generate_signals(data)
    
    # Restore the original signal function
    strategy.signal_function = original_signal_function
    
    # Print the signal
    print("\nManually created signal:")
    print(f"Direction: {signal['direction']}")
    print(f"Signal Strength: {signal['signal_strength']}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"Metadata: {signal['metadata']}")
    
    # Plot the data with the PIPs
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'])
    
    # Mark the PIPs
    if 'pip_points' in signal['metadata']:
        # Extract PIP points
        pip_points = signal['metadata']['pip_points']
        
        # Adjust x indices to match the data index
        start_idx = len(data) - strategy.parameters['lookback']
        for i, (x, y) in enumerate(pip_points):
            plt.plot(data.index[start_idx + x], data['close'].iloc[start_idx + x], 'ro', markersize=8)
            plt.text(data.index[start_idx + x], data['close'].iloc[start_idx + x], f'PIP {i+1}', fontsize=10)
    
    plt.title('Price Data with Perceptually Important Points (PIPs)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('manual_pip_pattern.png')
    print("\nManual PIP pattern plot saved to 'manual_pip_pattern.png'")
    
    return strategy, data, signal

if __name__ == "__main__":
    # Create a manual signal
    strategy, data, signal = create_manual_signal()
