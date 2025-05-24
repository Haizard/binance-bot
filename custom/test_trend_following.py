"""
Test script for the Trend Following Strategy
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

def test_trend_following_strategy():
    """Test the Trend Following Strategy"""
    print("\n=== Testing Trend Following Strategy ===")
    
    # Create strategy instance
    try:
        strategy = StrategyFactory.create_strategy("trend_following")
        print(f"✅ Successfully created strategy: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
    except Exception as e:
        print(f"❌ Error creating strategy: {str(e)}")
        return
    
    # Load test data
    try:
        data = pd.read_csv('BTCUSDT3600.csv')
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        print(f"✅ Successfully loaded data with {len(data)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print("Please make sure you have a BTCUSDT3600.csv file in the current directory")
        return
    
    # Generate signals
    try:
        # Use a window of data
        window = data.iloc[-200:]
        
        print(f"Generating signals for window from {window.index[0]} to {window.index[-1]}")
        print(f"Current price: {window['close'].iloc[-1]:.2f}")
        
        # Generate signal
        signal = strategy.generate_signals(window)
        print(f"✅ Successfully generated signal: {signal['direction']}")
        print(f"Signal strength: {signal.get('signal_strength', 0)}")
        print(f"Reason: {signal.get('metadata', {}).get('reason', 'No reason provided')}")
        
        # Print Moving Averages and MACD values
        fast_ma = signal.get('metadata', {}).get('fast_ma', 'N/A')
        slow_ma = signal.get('metadata', {}).get('slow_ma', 'N/A')
        macd_line = signal.get('metadata', {}).get('macd_line', 'N/A')
        macd_signal = signal.get('metadata', {}).get('macd_signal', 'N/A')
        macd_histogram = signal.get('metadata', {}).get('macd_histogram', 'N/A')
        ma_crossover = signal.get('metadata', {}).get('ma_crossover', 'N/A')
        macd_crossover = signal.get('metadata', {}).get('macd_crossover', 'N/A')
        
        print(f"\nIndicator values:")
        print(f"  Fast MA: {fast_ma}")
        print(f"  Slow MA: {slow_ma}")
        print(f"  MACD Line: {macd_line}")
        print(f"  MACD Signal: {macd_signal}")
        print(f"  MACD Histogram: {macd_histogram}")
        print(f"  MA Crossover: {ma_crossover}")
        print(f"  MACD Crossover: {macd_crossover}")
        
        # Print entry, stop loss, and take profit if available
        if signal['direction'] != 'NEUTRAL':
            print(f"\nEntry price: {signal.get('entry_price', 'N/A')}")
            print(f"Stop loss: {signal.get('stop_loss', 'N/A')}")
            print(f"Take profit: {signal.get('take_profit', 'N/A')}")
            
        # Print full signal for debugging
        print("\nFull signal details:")
        import json
        print(json.dumps(signal, indent=2, default=str))
    except Exception as e:
        print(f"❌ Error generating signals: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Visualize strategy
    try:
        visualize_trend_following(window, signal)
    except Exception as e:
        print(f"❌ Error visualizing strategy: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_trend_following(data, signal):
    """Visualize trend following strategy and signals"""
    print("\n=== Visualizing Trend Following Strategy ===")
    
    # Calculate indicators
    fast_ma_period = 20
    slow_ma_period = 50
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Calculate Moving Averages
    data['fast_ma'] = data['close'].ewm(span=fast_ma_period).mean()
    data['slow_ma'] = data['close'].ewm(span=slow_ma_period).mean()
    
    # Calculate MACD
    data['macd_line'] = data['close'].ewm(span=macd_fast).mean() - data['close'].ewm(span=macd_slow).mean()
    data['macd_signal'] = data['macd_line'].ewm(span=macd_signal).mean()
    data['macd_histogram'] = data['macd_line'] - data['macd_signal']
    
    # Calculate MA crossover
    data['ma_crossover'] = 0
    data.loc[(data['fast_ma'] > data['slow_ma']) & (data['fast_ma'].shift(1) <= data['slow_ma'].shift(1)), 'ma_crossover'] = 1
    data.loc[(data['fast_ma'] < data['slow_ma']) & (data['fast_ma'].shift(1) >= data['slow_ma'].shift(1)), 'ma_crossover'] = -1
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(data.index, data['fast_ma'], 'r--', label=f'Fast MA ({fast_ma_period})')
    plt.plot(data.index, data['slow_ma'], 'g--', label=f'Slow MA ({slow_ma_period})')
    plt.title('Trend Following Strategy - Moving Averages')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # Mark entry, stop loss, and take profit if available
    if signal['direction'] != 'NEUTRAL':
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        if entry_price:
            plt.axhline(y=entry_price, color='g' if signal['direction'] == 'BUY' else 'r', 
                       linestyle='-', label=f"Entry ({entry_price:.2f})")
        
        if stop_loss:
            plt.axhline(y=stop_loss, color='r', linestyle='--', 
                       label=f"Stop Loss ({stop_loss:.2f})")
        
        if take_profit:
            plt.axhline(y=take_profit, color='g', linestyle='--', 
                       label=f"Take Profit ({take_profit:.2f})")
    
    # Plot MACD
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['macd_line'], label='MACD Line')
    plt.plot(data.index, data['macd_signal'], 'r--', label='Signal Line')
    plt.bar(data.index, data['macd_histogram'], color=['g' if x > 0 else 'r' for x in data['macd_histogram']], label='Histogram')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('MACD')
    plt.ylabel('MACD')
    plt.grid(True)
    plt.legend()
    
    # Plot MA Crossover
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['ma_crossover'], label='MA Crossover')
    plt.fill_between(data.index, 0, 1, where=data['ma_crossover'] == 1, color='green', alpha=0.3, label='Bullish Crossover')
    plt.fill_between(data.index, -1, 0, where=data['ma_crossover'] == -1, color='red', alpha=0.3, label='Bearish Crossover')
    plt.title('Moving Average Crossover')
    plt.ylabel('Crossover')
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()
    
    # Add signal information as text
    signal_text = f"Signal: {signal['direction']} (Strength: {signal.get('signal_strength', 0):.2f})"
    reason = signal.get('metadata', {}).get('reason', 'No reason provided')
    plt.figtext(0.5, 0.01, f"{signal_text}\nReason: {reason}", ha='center', fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('trend_following_strategy.png')
    print("✅ Saved visualization to trend_following_strategy.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    test_trend_following_strategy()
