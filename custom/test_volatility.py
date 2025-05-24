"""
Test script for the Volatility Strategy
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

def test_volatility_strategy():
    """Test the Volatility Strategy"""
    print("\n=== Testing Volatility Strategy ===")
    
    # Create strategy instance
    try:
        strategy = StrategyFactory.create_strategy("volatility")
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
        
        # Print ATR and Bollinger Bands values
        atr = signal.get('metadata', {}).get('atr', 'N/A')
        atr_change = signal.get('metadata', {}).get('atr_change', 'N/A')
        volatility_expanding = signal.get('metadata', {}).get('volatility_expanding', 'N/A')
        bb_position = signal.get('metadata', {}).get('bb_position', 'N/A')
        upper_band = signal.get('metadata', {}).get('upper_band', 'N/A')
        lower_band = signal.get('metadata', {}).get('lower_band', 'N/A')
        middle_band = signal.get('metadata', {}).get('middle_band', 'N/A')
        
        print(f"\nIndicator values:")
        print(f"  ATR: {atr}")
        print(f"  ATR Change: {atr_change}")
        print(f"  Volatility Expanding: {volatility_expanding}")
        print(f"  BB Position: {bb_position}")
        print(f"  Upper Band: {upper_band}")
        print(f"  Middle Band: {middle_band}")
        print(f"  Lower Band: {lower_band}")
        
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
        visualize_volatility(window, signal)
    except Exception as e:
        print(f"❌ Error visualizing strategy: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_volatility(data, signal):
    """Visualize volatility strategy and signals"""
    print("\n=== Visualizing Volatility Strategy ===")
    
    # Calculate indicators
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_lookback = 5
    
    # Calculate Bollinger Bands
    data['sma'] = data['close'].rolling(window=bb_period).mean()
    data['std'] = data['close'].rolling(window=bb_period).std()
    data['upper_band'] = data['sma'] + (data['std'] * bb_std)
    data['lower_band'] = data['sma'] - (data['std'] * bb_std)
    
    # Calculate ATR
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['atr'] = true_range.rolling(window=atr_period).mean()
    
    # Calculate if ATR is increasing (volatility expansion)
    data['atr_change'] = data['atr'].pct_change(periods=atr_lookback)
    data['volatility_expanding'] = data['atr_change'] > 0.02
    
    # Calculate price position relative to Bollinger Bands
    data['bb_position'] = (data['close'] - data['lower_band']) / (data['upper_band'] - data['lower_band'])
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot price and Bollinger Bands
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(data.index, data['upper_band'], 'r--', label='Upper Band')
    plt.plot(data.index, data['sma'], 'g--', label='SMA')
    plt.plot(data.index, data['lower_band'], 'b--', label='Lower Band')
    plt.title('Volatility Strategy - Bollinger Bands')
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
    
    # Plot ATR
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['atr'], label='ATR')
    plt.title('Average True Range (ATR)')
    plt.ylabel('ATR')
    plt.grid(True)
    plt.legend()
    
    # Plot ATR Change and Volatility Expansion
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['atr_change'], label='ATR Change')
    plt.axhline(y=0.02, color='r', linestyle='--', label='Expansion Threshold')
    plt.fill_between(data.index, 0, 1, where=data['volatility_expanding'], color='green', alpha=0.3, label='Volatility Expanding')
    plt.title('ATR Change and Volatility Expansion')
    plt.ylabel('ATR Change')
    plt.grid(True)
    plt.legend()
    
    # Add signal information as text
    signal_text = f"Signal: {signal['direction']} (Strength: {signal.get('signal_strength', 0):.2f})"
    reason = signal.get('metadata', {}).get('reason', 'No reason provided')
    plt.figtext(0.5, 0.01, f"{signal_text}\nReason: {reason}", ha='center', fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('volatility_strategy.png')
    print("✅ Saved visualization to volatility_strategy.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    test_volatility_strategy()
