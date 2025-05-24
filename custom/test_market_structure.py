"""
Test script for the Market Structure Analysis Strategy
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

def test_market_structure_strategy():
    """Test the Market Structure Analysis Strategy"""
    print("\n=== Testing Market Structure Analysis Strategy ===")

    # Create strategy instance
    try:
        strategy = StrategyFactory.create_strategy("market_structure")
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

        # Print support and resistance levels
        support_levels = signal.get('metadata', {}).get('support_levels', [])
        resistance_levels = signal.get('metadata', {}).get('resistance_levels', [])

        print(f"\nFound {len(support_levels)} support levels:")
        for i, level in enumerate(support_levels):
            print(f"  {i+1}. Price: {level['price']:.2f}, Level: {level['level']}, Strength: {level['strength']:.2f}")

        print(f"\nFound {len(resistance_levels)} resistance levels:")
        for i, level in enumerate(resistance_levels):
            print(f"  {i+1}. Price: {level['price']:.2f}, Level: {level['level']}, Strength: {level['strength']:.2f}")

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

    # Visualize market structure
    try:
        visualize_market_structure(window, signal)
    except Exception as e:
        print(f"❌ Error visualizing market structure: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_market_structure(data, signal):
    """Visualize market structure and signals"""
    print("\n=== Visualizing Market Structure ===")

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.title('Market Structure Analysis')
    plt.ylabel('Price')
    plt.grid(True)

    # Plot support and resistance levels
    support_levels = signal.get('metadata', {}).get('support_levels', [])
    resistance_levels = signal.get('metadata', {}).get('resistance_levels', [])

    for level in support_levels:
        plt.axhline(y=level['price'], color='g', linestyle='--', alpha=0.5,
                   label=f"Support L{level['level']} ({level['price']:.2f})")

    for level in resistance_levels:
        plt.axhline(y=level['price'], color='r', linestyle='--', alpha=0.5,
                   label=f"Resistance L{level['level']} ({level['price']:.2f})")

    # Plot breakout level if available
    breakout_level = signal.get('metadata', {}).get('breakout_level')
    if breakout_level:
        plt.axhline(y=breakout_level['price'], color='m', linestyle='-',
                   label=f"Breakout Level ({breakout_level['price']:.2f})")

    # Plot entry, stop loss, and take profit if available
    if signal['direction'] != 'NEUTRAL':
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        if entry_price:
            plt.axhline(y=entry_price, color='b', linestyle='-',
                       label=f"Entry ({entry_price:.2f})")

        if stop_loss:
            plt.axhline(y=stop_loss, color='r', linestyle='-',
                       label=f"Stop Loss ({stop_loss:.2f})")

        if take_profit:
            plt.axhline(y=take_profit, color='g', linestyle='-',
                       label=f"Take Profit ({take_profit:.2f})")

    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot signal strength
    plt.subplot(2, 1, 2)
    signal_strength = signal.get('signal_strength', 0)
    plt.bar(['Signal Strength'], [signal_strength], color='b' if signal['direction'] == 'BUY' else 'r' if signal['direction'] == 'SELL' else 'gray')
    plt.title(f"Signal: {signal['direction']} (Strength: {signal_strength:.2f})")
    plt.ylim(0, 1)
    plt.grid(True)

    # Add reason as text
    reason = signal.get('metadata', {}).get('reason', 'No reason provided')
    plt.figtext(0.5, 0.01, f"Reason: {reason}", ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('market_structure_analysis.png')
    print("✅ Saved visualization to market_structure_analysis.png")

    # Show plot
    plt.show()

if __name__ == "__main__":
    test_market_structure_strategy()
