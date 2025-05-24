"""
Test script for the Pip Pattern Miner Strategy
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

def generate_test_data(n_days=200):
    """Generate synthetic test data with repeating patterns and a strong trend at the end"""
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create price data with repeating patterns
    np.random.seed(42)  # For reproducibility

    # Base trend
    x = np.arange(len(date_range))
    trend = 100 + 0.05 * x

    # Add repeating pattern
    pattern_length = 20
    n_patterns = len(date_range) // pattern_length

    # Create a pattern
    pattern = np.concatenate([
        np.linspace(0, 5, pattern_length // 4),  # Up
        np.linspace(5, 3, pattern_length // 4),  # Down a bit
        np.linspace(3, 8, pattern_length // 4),  # Up more
        np.linspace(8, 0, pattern_length // 4)   # Down to start
    ])

    # Repeat the pattern
    repeated_pattern = np.tile(pattern, n_patterns + 1)[:len(date_range)]

    # Add a strong uptrend at the end to generate a BUY signal
    last_30_days = np.linspace(0, 15, 30)  # Strong uptrend in last 30 days
    if len(date_range) >= 30:
        repeated_pattern[-30:] = last_30_days

    # Add noise (reduced for clearer pattern)
    noise = np.random.normal(0, 0.3, len(date_range))

    # Combine components
    price = trend + repeated_pattern + noise

    # Create OHLC data
    data = pd.DataFrame({
        'open': price - np.random.uniform(0, 0.2, len(date_range)),
        'high': price + np.random.uniform(0.2, 0.4, len(date_range)),
        'low': price - np.random.uniform(0.2, 0.4, len(date_range)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, len(date_range))
    }, index=date_range)

    return data

def test_pip_pattern_miner_strategy():
    """Test the Pip Pattern Miner Strategy"""
    print("Testing Pip Pattern Miner Strategy...")

    # Create strategy instance
    strategy = StrategyFactory.create_strategy("pip_pattern_miner")

    # Print strategy info
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    print(f"Timeframes: {strategy.timeframes}")
    print(f"Symbols: {strategy.symbols}")

    # Generate test data
    data = generate_test_data()

    # Add symbol column
    data['symbol'] = 'BTCUSD'

    # Generate signals
    signal = strategy.generate_signals(data)

    # Print signal
    print("\nSignal:")
    print(f"Direction: {signal['direction']}")
    print(f"Signal Strength: {signal['signal_strength']}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"Metadata: {signal['metadata']}")

    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'])
    plt.title('Test Data with Repeating Patterns')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pip_pattern_test.png')
    print("\nTest data plot saved to 'pip_pattern_test.png'")

    # Plot PIPs
    if 'pip_points' in signal['metadata']:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[-strategy.parameters['lookback']:], data['close'].iloc[-strategy.parameters['lookback']:])

        # Extract PIP points
        pip_points = signal['metadata']['pip_points']

        # Adjust x indices to match the data index
        start_idx = len(data) - strategy.parameters['lookback']
        for i, (x, y) in enumerate(pip_points):
            plt.plot(data.index[start_idx + x], y, 'ro', markersize=8)
            plt.text(data.index[start_idx + x], y, f'PIP {i+1}', fontsize=10)

        plt.title('Perceptually Important Points (PIPs)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pip_points.png')
        print("PIP points plot saved to 'pip_points.png'")

    return strategy, data, signal

def test_with_real_data(csv_file='BTCUSDT3600.csv'):
    """Test the strategy with real data"""
    print(f"\nTesting with real data from {csv_file}...")

    try:
        # Load data
        data = pd.read_csv(csv_file)
        data['date'] = data['date'].astype('datetime64[s]')
        data = data.set_index('date')

        # Add symbol column
        data['symbol'] = 'BTCUSD'

        # Create strategy instance
        strategy = StrategyFactory.create_strategy("pip_pattern_miner")

        # Generate signals
        signal = strategy.generate_signals(data)

        # Print signal
        print("\nSignal:")
        print(f"Direction: {signal['direction']}")
        print(f"Signal Strength: {signal['signal_strength']}")
        print(f"Entry Price: {signal['entry_price']}")
        print(f"Stop Loss: {signal['stop_loss']}")
        print(f"Take Profit: {signal['take_profit']}")
        print(f"Metadata: {signal['metadata']}")

        return strategy, data, signal

    except FileNotFoundError:
        print(f"File {csv_file} not found. Skipping real data test.")
        return None, None, None

if __name__ == "__main__":
    # Test with synthetic data
    strategy, data, signal = test_pip_pattern_miner_strategy()

    # Test with real data if available
    test_with_real_data()
