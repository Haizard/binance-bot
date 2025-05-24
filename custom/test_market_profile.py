"""
🌙 Moon Dev's Market Profile Strategy Test
Test script for the Market Profile Strategy
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.strategies.custom.strategy_factory import StrategyFactory

def load_test_data(symbol="BTCUSDT", timeframe="1D", n_bars=365):
    """Load test data for backtesting

    Args:
        symbol: Symbol to load data for
        timeframe: Timeframe to load data for
        n_bars: Number of bars to load

    Returns:
        DataFrame with OHLCV data
    """
    # Try to load from CSV
    data_dir = os.path.join(project_root, "data")
    file_path = os.path.join(data_dir, f"{symbol}{timeframe}.csv")

    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        return data.tail(n_bars)

    # If file doesn't exist, generate synthetic data
    print(f"Generating synthetic data for {symbol} {timeframe}")
    np.random.seed(42)

    # Generate dates
    end_date = pd.Timestamp.now().floor('D')
    start_date = end_date - pd.Timedelta(days=n_bars)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_bars)

    # Generate prices
    close = 10000 + np.cumsum(np.random.normal(0, 200, n_bars))
    high = close + np.random.uniform(0, 200, n_bars)
    low = close - np.random.uniform(0, 200, n_bars)
    open_price = close - np.random.uniform(-100, 100, n_bars)
    volume = np.random.uniform(1000, 5000, n_bars)

    # Create DataFrame
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return data

def test_market_profile_strategy():
    """Test the Market Profile Strategy"""
    print("Testing Market Profile Strategy...")

    # Load test data
    data = load_test_data()

    # Create strategy
    strategy = StrategyFactory.create_strategy("market_profile")

    # Generate signals
    signal = strategy.generate_signals(data)

    # Print signal
    print(f"Signal: {signal}")

    # Test with different parameters
    print("\nTesting with different parameters:")
    custom_params = {
        'lookback_period': 50,
        'first_weight': 0.2,
        'atr_multiplier': 2.5,
        'prominence_threshold': 0.3,
        'signal_threshold': 0.7
    }

    # Create a new strategy with custom parameters
    custom_strategy = StrategyFactory.create_strategy("market_profile")
    custom_strategy.set_parameters(custom_params)
    custom_signal = custom_strategy.generate_signals(data)
    print(f"Custom Signal: {custom_signal}")

    # Test with a smaller window of data
    window = data.iloc[-100:]
    window_signal = strategy.generate_signals(window)
    print(f"\nSignal with smaller window: {window_signal}")

    return signal

if __name__ == "__main__":
    test_market_profile_strategy()
