"""
Test script for the Flags and Pennants Strategy
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
from src.strategies.custom.flags_pennants import plot_flag

def generate_test_data(n_days=50, create_pattern_at_end=True):
    """Generate synthetic test data with a flag pattern

    Args:
        n_days: Number of days of data to generate
        create_pattern_at_end: If True, creates a pattern that completes at the last candle
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create price data with a flag pattern
    np.random.seed(42)  # For reproducibility

    # Create a very simple pattern without noise
    price = np.ones(len(date_range)) * 100  # Start with flat price

    if create_pattern_at_end:
        # Create a perfect bull flag pattern at the end
        # Structure:
        # 1. Flat base
        # 2. Sharp uptrend (pole)
        # 3. Slight downtrend (flag)
        # 4. Breakout at the very end

        # Define pattern parameters
        base_length = 20
        pole_length = 10
        flag_length = 10

        # Calculate positions
        pole_start = base_length
        pole_end = pole_start + pole_length
        flag_end = pole_end + flag_length

        # Create pole (sharp uptrend)
        pole_height = 30
        for i in range(pole_length):
            price[pole_start + i] = 100 + (i+1) * (pole_height / pole_length)

        # Create flag (slight downtrend)
        flag_drop = 5
        for i in range(flag_length):
            price[pole_end + i] = price[pole_end - 1] - (i+1) * (flag_drop / flag_length)

        # Create breakout at the very end
        price[-1] = price[flag_end - 1] + 5  # Sharp breakout at the end

        print(f"Pattern details:")
        print(f"Base: {0} to {pole_start}")
        print(f"Pole: {pole_start} to {pole_end}")
        print(f"Flag: {pole_end} to {flag_end}")
        print(f"Breakout: {flag_end} to {len(price)}")
        print(f"Last index: {len(price) - 1}")

    else:
        # Create a pattern not at the end
        # Define pattern parameters
        base_length = 10
        pole_length = 10
        flag_length = 10

        # Calculate positions
        pole_start = base_length
        pole_end = pole_start + pole_length
        flag_end = pole_end + flag_length

        # Create pole (sharp uptrend)
        pole_height = 30
        for i in range(pole_length):
            price[pole_start + i] = 100 + (i+1) * (pole_height / pole_length)

        # Create flag (slight downtrend)
        flag_drop = 5
        for i in range(flag_length):
            price[pole_end + i] = price[pole_end - 1] - (i+1) * (flag_drop / flag_length)

        # Create breakout after the flag
        for i in range(len(date_range) - flag_end):
            price[flag_end + i] = price[flag_end - 1] + (i+1) * 0.5

    # No need to combine components, price is already set

    # Create OHLC data with smaller ranges to make pattern clearer
    data = pd.DataFrame({
        'open': price - np.random.uniform(0, 0.2, len(date_range)),
        'high': price + np.random.uniform(0.2, 0.4, len(date_range)),
        'low': price - np.random.uniform(0.2, 0.4, len(date_range)),
        'close': price,
        'volume': np.random.uniform(1000, 5000, len(date_range))
    }, index=date_range)

    # Print the last few values to verify the pattern
    print("\nLast few close prices to verify pattern:")
    print(data['close'].tail(10))

    return data

def test_flags_pennants_strategy(debug=True):
    """Test the Flags and Pennants Strategy

    Args:
        debug: If True, adds debugging information
    """
    print("Testing Flags and Pennants Strategy...")

    # Create strategy instance
    strategy = StrategyFactory.create_strategy("flags_pennants")

    # Modify parameters for easier pattern detection
    strategy.parameters['order'] = 5  # Reduce order for easier detection
    strategy.parameters['log_transform'] = False  # Disable log transform for this test

    # Print strategy info
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    print(f"Timeframes: {strategy.timeframes}")
    print(f"Symbols: {strategy.symbols}")

    # Generate test data with pattern at the end
    data = generate_test_data(create_pattern_at_end=True)

    # Add symbol column
    data['symbol'] = 'BTCUSD'

    # Add debugging to see what's happening inside the signal generation
    if debug:
        # Import the flags and pennants functions for debugging
        from src.strategies.custom.flags_pennants import find_flags_pennants_trendline, find_flags_pennants_pips

        # Extract close prices
        close_prices = data['close'].to_numpy()

        # Apply log transform if enabled
        if strategy.parameters['log_transform']:
            close_prices_transformed = np.log(close_prices)
        else:
            close_prices_transformed = close_prices

        # Detect patterns using both methods
        print("\nDetecting patterns using trendline method:")
        bull_flags_tl, bear_flags_tl, bull_pennants_tl, bear_pennants_tl = find_flags_pennants_trendline(
            close_prices_transformed, strategy.parameters['order']
        )

        print(f"Bull Flags: {len(bull_flags_tl)}")
        print(f"Bear Flags: {len(bear_flags_tl)}")
        print(f"Bull Pennants: {len(bull_pennants_tl)}")
        print(f"Bear Pennants: {len(bear_pennants_tl)}")

        print("\nDetecting patterns using PIPs method:")
        bull_flags_pip, bear_flags_pip, bull_pennants_pip, bear_pennants_pip = find_flags_pennants_pips(
            close_prices_transformed, strategy.parameters['order']
        )

        print(f"Bull Flags: {len(bull_flags_pip)}")
        print(f"Bear Flags: {len(bear_flags_pip)}")
        print(f"Bull Pennants: {len(bull_pennants_pip)}")
        print(f"Bear Pennants: {len(bear_pennants_pip)}")

        # Check if any patterns were confirmed in the last candle
        print("\nChecking for patterns confirmed in the last candle:")
        last_candle_idx = len(close_prices) - 1

        for i, flag in enumerate(bull_flags_tl):
            if flag.conf_x == last_candle_idx:
                print(f"Bull Flag (trendline) confirmed in last candle: {flag}")

        for i, pennant in enumerate(bull_pennants_tl):
            if pennant.conf_x == last_candle_idx:
                print(f"Bull Pennant (trendline) confirmed in last candle: {pennant}")

        for i, flag in enumerate(bear_flags_tl):
            if flag.conf_x == last_candle_idx:
                print(f"Bear Flag (trendline) confirmed in last candle: {flag}")

        for i, pennant in enumerate(bear_pennants_tl):
            if pennant.conf_x == last_candle_idx:
                print(f"Bear Pennant (trendline) confirmed in last candle: {pennant}")

        for i, flag in enumerate(bull_flags_pip):
            if flag.conf_x == last_candle_idx:
                print(f"Bull Flag (PIPs) confirmed in last candle: {flag}")

        for i, pennant in enumerate(bull_pennants_pip):
            if pennant.conf_x == last_candle_idx:
                print(f"Bull Pennant (PIPs) confirmed in last candle: {pennant}")

        for i, flag in enumerate(bear_flags_pip):
            if flag.conf_x == last_candle_idx:
                print(f"Bear Flag (PIPs) confirmed in last candle: {flag}")

        for i, pennant in enumerate(bear_pennants_pip):
            if pennant.conf_x == last_candle_idx:
                print(f"Bear Pennant (PIPs) confirmed in last candle: {pennant}")

    # Try both detection methods with modified parameters for easier detection
    print("\nTrying trendline detection method:")
    strategy.parameters['detection_method'] = 'trendline'
    strategy.parameters['order'] = 5  # Reduce order for easier detection
    signal_trendline = strategy.generate_signals(data)

    print("\nTrying PIPs detection method:")
    strategy.parameters['detection_method'] = 'pips'
    strategy.parameters['order'] = 5  # Reduce order for easier detection
    signal_pips = strategy.generate_signals(data)

    # Use the signal that's not neutral, or default to trendline
    signal = signal_trendline if signal_trendline['direction'] != 'NEUTRAL' else signal_pips

    # Print signal
    print("\nFinal Signal:")
    print(f"Direction: {signal['direction']}")
    print(f"Signal Strength: {signal['signal_strength']}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"Metadata: {signal['metadata']}")

    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'])
    plt.title('Test Data with Flag Pattern')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('flag_pattern_test.png')
    print("\nTest data plot saved to 'flag_pattern_test.png'")

    # If we found a pattern, try to plot it
    if signal['direction'] != 'NEUTRAL' and 'pattern_metrics' in signal['metadata']:
        try:
            # Find the pattern that was detected
            pattern_type = signal['metadata']['pattern_metrics']['pattern_type']
            print(f"\nDetected pattern type: {pattern_type}")

            # Plot the pattern if possible
            if debug and 'bull_flag' in pattern_type:
                for flag in bull_flags_tl:
                    if flag.conf_x == len(close_prices) - 1:
                        print("Plotting detected bull flag pattern...")
                        plot_flag(data, flag, pad=10)
                        break
                for flag in bull_flags_pip:
                    if flag.conf_x == len(close_prices) - 1:
                        print("Plotting detected bull flag pattern (PIPs)...")
                        plot_flag(data, flag, pad=10)
                        break
        except Exception as e:
            print(f"Error plotting pattern: {str(e)}")

    return strategy, data, signal

def test_with_real_data(csv_file='BTCUSDT86400.csv'):
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
        strategy = StrategyFactory.create_strategy("flags_pennants")

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
    strategy, data, signal = test_flags_pennants_strategy(debug=True)

    # Test with real data if available
    # Uncomment the line below if you have real data
    # test_with_real_data('BTCUSDT3600.csv')
