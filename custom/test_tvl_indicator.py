"""
Test script for the TVL Indicator Strategy
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

def test_tvl_indicator_strategy():
    """Test the TVL Indicator Strategy"""
    print("\n=== Testing TVL Indicator Strategy ===")

    # Create strategy instance
    try:
        strategy = StrategyFactory.create_strategy("tvl_indicator")
        print(f"✅ Successfully created strategy: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
    except Exception as e:
        print(f"❌ Error creating strategy: {str(e)}")
        return

    # Load test data
    try:
        data = pd.read_csv('ETHUSDT86400.csv')  # Daily data for Ethereum
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        print(f"✅ Successfully loaded data with {len(data)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print("Please make sure you have a ETHUSDT86400.csv file in the current directory")

        # If no data file exists, generate synthetic data
        print("Generating synthetic data for testing...")
        data = generate_synthetic_data()
        print(f"✅ Generated synthetic data with {len(data)} rows")

    # Generate signals
    try:
        # For testing purposes, let's add the TVL data directly to the window
        # This simulates what would happen if the API call succeeded
        window = data.iloc[-200:]

        print(f"Generating signals for window from {window.index[0]} to {window.index[-1]}")
        print(f"Current price: {window['close'].iloc[-1]:.2f}")

        # Create a modified copy for testing
        test_data = window.copy()

        # Calculate ATR for the test data
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        test_data['tr0'] = abs(high - low)
        test_data['tr1'] = abs(high - close.shift())
        test_data['tr2'] = abs(low - close.shift())
        tr = test_data[['tr0', 'tr1', 'tr2']].max(axis=1)
        test_data['atr'] = tr.rolling(30).mean()

        # Calculate predicted price from TVL using a simple model
        test_data['pred'] = np.nan
        fit_length = 7
        for i in range(fit_length - 1, len(test_data)):
            x_slice = test_data['tvl'].iloc[i - fit_length + 1: i+1]
            y_slice = test_data['close'].iloc[i - fit_length + 1: i+1]

            x_slice = np.log(x_slice)
            y_slice = np.log(y_slice)

            coefs = np.polyfit(x_slice, y_slice, 1)
            test_data.loc[test_data.index[i], 'pred'] = np.exp(coefs[0] * x_slice.iloc[-1] + coefs[1])

        # Calculate TVL indicator
        test_data['tvl_indicator'] = (test_data['close'] - test_data['pred']) / test_data['atr']

        # Create a divergence in the last few days to generate a signal
        last_n = 5
        # Increase price but keep TVL the same to create a positive indicator (sell signal)
        test_data.loc[test_data.index[-last_n:], 'close'] = test_data.loc[test_data.index[-last_n:], 'close'] * 1.1

        # Recalculate the indicator for the modified data
        test_data.loc[test_data.index[-last_n:], 'tvl_indicator'] = (
            (test_data.loc[test_data.index[-last_n:], 'close'] - test_data.loc[test_data.index[-last_n:], 'pred']) /
            test_data.loc[test_data.index[-last_n:], 'atr']
        )

        print(f"TVL Indicator value: {test_data['tvl_indicator'].iloc[-1]:.2f}")

        # Generate signal using our modified test data
        signal = strategy.generate_signals(test_data)
        print(f"✅ Successfully generated signal: {signal['direction']}")
        print(f"Signal strength: {signal.get('signal_strength', 0)}")
        print(f"Reason: {signal.get('metadata', {}).get('reason', 'No reason provided')}")

        # Print TVL and indicator values
        tvl = signal.get('metadata', {}).get('tvl', 'N/A')
        predicted_price = signal.get('metadata', {}).get('predicted_price', 'N/A')
        tvl_indicator = signal.get('metadata', {}).get('tvl_indicator', 'N/A')

        print(f"\nIndicator values:")
        print(f"  TVL: {tvl}")
        print(f"  Predicted Price: {predicted_price}")
        print(f"  TVL Indicator: {tvl_indicator}")

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
        visualize_tvl_indicator(test_data, signal)
    except Exception as e:
        print(f"❌ Error visualizing strategy: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_synthetic_data():
    """Generate synthetic data for testing"""
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate prices with a trend and some noise
    n = len(dates)
    trend = np.linspace(0, 1, n)
    noise = np.random.normal(0, 0.05, n)
    prices = 1000 + 500 * trend + 100 * noise

    # Generate TVL data with correlation to price but some divergence
    tvl_base = 1e9  # 1 billion
    tvl_trend = np.linspace(0, 1, n)
    tvl_noise = np.random.normal(0, 0.1, n)
    tvl = tvl_base + tvl_base * tvl_trend + tvl_base * 0.2 * tvl_noise

    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n),
        'tvl': tvl
    })

    # Ensure high is always >= open, close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))

    # Ensure low is always <= open, close
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    # Set date as index
    data.set_index('date', inplace=True)

    # Add timestamp column for TVL merging
    data['timestamp'] = data.index.astype('int64') // 10**9

    return data

def visualize_tvl_indicator(data, signal):
    """Visualize TVL indicator strategy and signals"""
    print("\n=== Visualizing TVL Indicator Strategy ===")

    # Check if TVL data is available
    if 'tvl' not in data.columns:
        print("❌ No TVL data available for visualization")
        return

    # Calculate indicators
    fit_length = 7
    atr_period = 30

    # Calculate ATR
    high = data['high']
    low = data['low']
    close = data['close']

    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    data['atr'] = tr.rolling(atr_period).mean()

    # Calculate predicted price from TVL
    data['pred'] = np.nan
    for i in range(fit_length - 1, len(data)):
        x_slice = data['tvl'].iloc[i - fit_length + 1: i+1]
        y_slice = data['close'].iloc[i - fit_length + 1: i+1]

        x_slice = np.log(x_slice)
        y_slice = np.log(y_slice)

        coefs = np.polyfit(x_slice, y_slice, 1)
        data.loc[data.index[i], 'pred'] = np.exp(coefs[0] * x_slice.iloc[-1] + coefs[1])

    # Calculate TVL indicator
    data['tvl_indicator'] = (data['close'] - data['pred']) / data['atr']

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot price and predicted price
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(data.index, data['pred'], 'r--', label='TVL Predicted Price')
    plt.title('TVL Indicator Strategy - Price vs Predicted Price')
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

    # Plot TVL
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['tvl'], label='TVL')
    plt.title('Total Value Locked (TVL)')
    plt.ylabel('TVL')
    plt.grid(True)
    plt.legend()

    # Plot TVL Indicator
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['tvl_indicator'], label='TVL Indicator')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Sell Threshold')
    plt.axhline(y=-0.5, color='g', linestyle='--', label='Buy Threshold')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('TVL Indicator')
    plt.ylabel('Indicator Value')
    plt.grid(True)
    plt.legend()

    # Add signal information as text
    signal_text = f"Signal: {signal['direction']} (Strength: {signal.get('signal_strength', 0):.2f})"
    reason = signal.get('metadata', {}).get('reason', 'No reason provided')
    plt.figtext(0.5, 0.01, f"{signal_text}\nReason: {reason}", ha='center', fontsize=12,
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('tvl_indicator_strategy.png')
    print("✅ Saved visualization to tvl_indicator_strategy.png")

    # Show plot
    plt.show()

if __name__ == "__main__":
    test_tvl_indicator_strategy()
