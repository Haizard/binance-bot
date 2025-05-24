"""
Test script for the Intramarket Difference Strategy
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

def test_intramarket_difference_strategy():
    """Test the Intramarket Difference Strategy"""
    print("\n=== Testing Intramarket Difference Strategy ===")
    
    # Create strategy instance
    try:
        strategy = StrategyFactory.create_strategy("intramarket_difference")
        print(f"✅ Successfully created strategy: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
    except Exception as e:
        print(f"❌ Error creating strategy: {str(e)}")
        return
    
    # Load test data
    try:
        # Load BTC data
        btc_data = pd.read_csv('BTCUSDT3600.csv')
        btc_data['date'] = pd.to_datetime(btc_data['date'])
        btc_data.set_index('date', inplace=True)
        btc_data['symbol'] = 'BTC'
        
        # Load ETH data
        eth_data = pd.read_csv('ETHUSDT3600.csv')
        eth_data['date'] = pd.to_datetime(eth_data['date'])
        eth_data.set_index('date', inplace=True)
        eth_data['symbol'] = 'ETH'
        
        print(f"✅ Successfully loaded BTC data with {len(btc_data)} rows")
        print(f"✅ Successfully loaded ETH data with {len(eth_data)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print("Please make sure you have BTCUSDT3600.csv and ETHUSDT3600.csv files in the current directory")
        
        # If no data file exists, generate synthetic data
        print("Generating synthetic data for testing...")
        btc_data, eth_data = generate_synthetic_data()
        print(f"✅ Generated synthetic data with {len(btc_data)} rows")
    
    # Generate signals
    try:
        # Combine data
        combined_data = pd.concat([btc_data, eth_data])
        
        # Use a window of data
        start_date = combined_data.index.max() - timedelta(days=30)
        window = combined_data[combined_data.index >= start_date]
        
        print(f"Generating signals for window from {window.index.min()} to {window.index.max()}")
        print(f"Current ETH price: {window[window['symbol'] == 'ETH']['close'].iloc[-1]:.2f}")
        print(f"Current BTC price: {window[window['symbol'] == 'BTC']['close'].iloc[-1]:.2f}")
        
        # Generate signal
        signal = strategy.generate_signals(window)
        print(f"✅ Successfully generated signal: {signal['direction']}")
        print(f"Signal strength: {signal.get('signal_strength', 0)}")
        print(f"Reason: {signal.get('metadata', {}).get('reason', 'No reason provided')}")
        
        # Print indicator values
        primary_cmma = signal.get('metadata', {}).get('primary_cmma', 'N/A')
        reference_cmma = signal.get('metadata', {}).get('reference_cmma', 'N/A')
        intermarket_diff = signal.get('metadata', {}).get('intermarket_diff', 'N/A')
        
        print(f"\nIndicator values:")
        print(f"  Primary CMMA (ETH): {primary_cmma}")
        print(f"  Reference CMMA (BTC): {reference_cmma}")
        print(f"  Intermarket Difference: {intermarket_diff}")
        
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
        visualize_intramarket_difference(btc_data, eth_data, signal)
    except Exception as e:
        print(f"❌ Error visualizing strategy: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_synthetic_data():
    """Generate synthetic data for testing"""
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate BTC prices with a trend and some noise
    n = len(dates)
    btc_trend = np.linspace(0, 1, n)
    btc_noise = np.random.normal(0, 0.05, n)
    btc_prices = 10000 + 5000 * btc_trend + 500 * btc_noise
    
    # Generate ETH prices with correlation to BTC but some divergence
    eth_trend = np.linspace(0, 1.2, n)  # Slightly different trend
    eth_noise = np.random.normal(0, 0.08, n)  # More volatile
    eth_prices = 500 + 300 * eth_trend + 50 * eth_noise
    
    # Add some divergence periods
    # Period where ETH rises but BTC falls
    divergence_start = n // 3
    divergence_end = divergence_start + n // 10
    eth_prices[divergence_start:divergence_end] *= np.linspace(1, 1.2, divergence_end - divergence_start)
    btc_prices[divergence_start:divergence_end] *= np.linspace(1, 0.9, divergence_end - divergence_start)
    
    # Period where ETH falls but BTC rises
    divergence_start = 2 * n // 3
    divergence_end = divergence_start + n // 10
    eth_prices[divergence_start:divergence_end] *= np.linspace(1, 0.8, divergence_end - divergence_start)
    btc_prices[divergence_start:divergence_end] *= np.linspace(1, 1.1, divergence_end - divergence_start)
    
    # Create BTC DataFrame
    btc_data = pd.DataFrame({
        'date': dates,
        'open': btc_prices * (1 + np.random.normal(0, 0.01, n)),
        'high': btc_prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': btc_prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': btc_prices,
        'volume': np.random.uniform(1000, 5000, n),
        'symbol': 'BTC'
    })
    
    # Create ETH DataFrame
    eth_data = pd.DataFrame({
        'date': dates,
        'open': eth_prices * (1 + np.random.normal(0, 0.01, n)),
        'high': eth_prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': eth_prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': eth_prices,
        'volume': np.random.uniform(5000, 20000, n),
        'symbol': 'ETH'
    })
    
    # Ensure high is always >= open, close
    btc_data['high'] = np.maximum(btc_data['high'], btc_data[['open', 'close']].max(axis=1))
    eth_data['high'] = np.maximum(eth_data['high'], eth_data[['open', 'close']].max(axis=1))
    
    # Ensure low is always <= open, close
    btc_data['low'] = np.minimum(btc_data['low'], btc_data[['open', 'close']].min(axis=1))
    eth_data['low'] = np.minimum(eth_data['low'], eth_data[['open', 'close']].min(axis=1))
    
    # Set date as index
    btc_data.set_index('date', inplace=True)
    eth_data.set_index('date', inplace=True)
    
    return btc_data, eth_data

def calculate_cmma(data, lookback=24, atr_lookback=168):
    """Calculate CMMA (Close Minus Moving Average) indicator"""
    # Calculate ATR
    atr = data['high'].rolling(atr_lookback).max() - data['low'].rolling(atr_lookback).min()
    
    # Calculate moving average
    ma = data['close'].rolling(lookback).mean()
    
    # Calculate CMMA
    cmma = (data['close'] - ma) / (atr * np.sqrt(lookback))
    
    return cmma

def visualize_intramarket_difference(btc_data, eth_data, signal):
    """Visualize intramarket difference strategy and signals"""
    print("\n=== Visualizing Intramarket Difference Strategy ===")
    
    # Calculate indicators
    lookback = 24
    atr_lookback = 168
    
    # Calculate CMMA for both assets
    btc_cmma = calculate_cmma(btc_data, lookback, atr_lookback)
    eth_cmma = calculate_cmma(eth_data, lookback, atr_lookback)
    
    # Ensure both series have the same index
    common_index = btc_cmma.index.intersection(eth_cmma.index)
    btc_cmma = btc_cmma.loc[common_index]
    eth_cmma = eth_cmma.loc[common_index]
    
    # Calculate intermarket difference
    intermarket_diff = eth_cmma - btc_cmma
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot prices
    plt.subplot(3, 1, 1)
    ax1 = plt.gca()
    ax1.plot(btc_data.index, btc_data['close'], 'b-', label='BTC Close')
    ax2 = plt.twinx()
    ax2.plot(eth_data.index, eth_data['close'], 'r-', label='ETH Close')
    ax1.set_ylabel('BTC Price', color='b')
    ax2.set_ylabel('ETH Price', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('BTC and ETH Prices')
    plt.grid(True)
    
    # Plot CMMA indicators
    plt.subplot(3, 1, 2)
    plt.plot(common_index, btc_cmma, 'b-', label='BTC CMMA')
    plt.plot(common_index, eth_cmma, 'r-', label='ETH CMMA')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('CMMA Indicators')
    plt.ylabel('CMMA Value')
    plt.grid(True)
    plt.legend()
    
    # Plot intermarket difference
    plt.subplot(3, 1, 3)
    plt.plot(common_index, intermarket_diff, 'g-', label='ETH-BTC Difference')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=0.25, color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(y=-0.25, color='g', linestyle='--', label='Lower Threshold')
    plt.title('Intermarket Difference')
    plt.ylabel('Difference')
    plt.grid(True)
    plt.legend()
    
    # Add signal information as text
    signal_text = f"Signal: {signal['direction']} (Strength: {signal.get('signal_strength', 0):.2f})"
    reason = signal.get('metadata', {}).get('reason', 'No reason provided')
    plt.figtext(0.5, 0.01, f"{signal_text}\nReason: {reason}", ha='center', fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('intramarket_difference_strategy.png')
    print("✅ Saved visualization to intramarket_difference_strategy.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    test_intramarket_difference_strategy()
