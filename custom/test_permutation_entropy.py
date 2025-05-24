"""
Test script for the Permutation Entropy Strategy
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.strategies.custom.strategy_factory import StrategyFactory

def test_permutation_entropy_strategy():
    """Test the Permutation Entropy Strategy"""
    print("\n=== Testing Permutation Entropy Strategy ===")
    
    # Create strategy instance
    try:
        strategy = StrategyFactory.create_strategy("permutation_entropy")
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
        
        # If no data file exists, generate synthetic data
        print("Generating synthetic data for testing...")
        data = generate_synthetic_data()
        print(f"✅ Generated synthetic data with {len(data)} rows")
    
    # Generate signals
    try:
        # Use a window of data
        window = data.iloc[-500:]
        
        print(f"Generating signals for window from {window.index[0]} to {window.index[-1]}")
        print(f"Current price: {window['close'].iloc[-1]:.2f}")
        
        # Generate signal
        signal = strategy.generate_signals(window)
        print(f"✅ Successfully generated signal: {signal['direction']}")
        print(f"Signal strength: {signal.get('signal_strength', 0)}")
        print(f"Reason: {signal.get('metadata', {}).get('reason', 'No reason provided')}")
        
        # Print entropy values
        entropy = signal.get('metadata', {}).get('entropy', 'N/A')
        entropy_change = signal.get('metadata', {}).get('entropy_change', 'N/A')
        
        print(f"\nIndicator values:")
        print(f"  Entropy: {entropy}")
        print(f"  Entropy Change: {entropy_change}")
        
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
        visualize_permutation_entropy(window, signal)
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
    
    # Generate prices with different regimes
    n = len(dates)
    
    # Base trend
    trend = np.linspace(0, 1, n)
    
    # Add different volatility regimes
    volatility = np.ones(n) * 0.01
    
    # High volatility regime
    high_vol_start = n // 3
    high_vol_end = high_vol_start + n // 6
    volatility[high_vol_start:high_vol_end] = 0.03
    
    # Low volatility regime
    low_vol_start = 2 * n // 3
    low_vol_end = low_vol_start + n // 6
    volatility[low_vol_start:low_vol_end] = 0.005
    
    # Generate price with different regimes
    price = 10000
    prices = [price]
    for i in range(1, n):
        # Add trend
        price_change = price * 0.0002
        
        # Add volatility
        price_change += np.random.normal(0, price * volatility[i])
        
        # Add some mean reversion
        if i > 20:
            mean_price = np.mean(prices[-20:])
            price_change += (mean_price - price) * 0.05
        
        # Add some momentum
        if i > 5:
            momentum = prices[-1] - prices[-5]
            price_change += momentum * 0.1
        
        price += price_change
        prices.append(price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, v * 2)) for p, v in zip(prices, volatility)],
        'low': [p * (1 - np.random.uniform(0, v * 2)) for p, v in zip(prices, volatility)],
        'close': prices,
        'volume': [v * 1000 + np.random.uniform(0, 500) for v in volatility]
    })
    
    # Ensure high is always >= open, close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    
    # Ensure low is always <= open, close
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Set date as index
    data.set_index('date', inplace=True)
    
    return data

def ordinal_patterns(arr, d):
    """Calculate ordinal patterns from time series data"""
    assert(d >= 2)
    fac = math.factorial(d)
    d1 = d - 1
    mults = []
    for i in range(1, d):
        mult = fac / math.factorial(i + 1)
        mults.append(mult)
   
    # Create array to put ordinal pattern in
    ordinals = np.empty(len(arr))
    ordinals[:] = np.nan

    for i in range(d1, len(arr)):
        dat = arr[i - d1:  i+1] 
        pattern_ordinal = 0
        for l in range(1, d): 
            count = 0
            for r in range(l):
                if dat[d1 - l] >= dat[d1 - r]:
                   count += 1
             
            pattern_ordinal += count * mults[l - 1]
        ordinals[i] = int(pattern_ordinal)
    
    return ordinals

def permutation_entropy(arr, d, mult):
    """Calculate permutation entropy from time series data"""
    fac = math.factorial(d)
    lookback = fac * mult
    
    ent = np.empty(len(arr))
    ent[:] = np.nan
    ordinals = ordinal_patterns(arr, d)
    
    for i in range(lookback + d - 1, len(arr)):
        window = ordinals[i - lookback + 1:i+1]
        
        # Create distribution
        freqs = pd.Series(window).value_counts().to_dict()
        for j in range(fac):
            if j in freqs:
                freqs[j] = freqs[j] / lookback
       
        # Calculate entropy
        perm_entropy = 0.0
        for k, v in freqs.items():
            perm_entropy += v * math.log2(v)

        # Normalize to 0-1
        perm_entropy = -1. * (1. / math.log2(fac)) * perm_entropy
        ent[i] = perm_entropy
        
    return ent

def visualize_permutation_entropy(data, signal):
    """Visualize permutation entropy strategy and signals"""
    print("\n=== Visualizing Permutation Entropy Strategy ===")
    
    # Calculate indicators
    dimension = 3
    mult = 28
    entropy_ma_period = 10
    
    # Calculate permutation entropy
    prices = data['close'].values
    entropy = permutation_entropy(prices, dimension, mult)
    
    # Calculate smoothed entropy
    entropy_ma = pd.Series(entropy).rolling(entropy_ma_period).mean().values
    
    # Calculate entropy change
    entropy_change = np.zeros_like(entropy_ma)
    entropy_change[:] = np.nan
    for i in range(entropy_ma_period, len(entropy_ma)):
        entropy_change[i] = entropy_ma[i] - entropy_ma[i - entropy_ma_period]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot price
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.title('Permutation Entropy Strategy - Price')
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
    
    # Plot permutation entropy
    plt.subplot(3, 1, 2)
    plt.plot(data.index, entropy, 'b-', alpha=0.3, label='Raw Entropy')
    plt.plot(data.index, entropy_ma, 'r-', label='Smoothed Entropy')
    plt.axhline(y=0.8, color='r', linestyle='--', label='High Entropy Threshold')
    plt.axhline(y=0.4, color='g', linestyle='--', label='Low Entropy Threshold')
    plt.title('Permutation Entropy')
    plt.ylabel('Entropy')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Plot entropy change
    plt.subplot(3, 1, 3)
    plt.plot(data.index, entropy_change, 'g-', label='Entropy Change')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=0.1, color='r', linestyle='--', label='Positive Change Threshold')
    plt.axhline(y=-0.1, color='g', linestyle='--', label='Negative Change Threshold')
    plt.title('Entropy Change')
    plt.ylabel('Change')
    plt.grid(True)
    plt.legend()
    
    # Add signal information as text
    signal_text = f"Signal: {signal['direction']} (Strength: {signal.get('signal_strength', 0):.2f})"
    reason = signal.get('metadata', {}).get('reason', 'No reason provided')
    plt.figtext(0.5, 0.01, f"{signal_text}\nReason: {reason}", ha='center', fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('permutation_entropy_strategy.png')
    print("✅ Saved visualization to permutation_entropy_strategy.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    test_permutation_entropy_strategy()
