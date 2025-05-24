"""
Simple test script for the Market Structure Analysis Strategy
"""

import os
import sys
import pandas as pd

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
    except Exception as e:
        print(f"❌ Error generating signals: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_market_structure_strategy()
