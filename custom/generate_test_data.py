"""
Generate test data for strategy testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data(periods=500, with_trend=True, volatility=0.02, filename='BTCUSDT3600.csv'):
    """Generate test data for strategy testing
    
    Args:
        periods: Number of periods to generate
        with_trend: Whether to include a trend
        volatility: Volatility of the price
        filename: Filename to save the data
    """
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=periods)
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Generate prices
    if with_trend:
        # Generate a price series with a trend
        trend = np.linspace(0, 1, periods)
        noise = np.random.normal(0, volatility, periods)
        prices = 10000 + 5000 * trend + 1000 * noise
    else:
        # Generate a random walk
        noise = np.random.normal(0, volatility, periods)
        prices = 10000 + np.cumsum(noise * 1000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, periods)),
        'low': prices * (1 - np.random.uniform(0, 0.01, periods)),
        'close': prices * (1 + np.random.normal(0, 0.005, periods)),
        'volume': np.random.uniform(100, 1000, periods)
    })
    
    # Ensure high is always >= open, close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    
    # Ensure low is always <= open, close
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Generated test data with {periods} periods and saved to {filename}")
    
    return data

if __name__ == "__main__":
    generate_test_data(periods=500, with_trend=True, volatility=0.02, filename='BTCUSDT3600.csv')
