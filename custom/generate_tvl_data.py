"""
Generate synthetic TVL and price data for testing the TVL Indicator Strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_tvl_data(days=365, with_divergence=True, filename='ETHUSDT86400.csv'):
    """Generate synthetic TVL and price data for testing
    
    Args:
        days: Number of days to generate
        with_divergence: Whether to include periods of divergence between price and TVL
        filename: Filename to save the data
    """
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
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
    
    # Add divergence periods if requested
    if with_divergence:
        # Add a period where price rises but TVL falls
        divergence_start = n // 3
        divergence_end = divergence_start + n // 6
        tvl_trend[divergence_start:divergence_end] = np.linspace(
            tvl_trend[divergence_start], 
            tvl_trend[divergence_start] * 0.8, 
            divergence_end - divergence_start
        )
        
        # Add a period where price falls but TVL rises
        divergence_start = 2 * n // 3
        divergence_end = divergence_start + n // 6
        price_dip = np.linspace(
            prices[divergence_start], 
            prices[divergence_start] * 0.8, 
            divergence_end - divergence_start
        )
        prices[divergence_start:divergence_end] = price_dip
    
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
    
    # Add timestamp column for TVL merging
    data['timestamp'] = data['date'].astype(int) // 10**9
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Generated synthetic TVL data with {days} days and saved to {filename}")
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    
    # Plot price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(dates, prices, label='Price')
    ax1.set_title('Synthetic Price Data')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot TVL
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(dates, tvl, label='TVL')
    ax2.set_title('Synthetic TVL Data')
    ax2.set_ylabel('TVL')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('synthetic_tvl_data.png')
    print("Saved visualization to synthetic_tvl_data.png")
    
    return data

if __name__ == "__main__":
    generate_tvl_data(days=365, with_divergence=True, filename='ETHUSDT86400.csv')
