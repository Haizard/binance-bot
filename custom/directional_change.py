import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    """
    Detect directional changes in price data.

    Args:
        close: Array of closing prices
        high: Array of high prices
        low: Array of low prices
        sigma: Threshold for directional change (e.g., 0.02 for 2%)

    Returns:
        Tuple of (tops, bottoms) where each is a list of [confirmation_index, extreme_index, extreme_price]
    """
    # Convert pandas Series to numpy arrays if needed to avoid FutureWarnings
    if isinstance(close, pd.Series):
        close = close.to_numpy()
    if isinstance(high, pd.Series):
        high = high.to_numpy()
    if isinstance(low, pd.Series):
        low = low.to_numpy()

    up_zig = True # Last extreme is a bottom. Next is a top.
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig: # Last extreme is a bottom
            if high[i] > tmp_max:
                # New high, update
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma:
                # Price retraced by sigma %. Top confirmed, record it
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # Setup for next bottom
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else: # Last extreme is a top
            if low[i] < tmp_min:
                # New low, update
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma:
                # Price retraced by sigma %. Bottom confirmed, record it
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms

def get_extremes(ohlc: pd.DataFrame, sigma: float):
    """
    Get extremes (tops and bottoms) from OHLC data using directional change algorithm.

    Args:
        ohlc: DataFrame with OHLC data (must have 'close', 'high', 'low' columns)
        sigma: Threshold for directional change (e.g., 0.02 for 2%)

    Returns:
        DataFrame with extremes, indexed by confirmation index, with columns:
        - ext_i: Index of extreme point
        - ext_p: Price of extreme point
        - type: 1 for tops, -1 for bottoms
    """
    tops, bottoms = directional_change(ohlc['close'], ohlc['high'], ohlc['low'], sigma)
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1
    extremes = pd.concat([tops, bottoms])
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()
    return extremes






















if __name__ == '__main__':
    """
    Example usage of directional change algorithm.
    """
    try:
        # Try to load data from CSV file
        data = pd.read_csv('BTCUSDT3600.csv')
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

        # Get directional changes
        tops, bottoms = directional_change(data['close'], data['high'], data['low'], 0.02)

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(data['close'], label='Close Price')

        # Plot tops and bottoms
        for top in tops:
            plt.plot(top[1], top[2], marker='^', color='green', markersize=8)

        for bottom in bottoms:
            plt.plot(bottom[1], bottom[2], marker='v', color='red', markersize=8)

        plt.title('Directional Change Algorithm (sigma=0.02)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Alternative: Use get_extremes function
        print("Using get_extremes function:")
        extremes = get_extremes(data, 0.02)
        print(f"Found {len(extremes)} extremes ({sum(extremes['type'] > 0)} tops, {sum(extremes['type'] < 0)} bottoms)")
        print(extremes.head())

    except FileNotFoundError:
        print("Example data file not found. This is just an example of how to use the directional_change function.")














