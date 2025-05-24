"""
Volatility Hawkes Strategy Implementation

This module contains the implementation of the Volatility Hawkes strategy,
which uses a Hawkes process to model volatility clustering and generate
trading signals during periods of volatility expansion.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import scipy

def hawkes_process(data: pd.Series, kappa: float):
    """Apply a Hawkes process to a time series

    Args:
        data: Input time series
        kappa: Decay factor for the Hawkes process

    Returns:
        Series with Hawkes process applied
    """
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=data.index) * kappa

def vol_signal(close: pd.Series, vol_hawkes: pd.Series, lookback:int):
    """Generate trading signals based on volatility Hawkes process

    Args:
        close: Close price series
        vol_hawkes: Volatility Hawkes process series
        lookback: Lookback period for quantile calculation

    Returns:
        Array of signal values (1 for buy, -1 for sell, 0 for neutral)
    """
    signal = np.zeros(len(close))
    q05 = vol_hawkes.rolling(lookback).quantile(0.05)
    q95 = vol_hawkes.rolling(lookback).quantile(0.95)

    last_below = -1
    curr_sig = 0

    for i in range(len(signal)):
        if vol_hawkes.iloc[i] < q05.iloc[i]:
            last_below = i
            curr_sig = 0

        if vol_hawkes.iloc[i] > q95.iloc[i] \
           and vol_hawkes.iloc[i - 1] <= q95.iloc[i - 1] \
           and last_below > 0 :

            change = close.iloc[i] - close.iloc[last_below]
            if change > 0.0:
                curr_sig = 1
            else:
                curr_sig = -1
        signal[i] = curr_sig

    return signal

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    """Get trade entry and exit times from a signal array

    Args:
        data: DataFrame with OHLCV data
        signal: Array of signal values (1 for buy, -1 for sell, 0 for neutral)

    Returns:
        Tuple of (long_trades, short_trades) DataFrames
    """
    # Gets trade entry and exit times from a signal
    # that has values of -1, 0, 1. Denoting short,flat,and long.
    # No position sizing.

    long_trades = []
    short_trades = []

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index
    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0: # Long entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        if signal[i] == -1.0  and last_sig != -1.0: # Short entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]

        if signal[i] == 0.0 and last_sig == -1.0: # Short exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            short_trades.append(open_trade)
            open_trade = None

        if signal[i] == 0.0  and last_sig == 1.0: # Long exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            long_trades.append(open_trade)
            open_trade = None

        last_sig = signal[i]

    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
    short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
    long_trades = long_trades.set_index('entry_time')
    short_trades = short_trades.set_index('entry_time')
    return long_trades, short_trades

class VolatilityHawkesStrategy:
    """Volatility Hawkes Strategy

    This strategy uses a Hawkes process model for volatility analysis to identify
    periods of volatility clustering and generate trading signals.
    """

    def __init__(self, name="Volatility Hawkes Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name

        # Set parameters
        self.parameters = {
            'kappa': 0.1,  # Decay factor for the Hawkes process
            'lookback': 100,  # Lookback period for quantile calculation (reduced from 168)
            'risk_per_trade': 0.02,  # 2% risk per trade
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_period': 14,  # Period for ATR calculation
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'verbose': True  # Print detailed logs
        }

        # Set description
        self.description = (
            "Volatility Hawkes strategy that uses a Hawkes process to model volatility "
            "clustering and generate trading signals during periods of volatility expansion."
        )

    def generate_signals(self, data):
        """Generate trading signals based on the provided data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        try:
            # Ensure we have enough data
            if len(data) < self.parameters['lookback'] + 10:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'Not enough data for analysis'
                    }
                }

            # Calculate ATR for stop loss and normalize range
            atr = data['atr'] if 'atr' in data.columns else data['high'].rolling(window=self.parameters['atr_period']).max() - data['low'].rolling(window=self.parameters['atr_period']).min()
            if isinstance(atr, pd.Series):
                atr = atr.iloc[-1]

            # Calculate normalized range
            norm_range = (np.log(data['high']) - np.log(data['low'])) / atr

            # Apply Hawkes process
            v_hawk = hawkes_process(norm_range, self.parameters['kappa'])

            # Generate signal
            signal_value = vol_signal(data['close'], v_hawk, self.parameters['lookback'])

            # Get the most recent signal value
            current_signal = signal_value[-1] if isinstance(signal_value, np.ndarray) else signal_value

            # Generate trading signal based on the Hawkes process
            current_price = data['close'].iloc[-1]

            if current_signal > 0:
                # Calculate stop loss and take profit
                stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.parameters['take_profit_ratio'])

                return {
                    'direction': 'BUY',
                    'signal_strength': 0.8,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Volatility expansion with positive signal',
                        'hawkes_value': float(v_hawk.iloc[-1]) if isinstance(v_hawk, pd.Series) else float(v_hawk[-1])
                    }
                }
            elif current_signal < 0:
                # Calculate stop loss and take profit
                stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                risk = stop_loss - current_price
                take_profit = current_price - (risk * self.parameters['take_profit_ratio'])

                return {
                    'direction': 'SELL',
                    'signal_strength': 0.8,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'reason': 'Volatility expansion with negative signal',
                        'hawkes_value': float(v_hawk.iloc[-1]) if isinstance(v_hawk, pd.Series) else float(v_hawk[-1])
                    }
                }
            else:
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0,
                    'metadata': {
                        'reason': 'No volatility signal detected',
                        'hawkes_value': float(v_hawk.iloc[-1]) if isinstance(v_hawk, pd.Series) else float(v_hawk[-1])
                    }
                }
        except Exception as e:
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0,
                'metadata': {
                    'reason': f'Error in Hawkes signal generation: {str(e)}'
                }
            }
