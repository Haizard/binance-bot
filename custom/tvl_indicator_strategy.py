"""
TVL Indicator Strategy

This strategy uses Total Value Locked (TVL) data from DeFiLlama API to generate trading signals.
It fits a rolling linear model mapping TVL to closing price and calculates the difference between
actual close price and TVL-predicted close price. This difference is normalized by dividing by
the average true range (ATR) to create the TVL indicator.

The strategy generates buy signals when the TVL indicator is below a certain threshold (price
is undervalued compared to TVL) and sell signals when the TVL indicator is above a certain
threshold (price is overvalued compared to TVL).
"""

import pandas as pd
import numpy as np
import logging
import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import pandas_ta as ta
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger("TVL Indicator Strategy")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TVLIndicatorStrategy:
    """TVL Indicator Strategy

    This strategy uses Total Value Locked (TVL) data from DeFiLlama API to generate trading signals.
    It fits a rolling linear model mapping TVL to closing price and calculates the difference between
    actual close price and TVL-predicted close price.
    """

    def __init__(self, name: str = "TVL Indicator Strategy"):
        """Initialize the strategy

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.description = (
            "TVL Indicator Strategy that uses Total Value Locked (TVL) data from DeFiLlama API "
            "to generate trading signals. It fits a rolling linear model mapping TVL to closing "
            "price and calculates the difference between actual close price and TVL-predicted "
            "close price."
        )

        # Default parameters
        self.parameters = {
            'fit_length': 7,  # Rolling window size for linear model fitting
            'atr_period': 30,  # ATR period for volatility normalization
            'buy_threshold': -0.5,  # Threshold for buy signals
            'sell_threshold': 0.5,  # Threshold for sell signals
            'risk_per_trade': 0.02,  # Risk per trade (2%)
            'take_profit_ratio': 1.5,  # Risk:Reward ratio
            'atr_multiplier': 2.0,  # Multiplier for ATR (for stop loss)
            'signal_threshold': 0.5,  # Minimum signal strength to generate a trade
            'chain': 'Ethereum',  # Blockchain to get TVL data for
            'cache_expiry': 3600,  # Cache expiry time in seconds (1 hour)
        }

        # Initialize cache for TVL data and API calls
        self.tvl_cache = {}
        self.indicator_cache = {}  # For testing purposes
        self.last_api_call = 0
        self.api_rate_limit = 1  # Minimum seconds between API calls

    def _get_tvl_data(self, chain: str) -> pd.DataFrame:
        """Get TVL data from DeFiLlama API

        Args:
            chain: Blockchain to get TVL data for (e.g., 'Ethereum', 'Binance', 'Solana')

        Returns:
            DataFrame with TVL data
        """
        # Check cache first
        current_time = time.time()
        if chain in self.tvl_cache:
            cache_time, cache_data = self.tvl_cache[chain]
            if current_time - cache_time < self.parameters['cache_expiry']:
                logger.debug(f"Using cached TVL data for {chain}")
                return cache_data

        # Rate limit API calls
        if current_time - self.last_api_call < self.api_rate_limit:
            time.sleep(self.api_rate_limit - (current_time - self.last_api_call))

        # Make API call
        try:
            logger.info(f"Fetching TVL data for {chain} from DeFiLlama API")

            # For testing purposes, check if the data already has TVL column
            # This allows us to bypass the API call in test scenarios
            if 'tvl' in self.indicator_cache:
                logger.info("Using cached TVL data from indicator_cache")
                return self.indicator_cache['tvl']

            url = f'https://api.llama.fi/charts/{chain}'
            response = requests.get(url)
            self.last_api_call = time.time()

            if response.status_code != 200:
                logger.error(f"Error fetching TVL data: {response.status_code} - {response.text}")
                return pd.DataFrame()

            # Parse response
            data = json.loads(response.text)
            tvl_df = pd.DataFrame(data)

            # Convert date to timestamp
            tvl_df['date'] = tvl_df['date'].astype(int)

            # Shift date column so data is concurrent with price data
            tvl_df['date'] = tvl_df['date'].shift(1)
            tvl_df = tvl_df.set_index('date')

            # Cache the data
            self.tvl_cache[chain] = (current_time, tvl_df)

            return tvl_df
        except Exception as e:
            logger.error(f"Error fetching TVL data: {e}")
            return pd.DataFrame()

    def _rolling_fit(self, df: pd.DataFrame, x_col: str, y_col: str, window: int) -> np.ndarray:
        """Fit a rolling linear model mapping x to y

        Args:
            df: DataFrame with data
            x_col: Column name for x values
            y_col: Column name for y values
            window: Rolling window size

        Returns:
            Array with predicted y values
        """
        pred = np.full(df.shape[0], np.nan)

        try:
            for i in range(window - 1, df.shape[0]):
                x_slice = df[x_col].iloc[i - window + 1: i+1]
                y_slice = df[y_col].iloc[i - window + 1: i+1]

                # Apply log transform to handle scaling issues
                x_slice = np.log(x_slice)
                y_slice = np.log(y_slice)

                # Fit linear model
                coefs = np.polyfit(x_slice, y_slice, 1)

                # Predict current value
                pred[i] = coefs[0] * x_slice.iloc[-1] + coefs[1]

                # Convert back from log scale
                pred[i] = np.exp(pred[i])
        except Exception as e:
            logger.error(f"Error fitting rolling model: {e}")

        return pred

    def _calculate_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range (ATR)

        Args:
            df: DataFrame with OHLCV data
            window: ATR period

        Returns:
            Series with ATR values
        """
        try:
            data = df.copy()
            high = data['high']
            low = data['low']
            close = data['close']

            data['tr0'] = abs(high - low)
            data['tr1'] = abs(high - close.shift())
            data['tr2'] = abs(low - close.shift())

            tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
            atr = tr.rolling(window).mean()

            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(np.nan, index=df.index)

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on TVL indicator

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with signal information
        """
        if data is None or len(data) < self.parameters['fit_length'] + 10:
            logger.warning("Insufficient data for TVL indicator analysis")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': 'Insufficient data for TVL indicator analysis'
                }
            }

        try:
            # Ensure data has required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'metadata': {
                        'reason': f"Missing required columns: {missing_columns}"
                    }
                }

            # Get symbol from data if available
            symbol = data.get('symbol', ['Unknown'])[0] if 'symbol' in data else 'Unknown'

            # Map symbol to chain
            chain = self.parameters['chain']
            if symbol.startswith('ETH'):
                chain = 'Ethereum'
            elif symbol.startswith('BNB'):
                chain = 'Binance'
            elif symbol.startswith('SOL'):
                chain = 'Solana'

            # Check if data already has TVL column (for testing)
            if 'tvl' in data.columns:
                logger.info("Using TVL data from input dataframe")
                df = data.copy()
                # Store in cache for future use
                self.indicator_cache['tvl'] = df
            else:
                # Get TVL data from API
                tvl_df = self._get_tvl_data(chain)

                if tvl_df.empty:
                    logger.warning(f"No TVL data available for {chain}")
                    return {
                        'direction': 'NEUTRAL',
                        'signal_strength': 0.0,
                        'entry_price': None,
                        'stop_loss': None,
                        'take_profit': None,
                        'metadata': {
                            'reason': f"No TVL data available for {chain}"
                        }
                    }

                # Create a copy of the data
                df = data.copy()

                # Convert index to timestamp for merging with TVL data
                if isinstance(df.index, pd.DatetimeIndex):
                    df['timestamp'] = df.index.astype('int64') // 10**9
                else:
                    df['timestamp'] = df.index

                # Add TVL to price data
                df['tvl'] = np.nan
                for i, row in df.iterrows():
                    timestamp = row['timestamp']
                    if timestamp in tvl_df.index:
                        df.at[i, 'tvl'] = tvl_df.loc[timestamp, 'totalLiquidityUSD']

                # Drop rows with missing TVL data
                df = df.dropna(subset=['tvl'])

            if len(df) < self.parameters['fit_length'] + 10:
                logger.warning(f"Insufficient data after merging with TVL data")
                return {
                    'direction': 'NEUTRAL',
                    'signal_strength': 0.0,
                    'entry_price': None,
                    'stop_loss': None,
                    'take_profit': None,
                    'metadata': {
                        'reason': f"Insufficient data after merging with TVL data"
                    }
                }

            # Calculate predicted price from TVL
            df['pred'] = self._rolling_fit(
                df,
                'tvl',
                'close',
                self.parameters['fit_length']
            )

            # Calculate ATR
            df['atr'] = self._calculate_atr(df, self.parameters['atr_period'])

            # Calculate TVL indicator
            df['tvl_indicator'] = (df['close'] - df['pred']) / df['atr']

            # Get current values
            current_close = df['close'].iloc[-1]
            current_tvl = df['tvl'].iloc[-1]
            current_pred = df['pred'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            current_indicator = df['tvl_indicator'].iloc[-1]

            # Initialize signal
            signal = {
                'symbol': symbol,
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'tvl': current_tvl,
                    'predicted_price': current_pred,
                    'atr': current_atr,
                    'tvl_indicator': current_indicator,
                    'reason': 'No signal'
                }
            }

            # Check for buy signal
            if current_indicator < self.parameters['buy_threshold']:
                # Calculate signal strength based on indicator value
                signal_strength = min(1.0, abs(current_indicator / self.parameters['buy_threshold']))

                if signal_strength >= self.parameters['signal_threshold']:
                    # Calculate stop loss and take profit
                    stop_loss = current_close - (current_atr * self.parameters['atr_multiplier'])
                    risk = current_close - stop_loss
                    take_profit = current_close + (risk * self.parameters['take_profit_ratio'])

                    # Update signal
                    signal.update({
                        'direction': 'BUY',
                        'signal_strength': signal_strength,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            **signal['metadata'],
                            'reason': 'Price is undervalued compared to TVL',
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })

            # Check for sell signal
            elif current_indicator > self.parameters['sell_threshold']:
                # Calculate signal strength based on indicator value
                signal_strength = min(1.0, abs(current_indicator / self.parameters['sell_threshold']))

                if signal_strength >= self.parameters['signal_threshold']:
                    # Calculate stop loss and take profit
                    stop_loss = current_close + (current_atr * self.parameters['atr_multiplier'])
                    risk = stop_loss - current_close
                    take_profit = current_close - (risk * self.parameters['take_profit_ratio'])

                    # Update signal
                    signal.update({
                        'direction': 'SELL',
                        'signal_strength': signal_strength,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'metadata': {
                            **signal['metadata'],
                            'reason': 'Price is overvalued compared to TVL',
                            'risk_reward_ratio': self.parameters['take_profit_ratio']
                        }
                    })

            return signal
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'metadata': {
                    'reason': f"Error generating signals: {str(e)}"
                }
            }
