import os
import asyncio
import logging
from decimal import Decimal
from datetime import datetime
from collections import deque, defaultdict
import numpy as np
import time

class MarkovTradingAgent:
    def __init__(self, lookback=20, states=('up', 'down', 'flat'), threshold=0.001):
        self.name = "MarkovTradingAgent"
        self.lookback = lookback  # Number of candles to look back
        self.states = states
        self.threshold = Decimal(str(threshold))  # Flat threshold
        self.price_history = defaultdict(lambda: deque(maxlen=self.lookback+1))  # symbol -> deque
        self.transition_counts = defaultdict(lambda: np.zeros((len(self.states), len(self.states))))  # symbol -> matrix
        self.state_map = {s: i for i, s in enumerate(self.states)}
        self.logger = logging.getLogger(self.name)
        self.last_signal = {}  # symbol -> last signal
        self.last_buy_time = {}  # symbol -> timestamp of last 'buy' signal
        self.signal_history = {}  # symbol -> deque of last N signals
        self.signal_history_len = 10

    def _get_state(self, prev_price, curr_price):
        change = (curr_price - prev_price) / prev_price
        if change > self.threshold:
            return 'up'
        elif change < -self.threshold:
            return 'down'
        else:
            return 'flat'

    def _update_markov(self, symbol):
        prices = self.price_history[symbol]
        if len(prices) < self.lookback+1:
            return False  # Not enough data
        states_seq = [self._get_state(prices[i], prices[i+1]) for i in range(len(prices)-1)]
        for i in range(len(states_seq)-1):
            from_idx = self.state_map[states_seq[i]]
            to_idx = self.state_map[states_seq[i+1]]
            self.transition_counts[symbol][from_idx, to_idx] += 1
        return True

    def _predict_next_state(self, symbol):
        prices = self.price_history[symbol]
        if len(prices) < 2:
            return None
        last_state = self._get_state(prices[-2], prices[-1])
        from_idx = self.state_map[last_state]
        row = self.transition_counts[symbol][from_idx]
        if row.sum() == 0:
            return None
        next_idx = np.argmax(row)
        return self.states[next_idx]

    async def on_price_update(self, symbol, price):
        self.price_history[symbol].append(Decimal(str(price)))
        enough = self._update_markov(symbol)
        if not enough:
            self.last_signal[symbol] = None
            self.logger.debug(f"Not enough data for {symbol}, waiting for more.")
            return None
        next_state = self._predict_next_state(symbol)
        signal = None
        if next_state == 'up':
            signal = 'buy'
            self.last_buy_time[symbol] = time.time()
            self.logger.info(f"{symbol}: Markov predicts UP. Signal: BUY.")
        elif next_state == 'down':
            signal = 'sell'
            self.logger.debug(f"{symbol}: Markov predicts DOWN. Signal: SELL.")
        else:
            signal = 'hold'
            self.logger.debug(f"{symbol}: Markov predicts FLAT. Signal: HOLD.")
        self.last_signal[symbol] = signal
        # Store in history
        if symbol not in self.signal_history:
            from collections import deque
            self.signal_history[symbol] = deque(maxlen=self.signal_history_len)
        self.signal_history[symbol].append((time.time(), signal))
        self.logger.debug(f"{symbol}: Markov signal history: {list(self.signal_history[symbol])}")
        return signal

    def get_last_signal(self, symbol):
        return self.last_signal.get(symbol, None)

    def recent_buy_signal(self, symbol, window_seconds=10):
        """Return True if the last 'buy' signal was within window_seconds (default 10s)."""
        now = time.time()
        last_buy = self.last_buy_time.get(symbol)
        self.logger.debug(f"Checking recent_buy_signal for {symbol}: last_buy={last_buy}, now={now}, window={window_seconds}")
        if last_buy is not None and (now - last_buy) <= window_seconds:
            self.logger.info(f"{symbol}: Markov recent_buy_signal=True (within {window_seconds}s)")
            return True
        self.logger.info(f"{symbol}: Markov recent_buy_signal=False (no recent buy)")
        return False

    def log_state_sequence(self, symbol):
        prices = list(self.price_history[symbol])
        if len(prices) < 2:
            self.logger.info(f"{symbol}: Not enough price history for state sequence.")
            return
        states_seq = [self._get_state(prices[i], prices[i+1]) for i in range(len(prices)-1)]
        self.logger.info(f"{symbol}: Last {len(states_seq)} Markov states: {states_seq}")

    def print_signal_history(self, symbol):
        """Print the last N Markov signals for a symbol for diagnostics."""
        history = self.signal_history.get(symbol, [])
        self.logger.info(f"{symbol}: Last {self.signal_history_len} Markov signals: {history}")
        self.log_state_sequence(symbol)

    # Example integration method
    async def process_ticker_message(self, msg):
        symbol = msg.get('s')
        price = msg.get('c')
        if symbol and price:
            await self.on_price_update(symbol, price)

# Example usage:
# agent = MarkovTradingAgent()
# await agent.on_price_update('BTCUSDT', 30000) 