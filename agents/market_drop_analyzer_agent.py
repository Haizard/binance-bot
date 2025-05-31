"""
MarketDropAnalyzerAgent: Independent agent that uses WebSocket streams to analyze market data from Binance.
Detects coins that have dropped more than a target percentage in price compared to recent history.
"""
import os
from dotenv import load_dotenv
import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Set
from agents.dip_executor import DipExecutorModule
from binance import AsyncClient, BinanceSocketManager
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import linalg as la
import scipy.stats as stats
from agents.markov_trading_agent import MarkovTradingAgent
import sqlite3
import signal
from pymongo import MongoClient, ASCENDING
import traceback
import warnings
import threading

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for even more detail
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

print("Starting market_drop_analyzer_agent.py")

# --- Trendline Calculation Helper Functions (Copied from trendline_automation.py) ---
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices,
    # return negative val if invalid

    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line
    err = (diffs ** 2.0).sum()
    return err;

def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):

    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y)

    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    # assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases.
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;

            # If increasing by a small amount fails,
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                # raise Exception("Derivative failed. Check your data. ") # Avoid raising exception in bot loop
                return None, None # Indicate failure

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err
            best_slope = test_slope
            get_derivative = True # Recompute derivative

    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])

def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared)
    # coefs[0] = slope,  coefs[1] = intercept
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    if support_coefs is None or resist_coefs is None:
        return None, None # Indicate failure if optimization failed

    return (support_coefs, resist_coefs)
# --- End of Trendline Calculation Helper Functions ---

class MarketDropAnalyzerAgent:
    def __init__(self, drop_threshold=-4.0, profit_target=0.03, risk_percentage=0.01):
        self.name = "MarketDropAnalyzer"
        self.drop_threshold = Decimal(str(drop_threshold))
        self.profit_target = Decimal(str(profit_target))
        self.client = None
        self.bm = None  # BinanceSocketManager
        self.holdings = {}  # symbol -> buy_price
        self.price_data = {}  # symbol -> {'price': Decimal, 'volume': Decimal, 'change': Decimal, 'last_update': datetime}
        self.active_symbols = set()  # Set of symbols we're currently monitoring
        self.technical_confirmations = []  # Empty list to disable all technical confirmations
        # Parameters for RSI-PCA model
        self.pca_evecs = None
        self.pca_means = None
        self.model_coefs = None
        self.long_threshold = None
        self.rsi_periods = list(range(2, 25))
        self.pca_n_components = 3 # Example value, should match training
        # klines lookback for RSI-PCA needs to be at least max(rsi_periods) + lookahead, plus some buffer
        self.rsi_pca_klines_lookback = max(self.rsi_periods) + 6 + 10 # max_rsi_period + lookahead + buffer
        self.model_dir = 'models/rsi_pca'
        self.evecs_path = os.path.join(self.model_dir, 'pca_evecs.npy')
        self.means_path = os.path.join(self.model_dir, 'pca_means.npy')
        self.coefs_path = os.path.join(self.model_dir, 'model_coefs.npy')
        self.l_thresh_path = os.path.join(self.model_dir, 'long_threshold.txt')

        # Parameters for Hawkes model
        self.hawkes_kappa = 0.1 # Example value from hawkes.py
        self.hawkes_lookback = 168 # Example value from hawkes.py
        self.hawkes_norm_lookback = 336 # Example value from hawkes.py (for ATR/norm_range)
        self.hawkes_klines_lookback = max(self.hawkes_lookback, self.hawkes_norm_lookback) + 10 # Need enough data for calculations

        # Parameters for Trendline Breakout model
        self.trendline_lookback = 72 # Example value from trendline_breakout.py
        self.trendline_klines_lookback = self.trendline_lookback + 10 # Need enough data for calculation

        # Parameters for VSA model
        self.vsa_norm_lookback = 168 # Example value from vsa.py
        self.vsa_range_dev_threshold = 1.0 # Example threshold for buy signal (can be tuned)
        self.vsa_klines_lookback = self.vsa_norm_lookback * 2 + 10 # Need enough data for regression and lookback

        # Parameters for Position Sizing by Volatility
        self.risk_percentage = Decimal(str(risk_percentage)) # Percentage of account to risk per trade (e.g., 0.01 for 1%)
        self.atr_lookback = 14 # ATR period
        self.atr_multiplier = 1.0 # Multiplier for ATR to define risk per share/unit
        self.sizing_klines_lookback = self.atr_lookback + 10 # Need enough data for ATR calculation

        self.symbol_queue = asyncio.Queue(maxsize=1000)
        self.processing_symbols = set()
        self.api_semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls to 3
        self.markov_agent = MarkovTradingAgent(lookback=5, states=('up', 'down', 'flat'), threshold=0.0001)
        self.symbol_filters = {}  # symbol -> {'stepSize': ..., 'minQty': ..., 'precision': ...}
        self.kline_cache = {}  # symbol -> interval -> list of klines (most recent last)
        self.kline_cache_limit = 500  # Max klines to keep per symbol/interval

        # MongoDB persistent kline cache
        mongo_url = os.getenv('MONGODB_URL', 'mongodb+srv://haithammisape:hrz123@binance.5hz1tvp.mongodb.net/?retryWrites=true&w=majority&appName=binance')
        self.mongo_client = MongoClient(mongo_url)
        self.mongo_db = self.mongo_client['binance']
        self.klines_collection = self.mongo_db['klines']
        self.klines_collection.create_index([('symbol', ASCENDING), ('interval', ASCENDING), ('open_time', ASCENDING)], unique=True)

    async def batch_fetch_klines(self, symbols, interval, limit, batch_size=1, delay=5.0):
        """Fetch klines for a list of symbols in batches, with a delay between batches to avoid rate limits."""
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [self.fetch_klines(symbol, interval, limit) for symbol in batch]
            await asyncio.gather(*tasks)
            await asyncio.sleep(delay)

    async def setup(self):
        print("Starting setup()")
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
        self.bm = BinanceSocketManager(self.client)
        
        # Get initial exchange info and trading symbols
        exchange_info = await self.client.get_exchange_info()
        self.trading_symbols = {s['symbol'] for s in exchange_info['symbols'] 
                              if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')}
        
        # Get initial 24h ticker data
        tickers = await self.client.get_ticker()
        for ticker in tickers:
            if ticker['symbol'] in self.trading_symbols:
                try:
                    self.price_data[ticker['symbol']] = {
                        'price': Decimal(ticker['lastPrice']),
                        'volume': Decimal(ticker['quoteVolume']),
                        'change': Decimal(ticker['priceChangePercent']),
                        'last_update': datetime.now()
                    }
                except Exception as e:
                    logger.error(f"Error processing ticker for {ticker['symbol']}: {e}")
        
        # Load pre-trained models if specified
        if 'rsi_pca' in self.technical_confirmations:
            await self.load_rsi_pca_model()
        
        # Store stepSize, minQty, precision, and minNotional for each symbol
        for s in exchange_info['symbols']:
            if s['symbol'] in self.trading_symbols:
                filters = {f['filterType']: f for f in s['filters']}
                step_size = Decimal(filters['LOT_SIZE']['stepSize']) if 'LOT_SIZE' in filters else Decimal('0.00000001')
                min_qty = Decimal(filters['LOT_SIZE']['minQty']) if 'LOT_SIZE' in filters else Decimal('0.0')
                min_notional = Decimal(filters['MIN_NOTIONAL']['minNotional']) if 'MIN_NOTIONAL' in filters else Decimal('0.0')
                # Failsafe: if minNotional is missing or zero, set to 5.0
                if min_notional is None or min_notional == 0:
                    min_notional = Decimal('5.0')
                precision = s.get('baseAssetPrecision', 8)
                self.symbol_filters[s['symbol']] = {
                    'stepSize': step_size,
                    'minQty': min_qty,
                    'precision': precision,
                    'minNotional': min_notional
                }
                logger.debug(f"Symbol {s['symbol']} minNotional set to {min_notional}")
        
        # Prefill kline cache and DB for all trading symbols (batch, rate-limit aware)
        await self.batch_fetch_klines(list(self.trading_symbols), AsyncClient.KLINE_INTERVAL_1HOUR, 100)
        
        # Subscribe to kline WebSocket streams for all trading symbols
        self.kline_ws_tasks = []
        for symbol in self.trading_symbols:
            self.kline_ws_tasks.append(asyncio.create_task(self.kline_stream_worker(symbol, AsyncClient.KLINE_INTERVAL_1HOUR)))
        
        logger.info(f"{self.name} setup complete.")
        print("Starting heartbeat thread")
        # Start heartbeat log thread
        def heartbeat():
            while True:
                logger.info("Bot is alive and running...")
                time.sleep(300)  # 5 minutes
        threading.Thread(target=heartbeat, daemon=True).start()

    async def kline_stream_worker(self, symbol, interval):
        """WebSocket worker to keep kline cache and MongoDB DB updated for a symbol/interval."""
        stream = self.bm.kline_socket(symbol=symbol, interval=interval)
        if symbol not in self.kline_cache:
            self.kline_cache[symbol] = {}
        if interval not in self.kline_cache[symbol]:
            self.kline_cache[symbol][interval] = []
        async with stream as s:
            async for msg in s:
                if msg and msg.get('k'):
                    k = msg['k']
                    # Build kline in REST format
                    kline = [
                        k['t'], k['o'], k['h'], k['l'], k['c'], k['v'],
                        k['T'], k['q'], k['n'], k['V'], k['Q'], k['B']
                    ]
                    cache = self.kline_cache[symbol][interval]
                    # If this is a new kline, append; if update, replace last
                    if not cache or k['t'] > cache[-1][0]:
                        cache.append(kline)
                        if len(cache) > self.kline_cache_limit:
                            cache.pop(0)
                        # Insert new kline into MongoDB
                        self.insert_klines(symbol, interval, [kline])
                    elif k['t'] == cache[-1][0]:
                        cache[-1] = kline
                        # Update kline in MongoDB
                        self.insert_klines(symbol, interval, [kline])

    async def start_symbol_stream(self, symbol: str):
        """Start WebSocket stream for a symbol."""
        if symbol in self.active_symbols:
            return
        
        try:
            # Start 24hr ticker stream
            stream = self.bm.ticker_24hr_socket(symbol)
            self.active_symbols.add(symbol)
            
            async with stream as stream:
                while True:
                    msg = await stream.recv()
                    if msg:
                        try:
                            self.price_data[symbol] = {
                                'price': Decimal(msg['c']),  # Last price
                                'volume': Decimal(msg['q']),  # Quote volume
                                'change': Decimal(msg['P']),  # Price change percent
                                'last_update': datetime.now()
                            }
                            await self.check_symbol(symbol)
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error in WebSocket stream for {symbol}: {e}")
            self.active_symbols.remove(symbol)

    async def check_symbol(self, symbol: str):
        """Check a single symbol for trading opportunities, with detailed logs."""
        if symbol not in self.price_data:
            logger.info(f"{symbol} not in price_data, skipping.")
            return
        data = self.price_data[symbol]
        # Skip if data is too old (more than 5 minutes)
        if (datetime.now() - data['last_update']).total_seconds() > 300:
            logger.info(f"{symbol} data too old, skipping.")
            return
        # Skip if we already hold this symbol
        if symbol in self.holdings:
            buy_price = self.holdings[symbol]
            current_price = data['price']
            # Take profit
            if current_price >= buy_price * (1 + self.profit_target):
                logger.info(f"Take profit: Selling {symbol} at {current_price} (bought at {buy_price})")
                await self.place_order(symbol, 'SELL', current_price)
                del self.holdings[symbol]
                return
            # Stop loss
            if current_price <= buy_price * (1 - self.risk_percentage):
                logger.info(f"Stop loss: Selling {symbol} at {current_price} (bought at {buy_price})")
                await self.place_order(symbol, 'SELL', current_price)
                del self.holdings[symbol]
                return
            logger.info(f"{symbol} already held, no sell condition met.")
            return
        # Check for drop
        drop_percent = data['change']
        if drop_percent <= self.drop_threshold:
            logger.info(f"Drop detected: {symbol} dropped {drop_percent:.3f}% (threshold: {self.drop_threshold}%) - considering for trade.")
            # Run filters (e.g., volume, volatility, etc.)
            filters_passed = await self.apply_filters(symbol, data)
            logger.info(f"{symbol} filters_passed: {filters_passed}")
            if not filters_passed:
                logger.info(f"{symbol} skipped: did not pass filters.")
                return
            # Technical confirmation
            confirmed = await self.perform_technical_confirmations(symbol, data['price'])
            logger.info(f"{symbol} technical confirmation: {confirmed}")
            if not confirmed:
                logger.info(f"{symbol} skipped: no technical confirmation.")
                return
            # Pre-check: skip if USDT balance < minNotional
            usdt_balance = await self.get_balance('USDT')
            min_notional = self.symbol_filters.get(symbol, {}).get('minNotional', Decimal('5.0'))
            logger.info(f"[Pre-Check] {symbol} minNotional: {min_notional}, usdt_balance: {usdt_balance}")
            if usdt_balance is None or usdt_balance < min_notional:
                logger.info(f"{symbol} skipped: available USDT balance {usdt_balance} is less than minNotional {min_notional}")
                return
            # Calculate quantity
            try:
                quantity = await self.calculate_quantity(symbol, data['price'])
                logger.info(f"Calculated quantity for {symbol}: {quantity}")
            except Exception as e:
                logger.error(f"Error calculating quantity for {symbol}: {e}")
                return
            if quantity is None or quantity <= 0:
                logger.info(f"{symbol} skipped: calculated quantity is None or <= 0.")
                return
            needed = data['price'] * Decimal(str(quantity))
            logger.info(f"{symbol} USDT balance: {usdt_balance}, needed: {needed}")
            if usdt_balance < needed:
                logger.info(f"{symbol} skipped: insufficient USDT balance. Needed: {needed}, Available: {usdt_balance}")
                return
            # Check notional minimum
            if needed < min_notional:
                logger.info(f"{symbol} skipped: notional value {needed} < minNotional {min_notional}")
                return
            # Place buy order
            try:
                logger.info(f"Placing BUY order for {symbol}: quantity={quantity}, price={data['price']}")
                order = await self.place_order(symbol, 'BUY', data['price'], quantity)
                logger.info(f"Order placed for {symbol}: {order}")
                self.holdings[symbol] = data['price']
            except Exception as e:
                logger.error(f"Error placing BUY order for {symbol}: {e}")
                return

    async def apply_filters(self, symbol: str, data: dict) -> bool:
        """Apply all filters to a symbol, with detailed logs."""
        # Volume filter (relaxed for testing)
        if data['volume'] < Decimal('10000'):
            logger.info(f"{symbol} failed volume filter: {data['volume']} < 10000")
            return False
        logger.info(f"{symbol} passed volume filter: {data['volume']} >= 10000")
        # Price filter (relaxed for testing)
        if data['price'] < Decimal('0.001'):
            logger.info(f"{symbol} failed price filter: {data['price']} < 0.001")
            return False
        logger.info(f"{symbol} passed price filter: {data['price']} >= 0.001")
        # Cooldown filter
        if hasattr(self, 'last_sell_time') and symbol in self.last_sell_time:
            if time.time() - self.last_sell_time[symbol] < 24 * 60 * 60:
                logger.info(f"{symbol} failed cooldown filter: sold within last 24h")
                return False
        logger.info(f"{symbol} passed cooldown filter")
        # Market condition filter
        btc_data = self.price_data.get('BTCUSDT', {})
        eth_data = self.price_data.get('ETHUSDT', {})
        if btc_data and btc_data.get('change', Decimal('0')) <= Decimal('-4'):
            logger.info(f"{symbol} failed market filter: BTCUSDT change {btc_data.get('change')} <= -4")
            return False
        if eth_data and eth_data.get('change', Decimal('0')) <= Decimal('-4'):
            logger.info(f"{symbol} failed market filter: ETHUSDT change {eth_data.get('change')} <= -4")
            return False
        logger.info(f"{symbol} passed market filter")
        return True

    async def run(self):
        """Main run loop that manages a single WebSocket stream for all tickers and multiple background workers."""
        try:
            if not self.bm:
                self.bm = BinanceSocketManager(self.client)
            logger.info("Starting 10 background workers...")
            worker_tasks = [asyncio.create_task(self.symbol_worker(i)) for i in range(10)]
            logger.info("Starting WebSocket ticker stream...")
            stream = self.bm.ticker_socket()
            async with stream as tscm:
                while True:
                    try:
                        msg = await tscm.recv()
                        logger.debug("Received WebSocket message.")
                        if msg:
                            await self._enqueue_ticker_message(msg)
                    except Exception as e:
                        logger.error(f"Error processing ticker stream: {e}")
                        break
            for task in worker_tasks:
                await task
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(5)
            await self.run()

    async def _enqueue_ticker_message(self, msg):
        """Enqueue USDT symbols from ticker messages for background processing."""
        try:
            if isinstance(msg, list):
                for ticker in msg:
                    await self._enqueue_ticker_message(ticker)
                return
            symbol = msg.get('s')
            if not symbol or not symbol.endswith('USDT'):
                return
            self.price_data[symbol] = {
                'price': Decimal(msg['c']),
                'volume': Decimal(msg['q']),
                'change': Decimal(msg['P']),
                'last_update': datetime.now()
            }
            # Pass price update to Markov agent
            await self.markov_agent.on_price_update(symbol, msg['c'])
            if symbol not in self.processing_symbols:
                try:
                    self.processing_symbols.add(symbol)
                    self.symbol_queue.put_nowait(symbol)
                    logger.debug(f"Enqueued symbol: {symbol}")
                except asyncio.QueueFull:
                    logger.warning(f"Symbol queue full, dropping symbol: {symbol}")
        except Exception as e:
            logger.error(f"Error enqueuing ticker message: {e}")

    async def symbol_worker(self, worker_id=0):
        """Background worker to process symbols from the queue."""
        logger.info(f"Worker {worker_id} started.")
        while True:
            try:
                symbol = await asyncio.wait_for(self.symbol_queue.get(), timeout=30)
            except asyncio.TimeoutError:
                logger.info(f"Worker {worker_id}: Queue empty for 30 seconds.")
                continue
            try:
                logger.info(f"Worker {worker_id} processing symbol: {symbol}")
                start_time = time.time()
                async with self.api_semaphore:
                    await self.check_symbol(symbol)
                elapsed = time.time() - start_time
                logger.info(f"Worker {worker_id} finished {symbol} in {elapsed:.2f} seconds.")
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
            finally:
                self.processing_symbols.discard(symbol)
                self.symbol_queue.task_done()

    async def load_rsi_pca_model(self):
        """
        Loads pre-trained RSI-PCA model parameters from files.
        """
        try:
            self.pca_evecs = np.load(self.evecs_path)
            self.pca_means = np.load(self.means_path)
            self.model_coefs = np.load(self.coefs_path)
            with open(self.l_thresh_path, 'r') as f:
                self.long_threshold = float(f.read())
            logger.info("RSI-PCA model parameters loaded successfully.")
        except FileNotFoundError:
            logger.warning("RSI-PCA model parameters not found. Please run scripts/train_rsi_pca_model.py to train the model.")
        except Exception as e:
            logger.error(f"Error loading RSI-PCA model parameters: {e}")

    async def perform_technical_confirmations(self, symbol: str, current_price: Decimal) -> bool:
        """
        Performs the configured technical confirmations.
        Returns True if at least one configured indicator provides a buy signal.
        """
        if not self.technical_confirmations:
            return True

        for confirmation_method in self.technical_confirmations:
            if confirmation_method == 'rsi_pca':
                if await self.check_rsi_pca_buy_signal(symbol, current_price):
                    return True
            elif confirmation_method == 'hawkes':
                if await self.check_hawkes_buy_signal(symbol):
                    return True
            elif confirmation_method == 'trendline':
                if await self.check_trendline_buy_signal(symbol):
                    return True
            elif confirmation_method == 'vsa':
                if await self.check_vsa_buy_signal(symbol):
                    return True
            elif confirmation_method == 'markov':
                # Markov confirmation: confirm if current signal is 'buy'
                if self.markov_agent.get_last_signal(symbol) == 'buy':
                    logger.info(f"Markov agent confirms buy for {symbol} (current signal is 'buy').")
                    if symbol in ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT']:
                        self.markov_agent.print_signal_history(symbol)
                    return True
                else:
                    if symbol in ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT']:
                        self.markov_agent.print_signal_history(symbol)
            else:
                logger.warning(f"Unknown technical confirmation method configured: {confirmation_method}")
        # If Markov is the only confirmation and no trade is placed, log debug
        if self.technical_confirmations == ['markov']:
            logger.debug(f"No Markov confirmation for {symbol} at price {current_price}.")
        return False

    async def check_rsi_pca_buy_signal(self, symbol: str, current_price: Decimal) -> bool:
        """
        Checks for a buy signal from the RSI-PCA indicator for a given symbol.
        Requires pre-calculated PCA components, means, model coefficients, and long threshold.
        """
        if self.pca_evecs is None or self.pca_means is None or self.model_coefs is None or self.long_threshold is None:
            logger.warning("RSI-PCA model parameters not loaded. Skipping technical confirmation.")
            return False

        try:
            # 1. Fetch recent klines data
            # Fetch enough data for the max RSI period and PCA transformation
            # Calculate start time based on required klines lookback
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.rsi_pca_klines_lookback)

            klines = await self.client.get_historical_klines(
                symbol,
                AsyncClient.KLINE_INTERVAL_1HOUR,
                start_time.strftime('%d %b, %Y %H:%M:%S'),
                end_time.strftime('%d %b, %Y %H:%M:%S')
            )

            # Ensure we have enough data points
            if len(klines) < self.rsi_pca_klines_lookback:
                # Suppressed: Not enough klines data for RSI-PCA
                return False

            # Convert klines to DataFrame
            # Index 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', \
                                               'close_time', 'quote_asset_volume', 'number_of_trades', \
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = df['close'].astype(float)
            # Set datetime index (needed for pandas_ta and aligning with model training)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')

            # 2. Calculate RSI for multiple periods
            rsis_df = pd.DataFrame()
            for p in self.rsi_periods:
                rsis_df[p] = ta.rsi(df['close'], length=p)

            # Drop rows with NaN created by RSI calculation
            rsis_df = rsis_df.dropna()

            if rsis_df.empty:
                # Suppressed: RSI calculation resulted in empty DataFrame
                return False

            # Get the latest row of RSI data
            # Ensure the index of latest_rsi_row aligns with the expected structure for centering
            latest_rsi_row = rsis_df.iloc[-1]

            # Ensure the order of features in latest_rsi_row matches the training data
            # Assuming rsis_df columns are in the same order as during training (2, 3, ..., 24)
            # If not, reindex latest_rsi_row according to the columns used during training.
            # For this implementation, we assume the order is consistent.

            # 3. Apply PCA transformation (using pre-calculated means and evecs)
            # Center the data using pre-calculated means
            centered_rsi_row = latest_rsi_row - self.pca_means

            # Apply PCA transformation (matrix multiplication with evecs)
            # Need to select the top n_components eigenvectors
            # Ensure dimensionality matches: centered_rsi_row (1, 23), self.pca_evecs (23, 23), result (1, 3)
            pca_transformed_row = np.dot(centered_rsi_row.values, self.pca_evecs[:, :self.pca_n_components])

            # 4. Make a prediction using pre-calculated model coefficients
            # Ensure dimensionality matches: pca_transformed_row (1, 3), self.model_coefs (3,), result (1,)
            prediction = np.dot(pca_transformed_row, self.model_coefs)

            # 5. Compare prediction to the long threshold
            # The prediction is a numpy array with one element, extract the scalar value
            if prediction[0] > self.long_threshold:
                logger.info(f"RSI-PCA buy signal confirmed for {symbol}. Prediction: {prediction[0]:.4f}, Threshold: {self.long_threshold:.4f}")
                return True
            else:
                logger.info(f"RSI-PCA buy signal not confirmed for {symbol}. Prediction: {prediction[0]:.4f}, Threshold: {self.long_threshold:.4f}")
                return False

        except Exception as e:
            logger.error(f"Error checking RSI-PCA signal for {symbol}: {e}")
            return False # Return False in case of error

    async def fetch_klines_with_retry(self, symbol, interval, limit, retries=3, delay=2):
        for attempt in range(retries):
            try:
                await asyncio.sleep(0.3)  # Throttle REST requests (300ms delay)
                return await self.client.get_historical_klines(symbol, interval, limit=limit)
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout fetching klines for {symbol}, attempt {attempt+1}/{retries}")
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            except Exception as e:
                logger.error(f"Error fetching klines for {symbol} (attempt {attempt+1}/{retries}): {e}\n{traceback.format_exc()}")
                break
        logger.error(f"Failed to fetch klines for {symbol} after {retries} retries.")
        return None

    async def fetch_klines(self, symbol: str, interval: str, limit: int):
        """
        Fetches recent klines data for a symbol with a specified limit, using MongoDB cache and WebSocket updates.
        Adds a delay to avoid hitting REST rate limits.
        """
        # Try to use MongoDB cache first
        cached_klines = self.fetch_recent_klines_from_db(symbol, interval, limit)
        if len(cached_klines) >= limit:
            return cached_klines[-limit:]
        # If not enough cached, fetch missing from REST and update cache
        klines = await self.fetch_klines_with_retry(symbol, interval, limit, retries=3, delay=2)
        if klines:
            self.insert_klines(symbol, interval, klines)
            return klines[-limit:]
        return None

    async def check_hawkes_buy_signal(self, symbol: str) -> bool:
        """
        Checks for a buy signal from the Hawkes volatility indicator.
        """
        try:
            # Fetch klines data - need enough for ATR, normalized range, and Hawkes process
            klines = await self.fetch_klines(symbol, AsyncClient.KLINE_INTERVAL_1HOUR, self.hawkes_klines_lookback)
            if not klines or len(klines) < self.hawkes_klines_lookback:
                # Suppressed: Not enough klines data for Hawkes indicator
                return False

            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', \
                                               'close_time', 'quote_asset_volume', 'number_of_trades', \
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)

            # Calculate ATR
            df['atr'] = ta.atr(np.log(df['high']), np.log(df['low']), np.log(df['close']), self.hawkes_norm_lookback)

            # Calculate Normalized Range
            df['norm_range'] = (np.log(df['high']) - np.log(df['low'])) / df['atr']

            # Calculate Hawkes Process on Normalized Range (adapting from hawkes.py)
            # Note: The original hawkes_process function takes a Series and kappa
            # We need to apply this to the 'norm_range' column
            alpha = np.exp(-self.hawkes_kappa)
            norm_range_arr = df['norm_range'].fillna(0).to_numpy() # Fill NaN for calculation
            hawkes_values = np.zeros(len(df))
            hawkes_values[0] = norm_range_arr[0] if not np.isnan(norm_range_arr[0]) else 0
            for i in range(1, len(hawkes_values)):
                 if np.isnan(norm_range_arr[i]): # Handle potential NaNs in input
                     hawkes_values[i] = hawkes_values[i - 1] * alpha # Decay without new event
                 else:
                    hawkes_values[i] = hawkes_values[i - 1] * alpha + norm_range_arr[i]
            df['v_hawk'] = pd.Series(hawkes_values * self.hawkes_kappa, index=df.index)

            # Calculate Rolling Quantiles (adapting from hawkes.py vol_signal)
            q05 = df['v_hawk'].rolling(self.hawkes_lookback).quantile(0.05)
            q95 = df['v_hawk'].rolling(self.hawkes_lookback).quantile(0.95)

            # Determine buy signal (adapting from hawkes.py vol_signal)
            # A buy signal occurs when v_hawk crosses above the 95th percentile
            # and the price has increased since the last time v_hawk was below the 5th percentile.
            # This requires tracking state (last_below_q05_index).
            # For simplicity in real-time check, we will look for a cross above 95th percentile
            # and assume the increasing price condition is handled by the main bot's price drop check.
            latest_v_hawk = df['v_hawk'].iloc[-1]
            latest_q95 = q95.iloc[-1]

            if latest_v_hawk > latest_q95:
                 logger.info(f"Hawkes buy signal confirmed for {symbol}. v_hawk: {latest_v_hawk:.4f}, Q95: {latest_q95:.4f}")
                 return True
            else:
                 logger.info(f"Hawkes buy signal not confirmed for {symbol}. v_hawk: {latest_v_hawk:.4f}, Q95: {latest_q95:.4f}")
                 return False

        except Exception as e:
            logger.error(f"Error checking Hawkes signal for {symbol}: {e}")
            return False # Return False in case of error

    async def check_trendline_buy_signal(self, symbol: str) -> bool:
        """
        Checks for a buy signal from the Trendline Breakout indicator.
        A buy signal occurs when the current price breaks above the resistance trendline.
        """
        # Note: The original trendline_breakout.py depends on trendline_automation.py's fit_trendlines_single.
        # We have integrated that logic by copying the functions.

        try:
            # Fetch klines data - need enough for trendline calculation
            # Using close prices for single trendline calculation as in original trendline_automation.py example
            klines = await self.fetch_klines(symbol, AsyncClient.KLINE_INTERVAL_1HOUR, self.trendline_klines_lookback)
            if not klines or len(klines) < self.trendline_klines_lookback:
                # Suppressed: Not enough klines data for Trendline indicator
                return False

            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', \
                                               'close_time', 'quote_asset_volume', 'number_of_trades', \
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            # Ensure close prices are float and convert to numpy array for trendline calculation
            close_prices = df['close'].astype(float).to_numpy()

            # Calculate trendlines using the imported function
            support_coefs, resist_coefs = fit_trendlines_single(close_prices)

            if support_coefs is None or resist_coefs is None:
                 logger.warning(f"Trendline calculation failed for {symbol}.")
                 return False # Indicate failure

            # Extract resistance line parameters (slope and intercept)
            resist_slope, resist_intercept = resist_coefs

            # Calculate the expected resistance line value at the current point
            # The trendline is calculated on an array starting from index 0.
            # The current point corresponds to the last element in the close_prices array,
            # which has an index of len(close_prices) - 1.
            current_index = len(close_prices) - 1
            expected_resistance = resist_slope * current_index + resist_intercept

            # Get the latest actual close price
            latest_close = close_prices[-1]

            # Determine buy signal: latest close breaks above resistance
            if latest_close > expected_resistance:
                 logger.info(f"Trendline Breakout buy signal confirmed for {symbol}. Close: {latest_close:.4f}, Resistance: {expected_resistance:.4f}")
                 return True
            else:
                 logger.info(f"Trendline Breakout buy signal not confirmed for {symbol}. Close: {latest_close:.4f}, Resistance: {expected_resistance:.4f}")
                 return False

        except Exception as e:
            logger.error(f"Error checking Trendline Breakout signal for {symbol}: {e}")
            return False # Return False in case of error

    async def check_vsa_buy_signal(self, symbol: str) -> bool:
        """
        Checks for a buy signal from the VSA indicator.
        A buy signal is indicated by a significantly positive range_dev.
        """
        try:
            # Fetch klines data - need enough for VSA calculation (ATR, volume median, regression)
            klines = await self.fetch_klines(symbol, AsyncClient.KLINE_INTERVAL_1HOUR, self.vsa_klines_lookback)
            if not klines or len(klines) < self.vsa_klines_lookback:
                # Suppressed: Not enough klines data for VSA indicator
                return False

            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', \
                                               'close_time', 'quote_asset_volume', 'number_of_trades', \
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Calculate VSA indicator (adapting from vsa.py)
            # Calculate ATR and Volume Median
            atr = ta.atr(df['high'], df['low'], df['close'], self.vsa_norm_lookback)
            vol_med = df['volume'].rolling(self.vsa_norm_lookback).median()

            df['norm_range'] = (df['high'] - df['low']) / atr
            df['norm_volume'] = df['volume'] / vol_med

            # Perform linear regression and calculate range_dev
            # Need enough data points for the regression window (vsa_norm_lookback)
            range_dev = np.zeros(len(df))
            range_dev[:] = np.nan

            # Ensure we have enough data before starting the loop for regression window
            start_idx = self.vsa_norm_lookback * 2 # Regression needs window, which needs prior data

            if len(df) > start_idx:
                for i in range(start_idx, len(df)):
                    window = df.iloc[i - self.vsa_norm_lookback + 1: i + 1].copy() # Regression window
                    # Ensure window has enough data for regression
                    if len(window) == self.vsa_norm_lookback:
                         # Handle potential inf/nan values before regression
                         window_cleaned = window.replace([np.inf, -np.inf], np.nan).dropna()

                         if len(window_cleaned) > 1 and window_cleaned['norm_volume'].nunique() > 1: # Need at least 2 unique points for regression
                             try:
                                 slope, intercept, r_val, _, _ = stats.linregress(window_cleaned['norm_volume'], window_cleaned['norm_range'])

                                 if slope > 0.0 and r_val >= 0.2: # Check conditions from vsa.py
                                     latest_norm_vol = df['norm_volume'].iloc[i]
                                     # Handle potential inf/nan in latest_norm_vol
                                     if not np.isinf(latest_norm_vol) and not np.isnan(latest_norm_vol):
                                          pred_range = intercept + slope * latest_norm_vol
                                          range_dev[i] = df['norm_range'].iloc[i] - pred_range
                                     else:
                                          range_dev[i] = 0.0 # Treat as no signal if input is invalid
                                 else:
                                     range_dev[i] = 0.0 # No signal if regression conditions not met
                             except ValueError: # Handle cases where linregress might fail (e.g., all y values are same)
                                 range_dev[i] = 0.0
                         else:
                             range_dev[i] = 0.0 # Not enough valid data for regression window
                    else:
                         range_dev[i] = 0.0 # Window size not as expected
            else:
                # Suppressed: Not enough data points to start VSA regression
                return False # Not enough data to even start regression

            df['range_dev'] = pd.Series(range_dev, index=df.index)

            # Determine buy signal: significantly positive range_dev above a threshold
            latest_range_dev = df['range_dev'].iloc[-1] if not df['range_dev'].empty else np.nan

            if not np.isnan(latest_range_dev) and latest_range_dev > self.vsa_range_dev_threshold:
                 logger.info(f"VSA buy signal confirmed for {symbol}. range_dev: {latest_range_dev:.4f}, Threshold: {self.vsa_range_dev_threshold:.4f}")
                 return True
            else:
                 logger.info(f"VSA buy signal not confirmed for {symbol}. range_dev: {latest_range_dev:.4f}, Threshold: {self.vsa_range_dev_threshold:.4f}")
                 return False

        except Exception as e:
            logger.error(f"Error checking VSA signal for {symbol}: {e}")
            return False # Return False in case of error

    def round_quantity(self, symbol, qty):
        """Round quantity to allowed step size and precision for the symbol."""
        try:
            f = self.symbol_filters.get(symbol)
            if not f:
                return float(Decimal(str(qty)))
            
            # Convert all values to Decimal
            qty = Decimal(str(qty))
            step = Decimal(str(f['stepSize']))
            min_qty = Decimal(str(f['minQty']))
            
            # Floor to nearest step size
            rounded = (qty // step) * step
            # Round to allowed decimal places
            rounded = rounded.quantize(step)
            # Also ensure not below minQty
            if rounded < min_qty:
                return float(min_qty)
            return float(rounded)
        except Exception as e:
            logger.error(f"Error in round_quantity for {symbol}: {e}")
            return float(qty)  # Return original quantity if rounding fails

    async def calculate_quantity(self, symbol, price):
        """
        Calculates the quantity to buy using a fixed margin of $15 per position by default.
        Only allows opening a new position if the number of open positions is less than the allowed (max positions - reserve).
        Leaves 2 margins as a reserve (e.g., if 6 possible, only use 4).
        """
        try:
            price = Decimal(str(price))
            usdt_balance = await self.get_balance('USDT')
            usdt_balance = Decimal(str(usdt_balance))
            if usdt_balance <= 0:
                logger.warning(f"Insufficient USDT balance ({usdt_balance}) for position sizing.")
                return 0.0
            margin = Decimal('15')
            reserve_count = 2
            max_positions = int(usdt_balance // margin)
            usable_positions = max(max_positions - reserve_count, 0)
            open_positions = len(self.holdings)
            if open_positions >= usable_positions or usable_positions == 0:
                logger.info(f"Position limit reached or no usable positions (open: {open_positions}, usable: {usable_positions}). No new position will be opened.")
                return 0.0
            # Only use margin if enough for at least one more position
            if usdt_balance < margin:
                position_size = usdt_balance
            else:
                position_size = margin
            if price <= 0:
                logger.warning(f"Current price for {symbol} is non-positive: {price}.")
                return 0.0
            quantity = position_size / price
            quantity_rounded = Decimal(str(self.round_quantity(symbol, quantity)))
            logger.info(f"[Margin] Rounded quantity for {symbol}: {quantity_rounded} (stepSize: {self.symbol_filters.get(symbol, {}).get('stepSize', 'N/A')})")
            return float(quantity_rounded)
        except Exception as e:
            logger.error(f"Error calculating quantity for {symbol}: {e}")
            return 0.0

    async def place_order(self, symbol, side, price, quantity=None):
        """Unified order placement method for BUY and SELL."""
        if side == 'BUY':
            return await self.buy(symbol, quantity, price)
        elif side == 'SELL':
            return await self.sell(symbol, quantity, price)
        else:
            raise ValueError(f"Unknown order side: {side}")

    async def buy(self, symbol, qty, price):
        # Place a market buy order (real trading, be careful!)
        try:
            order = await self.client.create_order(symbol=symbol, side='BUY', type='MARKET', quantity=qty)
            self.holdings[symbol] = price
            logger.info(f"Bought {symbol} at {price} (qty: {qty}) | Order: {order}")
            return order
        except Exception as e:
            logger.error(f"Buy failed for {symbol}: {e}")
            return None

    async def sell(self, symbol, qty, price):
        # Place a market sell order (real trading, be careful!)
        try:
            order = await self.client.create_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty)
            del self.holdings[symbol]
            logger.info(f"Sold {symbol} at {price} (qty: {qty}) | Order: {order}")
            return order
        except Exception as e:
            logger.error(f"Sell failed for {symbol}: {e}")
            return None

    async def get_balance(self, asset):
        """
        Fetches the available balance for a given asset from the Binance account.
        """
        try:
            balance = await self.client.get_asset_balance(asset=asset)
            # Use 'free' balance which is available for trading
            available_balance = Decimal(balance['free'])
            logger.info(f"Available balance for {asset}: {available_balance}")
            return available_balance
        except Exception as e:
            logger.error(f"Error fetching balance for {asset}: {e}")
            return Decimal('0.0') # Return 0 if balance cannot be fetched

    async def cleanup(self):
        """Cleanup resources."""
        if self.bm:
            await self.bm.close()
        if self.client:
            await self.client.close_connection()
        logger.info(f"{self.name} cleaned up.")

    def insert_klines(self, symbol, interval, klines):
        """Insert multiple klines into MongoDB."""
        for k in klines:
            doc = {
                'symbol': symbol,
                'interval': interval,
                'open_time': int(k[0]),
                'open': k[1],
                'high': k[2],
                'low': k[3],
                'close': k[4],
                'volume': k[5],
                'close_time': int(k[6]),
                'quote_asset_volume': k[7],
                'number_of_trades': int(k[8]),
                'taker_buy_base_asset_volume': k[9],
                'taker_buy_quote_asset_volume': k[10],
                'ignore': k[11]
            }
            self.klines_collection.update_one(
                {'symbol': symbol, 'interval': interval, 'open_time': doc['open_time']},
                {'$set': doc},
                upsert=True
            )

    def fetch_recent_klines_from_db(self, symbol, interval, limit):
        """Fetch the most recent N klines for a symbol/interval from MongoDB."""
        cursor = self.klines_collection.find({'symbol': symbol, 'interval': interval}).sort('open_time', -1).limit(limit)
        rows = list(cursor)[::-1]  # Return in ascending order
        return [
            [
                row['open_time'], row['open'], row['high'], row['low'], row['close'], row['volume'],
                row['close_time'], row['quote_asset_volume'], row['number_of_trades'],
                row['taker_buy_base_asset_volume'], row['taker_buy_quote_asset_volume'], row['ignore']
            ]
            for row in rows
        ]

async def main():
    agent = MarketDropAnalyzerAgent()
    await agent.setup()
    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    def _signal_handler():
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # add_signal_handler may not be implemented on Windows
            pass
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
