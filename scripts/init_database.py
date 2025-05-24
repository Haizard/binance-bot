"""
Initialize the database with sample data for testing.
"""
import sys
import os
import logging
from datetime import datetime, timedelta
import random
from decimal import Decimal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import (
    get_database, init_collections, update_market_data,
    insert_trade, update_account_info
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_market_data(symbol: str, timeframe: str, days: int = 30):
    """Generate sample market data"""
    data = []
    base_price = 50000 if symbol == "BTCUSDT" else 3000  # Base price for BTC/ETH
    timestamp = datetime.utcnow() - timedelta(days=days)
    
    for _ in range(days * 24 * 60):  # 1-minute candles for X days
        open_price = base_price * (1 + random.uniform(-0.001, 0.001))
        close_price = open_price * (1 + random.uniform(-0.002, 0.002))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.001))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.001))
        volume = random.uniform(1, 100)
        
        candle = {
            'timestamp': timestamp,
            'open': float(Decimal(str(open_price)).quantize(Decimal('0.01'))),
            'high': float(Decimal(str(high_price)).quantize(Decimal('0.01'))),
            'low': float(Decimal(str(low_price)).quantize(Decimal('0.01'))),
            'close': float(Decimal(str(close_price)).quantize(Decimal('0.01'))),
            'volume': float(Decimal(str(volume)).quantize(Decimal('0.01')))
        }
        data.append(candle)
        
        timestamp += timedelta(minutes=1)
        base_price = close_price
    
    return data

def generate_sample_trades(symbol: str, days: int = 30):
    """Generate sample trades"""
    trades = []
    timestamp = datetime.utcnow() - timedelta(days=days)
    
    for _ in range(100):  # Generate 100 sample trades
        entry_price = 50000 * (1 + random.uniform(-0.1, 0.1)) if symbol == "BTCUSDT" else 3000 * (1 + random.uniform(-0.1, 0.1))
        volume = random.uniform(0.1, 1.0)
        profit_loss = random.uniform(-100, 200)
        
        trade = {
            'symbol': symbol,
            'timestamp': timestamp + timedelta(minutes=random.randint(1, days * 24 * 60)),
            'side': random.choice(['BUY', 'SELL']),
            'entry_price': float(Decimal(str(entry_price)).quantize(Decimal('0.01'))),
            'volume': float(Decimal(str(volume)).quantize(Decimal('0.001'))),
            'realized_pnl': float(Decimal(str(profit_loss)).quantize(Decimal('0.01'))),
            'status': random.choice(['active', 'closed']),
            'strategy': random.choice(['TREND_FOLLOWING', 'MEAN_REVERSION', 'BREAKOUT'])
        }
        trades.append(trade)
    
    return sorted(trades, key=lambda x: x['timestamp'])

def initialize_sample_data():
    """Initialize database with sample data"""
    try:
        logger.info("Initializing database collections...")
        init_collections()
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Generate and insert market data
        logger.info("Generating market data...")
        for symbol in symbols:
            for timeframe in timeframes:
                market_data = generate_sample_market_data(symbol, timeframe)
                for data in market_data:
                    update_market_data(symbol, timeframe, data)
                logger.info(f"Inserted market data for {symbol} {timeframe}")
        
        # Generate and insert trades
        logger.info("Generating trades...")
        for symbol in symbols:
            trades = generate_sample_trades(symbol)
            for trade in trades:
                insert_trade(trade)
            logger.info(f"Inserted trades for {symbol}")
        
        # Insert account information
        logger.info("Inserting account information...")
        account_info = {
            'balance': 10000.00,
            'equity': 12000.00,
            'used_margin': 2000.00,
            'free_margin': 8000.00,
            'margin_level': 600.00,
            'timestamp': datetime.utcnow()
        }
        update_account_info(account_info)
        
        logger.info("Sample data initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing sample data: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    initialize_sample_data() 