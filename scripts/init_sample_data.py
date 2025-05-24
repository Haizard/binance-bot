"""
Initialize sample data for the trading bot dashboard.
"""
import sys
from pathlib import Path
import random
from datetime import datetime, timedelta
import logging

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.database import get_database, close_connections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_alerts():
    """Generate sample alert data"""
    alerts = []
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
    alert_types = ["PRICE", "VOLUME", "RSI", "MACD"]
    conditions = ["ABOVE", "BELOW", "CROSSES_ABOVE", "CROSSES_BELOW"]
    now = datetime.utcnow()
    
    # Generate active alerts
    for _ in range(5):
        symbol = random.choice(symbols)
        current_price = random.uniform(100, 30000)
        target_price = current_price * (1 + random.uniform(-0.05, 0.05))
        
        alert = {
            "timestamp": now - timedelta(hours=random.randint(1, 24)),
            "symbol": symbol,
            "type": random.choice(alert_types),
            "condition": random.choice(conditions),
            "target_price": target_price,
            "current_price": current_price,
            "status": "ACTIVE"
        }
        alerts.append(alert)
    
    # Generate triggered alerts
    for _ in range(3):
        symbol = random.choice(symbols)
        target_price = random.uniform(100, 30000)
        triggered_price = target_price * (1 + random.uniform(-0.01, 0.01))
        
        alert = {
            "timestamp": now - timedelta(hours=random.randint(24, 48)),
            "symbol": symbol,
            "type": random.choice(alert_types),
            "condition": random.choice(conditions),
            "target_price": target_price,
            "triggered_price": triggered_price,
            "status": "TRIGGERED"
        }
        alerts.append(alert)
    
    # Generate completed alerts
    for _ in range(7):
        symbol = random.choice(symbols)
        target_price = random.uniform(100, 30000)
        final_price = target_price * (1 + random.uniform(-0.02, 0.02))
        
        alert = {
            "timestamp": now - timedelta(days=random.randint(1, 7)),
            "symbol": symbol,
            "type": random.choice(alert_types),
            "condition": random.choice(conditions),
            "target_price": target_price,
            "final_price": final_price,
            "status": "COMPLETED"
        }
        alerts.append(alert)
    
    return alerts

def generate_sample_trades(num_trades=50):
    """Generate sample trade data"""
    trades = []
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
    now = datetime.utcnow()
    
    for i in range(num_trades):
        symbol = random.choice(symbols)
        side = random.choice(["BUY", "SELL"])
        entry_price = random.uniform(100, 30000)
        price_change = entry_price * random.uniform(-0.05, 0.05)  # -5% to +5%
        exit_price = entry_price + price_change
        
        trade = {
            "timestamp": now - timedelta(minutes=i*15),  # Spread trades over time
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": random.uniform(0.1, 2.0),
            "realized_pnl": price_change * random.uniform(0.1, 2.0),
            "status": "CLOSED"
        }
        trades.append(trade)
    
    return trades

def generate_market_data():
    """Generate sample market data"""
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    now = datetime.utcnow()
    market_data = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            base_price = 30000 if symbol == "BTCUSDT" else 2000
            for i in range(100):  # 100 candles per timeframe
                close = base_price * (1 + random.uniform(-0.02, 0.02))
                open_price = close * (1 + random.uniform(-0.01, 0.01))
                high = max(close, open_price) * (1 + random.uniform(0, 0.01))
                low = min(close, open_price) * (1 - random.uniform(0, 0.01))
                
                candle = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": now - timedelta(minutes=i),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": random.uniform(10, 100)
                }
                market_data.append(candle)
    
    return market_data

def generate_active_trades():
    """Generate sample active trades"""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    active_trades = []
    
    for symbol in symbols:
        if random.random() > 0.5:  # 50% chance of having an active trade
            entry_price = random.uniform(100, 30000)
            active_trades.append({
                "symbol": symbol,
                "side": random.choice(["BUY", "SELL"]),
                "entry_price": entry_price,
                "current_price": entry_price * (1 + random.uniform(-0.02, 0.02)),
                "quantity": random.uniform(0.1, 2.0),
                "timestamp": datetime.utcnow() - timedelta(hours=random.randint(1, 24)),
                "unrealized_pnl": random.uniform(-100, 100),
                "status": "OPEN"
            })
    
    return active_trades

def main():
    """Initialize sample data in the database"""
    try:
        # Get database connection
        db = get_database()
        logger.info("Connected to database")
        
        # Generate and insert sample data
        trades = generate_sample_trades()
        market_data = generate_market_data()
        active_trades = generate_active_trades()
        alerts = generate_sample_alerts()
        
        # Clear existing data
        db.trades.delete_many({})
        db.market_data.delete_many({})
        db.active_trades.delete_many({})
        db.alerts.delete_many({})
        
        # Insert new data
        if trades:
            db.trades.insert_many(trades)
            logger.info(f"Inserted {len(trades)} sample trades")
        
        if market_data:
            db.market_data.insert_many(market_data)
            logger.info(f"Inserted {len(market_data)} market data points")
        
        if active_trades:
            db.active_trades.insert_many(active_trades)
            logger.info(f"Inserted {len(active_trades)} active trades")
            
        if alerts:
            db.alerts.insert_many(alerts)
            logger.info(f"Inserted {len(alerts)} alerts")
        
        logger.info("Sample data initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing sample data: {str(e)}")
        raise
    finally:
        close_connections()

if __name__ == "__main__":
    main() 