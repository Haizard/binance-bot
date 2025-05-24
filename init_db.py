from datetime import datetime, timedelta
from config.database import get_database, init_collections
import random

def generate_sample_trades(num_trades=50):
    """Generate sample trade data"""
    trades = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_trades):
        profit_loss = random.uniform(-100, 200)
        trade_time = base_time + timedelta(hours=i*4)
        
        trade = {
            'pair': random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
            'type': random.choice(['long', 'short']),
            'entry_price': random.uniform(25000, 30000),
            'exit_price': random.uniform(25000, 30000),
            'volume': random.uniform(0.1, 1.0),
            'pl': profit_loss,
            'status': random.choice(['active', 'closed']),
            'timestamp': trade_time
        }
        trades.append(trade)
    
    return trades

def generate_sample_alerts(num_alerts=20):
    """Generate sample alert data"""
    alerts = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(num_alerts):
        alert_time = base_time + timedelta(hours=i*8)
        
        alert = {
            'symbol': random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
            'type': random.choice(['price', 'volume', 'indicator']),
            'condition': random.choice(['above', 'below', 'crosses']),
            'value': random.uniform(25000, 30000),
            'status': random.choice(['active', 'triggered', 'pending']),
            'timestamp': alert_time
        }
        alerts.append(alert)
    
    return alerts

def generate_sample_strategies(num_strategies=5):
    """Generate sample strategy data"""
    strategies = []
    
    for i in range(num_strategies):
        strategy = {
            'name': f'Strategy {i+1}',
            'type': random.choice(['trend', 'momentum', 'mean_reversion']),
            'pairs': random.sample(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'], 2),
            'performance': random.uniform(-10, 20),
            'status': random.choice(['active', 'testing', 'stopped']),
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
            'risk': {
                'stop_loss': random.uniform(1, 5),
                'take_profit': random.uniform(2, 10),
                'max_position': random.uniform(10, 50)
            }
        }
        strategies.append(strategy)
    
    return strategies

def generate_sample_market_data(num_points=100):
    """Generate sample market data"""
    market_data = []
    base_time = datetime.now() - timedelta(days=7)
    base_price = 27000
    
    for i in range(num_points):
        price_change = random.uniform(-500, 500)
        current_price = base_price + price_change
        
        data = {
            'symbol': 'BTC/USDT',
            'price': current_price,
            'high': current_price + random.uniform(10, 100),
            'low': current_price - random.uniform(10, 100),
            'volume': random.uniform(100, 1000),
            'change': (price_change / base_price) * 100,
            'timestamp': base_time + timedelta(hours=i)
        }
        market_data.append(data)
    
    return market_data

def initialize_sample_data():
    """Initialize database with sample data"""
    try:
        # Initialize collections first
        init_collections()
        
        db = get_database()
        
        # Insert sample trades
        trades = generate_sample_trades()
        db.trades.insert_many(trades)
        
        # Insert sample alerts
        alerts = generate_sample_alerts()
        db.alerts.insert_many(alerts)
        
        # Insert sample strategies
        strategies = generate_sample_strategies()
        db.strategies.insert_many(strategies)
        
        # Insert sample market data
        market_data = generate_sample_market_data()
        db.market_data.insert_many(market_data)
        
        # Insert sample account info
        account_info = {
            'balance': 10000.00,
            'equity': 12000.00,
            'used_margin': 2000.00,
            'timestamp': datetime.now()
        }
        db.account.insert_one(account_info)
        
        print("Sample data initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing sample data: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_sample_data() 