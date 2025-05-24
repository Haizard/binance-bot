import os
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = "mongodb+srv://haithammisape:hrz123@algobot.pzhq7bm.mongodb.net/?retryWrites=true&w=majority&appName=algobot"

# Collection names
COLLECTIONS = {
    'trades': 'trades',              # Store all trades (active and historical)
    'performance': 'performance',    # Store performance metrics
    'alerts': 'alerts',             # Store trading alerts
    'strategies': 'strategies',      # Store trading strategies
    'market_data': 'market_data',   # Store market data
    'account': 'account'            # Store account information
}

def get_database():
    """Get MongoDB client"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client.trading_bot
        return db
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

async def get_async_database():
    """Get asynchronous MongoDB client"""
    try:
        client = AsyncIOMotorClient(MONGODB_URI)
        db = client.trading_bot
        return db
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def init_collections():
    """Initialize database collections with proper indexes"""
    try:
        db = get_database()
        
        # Trades collection
        if 'trades' not in db.list_collection_names():
            trades_col = db.create_collection('trades')
            trades_col.create_index([('timestamp', -1)])
            trades_col.create_index([('status', 1)])
            trades_col.create_index([('pair', 1)])
        
        # Performance collection
        if 'performance' not in db.list_collection_names():
            perf_col = db.create_collection('performance')
            perf_col.create_index([('timestamp', -1)])
            perf_col.create_index([('metric', 1)])
        
        # Alerts collection
        if 'alerts' not in db.list_collection_names():
            alerts_col = db.create_collection('alerts')
            alerts_col.create_index([('timestamp', -1)])
            alerts_col.create_index([('status', 1)])
        
        # Strategies collection
        if 'strategies' not in db.list_collection_names():
            strat_col = db.create_collection('strategies')
            strat_col.create_index([('name', 1)], unique=True)
        
        # Market data collection
        if 'market_data' not in db.list_collection_names():
            market_col = db.create_collection('market_data')
            market_col.create_index([('timestamp', -1)])
            market_col.create_index([('pair', 1)])
        
        # Account collection
        if 'account' not in db.list_collection_names():
            account_col = db.create_collection('account')
            account_col.create_index([('timestamp', -1)])
        
        logger.info("Database collections initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing collections: {str(e)}")
        raise

def insert_trade(trade_data):
    """Insert a new trade"""
    try:
        db = get_database()
        trade_data['timestamp'] = datetime.utcnow()
        result = db.trades.insert_one(trade_data)
        return result.inserted_id
    except Exception as e:
        logger.error(f"Error inserting trade: {str(e)}")
        raise

def get_active_trades():
    """Get all active trades"""
    try:
        db = get_database()
        return list(db.trades.find({'status': 'active'}))
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}")
        raise

def get_trade_history(limit=50):
    """Get trade history"""
    try:
        db = get_database()
        return list(db.trades.find({'status': 'closed'}).sort('timestamp', -1).limit(limit))
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        raise

def update_account_info(account_data):
    """Update account information"""
    try:
        db = get_database()
        account_data['timestamp'] = datetime.utcnow()
        result = db.account.insert_one(account_data)
        return result.inserted_id
    except Exception as e:
        logger.error(f"Error updating account info: {str(e)}")
        raise

def get_latest_account_info():
    """Get the latest account information"""
    try:
        db = get_database()
        return db.account.find_one(sort=[('timestamp', -1)])
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        raise

def get_performance_metrics():
    """Get performance metrics"""
    try:
        db = get_database()
        return list(db.performance.find().sort('timestamp', -1).limit(1))
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise 