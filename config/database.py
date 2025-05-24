import os
from typing import Optional, Dict, Any
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import logging
import dns.resolver
from datetime import datetime
from functools import wraps
import time
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# Load environment variables or use defaults
def get_env_var(key: str, default: str) -> str:
    return os.getenv(key, default)

# MongoDB Configuration
LOCAL_MONGODB_URI = get_env_var("LOCAL_MONGODB_URI", "mongodb://localhost:27017/trading_bot")
REMOTE_MONGODB_URI = get_env_var("REMOTE_MONGODB_URI", LOCAL_MONGODB_URI)  # Fallback to local if remote not specified
DB_NAME = get_env_var("MONGODB_DB_NAME", "trading_bot")

# Connection settings
MONGODB_SETTINGS = {
    "serverSelectionTimeoutMS": 5000,  # Reduced timeout for local connection
    "connectTimeoutMS": 5000,
    "socketTimeoutMS": 5000,
    "maxPoolSize": 100,
    "minPoolSize": 10,
    "retryWrites": True,
    "maxIdleTimeMS": 45000
}

# Collection names
COLLECTIONS = {
    'trades': 'trades',
    'performance': 'performance',
    'alerts': 'alerts',
    'strategies': 'strategies',
    'market_data': 'market_data',
    'account': 'account'
}

# Global connection pool
_local_client = None
_remote_client = None

def retry_with_backoff(retries=2, backoff_in_seconds=1):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            last_exception = None
            while True:
                try:
                    return func(*args, **kwargs)
                except (ConnectionFailure, pymongo.errors.ServerSelectionTimeoutError) as e:
                    # Don't retry these specific errors
                    raise
                except Exception as e:
                    last_exception = e
                    if x == retries:
                        raise last_exception
                    else:
                        x += 1
                        wait = (backoff_in_seconds * 2 ** x)
                        logger.warning(f"Retrying {func.__name__} in {wait} seconds... (Attempt {x}/{retries})")
                        time.sleep(wait)
        return wrapper
    return decorator

def try_connect_mongodb(uri: str) -> Optional[MongoClient]:
    """Try to connect to MongoDB with the given URI"""
    try:
        client = MongoClient(uri, **MONGODB_SETTINGS)
        # Test connection
        client.admin.command('ping')
        return client
    except ConnectionFailure as e:
        logger.warning(f"Failed to connect to MongoDB at {uri.split('@')[-1]}: {str(e)}")
        raise  # Re-raise ConnectionFailure
    except Exception as e:
        logger.warning(f"Failed to connect to MongoDB at {uri.split('@')[-1]}: {str(e)}")
        return None

@retry_with_backoff(retries=2)
def get_database(remote=False, db_name=None):
    """
    Get database connection from the connection pool.
    Args:
        remote (bool): Whether to use remote MongoDB. Defaults to False.
        db_name (str): Optional database name to override default.
    """
    global _local_client, _remote_client
    
    try:
        # Always try local connection first unless remote is explicitly requested
        if not remote:
            if _local_client is None:
                logger.info("Connecting to local MongoDB...")
                _local_client = MongoClient(LOCAL_MONGODB_URI, **MONGODB_SETTINGS)
                # Test connection
                _local_client.admin.command('ping')
            return _local_client[db_name or DB_NAME]
        else:
            # Only try remote connection if explicitly requested
            if _remote_client is None:
                logger.info("Connecting to remote MongoDB...")
                _remote_client = MongoClient(REMOTE_MONGODB_URI, **MONGODB_SETTINGS)
                # Test connection
                _remote_client.admin.command('ping')
            return _remote_client[db_name or DB_NAME]
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def close_connections():
    """Close all database connections"""
    global _local_client, _remote_client
    
    if _local_client:
        try:
            _local_client.close()
            logger.info("Closed local MongoDB connection")
        except Exception as e:
            logger.error(f"Error closing local MongoDB connection: {str(e)}")
        finally:
            _local_client = None
            
    if _remote_client:
        try:
            _remote_client.close()
            logger.info("Closed remote MongoDB connection")
        except Exception as e:
            logger.error(f"Error closing remote MongoDB connection: {str(e)}")
        finally:
            _remote_client = None

def get_connection_info() -> Dict[str, Any]:
    """Get information about the current database connection"""
    return {
        "is_connected": _local_client is not None or _remote_client is not None,
        "using_local": _local_client is not None,
        "database_name": DB_NAME,
        "uri": LOCAL_MONGODB_URI if _local_client is not None else REMOTE_MONGODB_URI
    }

@retry_with_backoff(retries=3)
def init_collections():
    """Initialize database collections with proper indexes"""
    try:
        db = get_database()
        
        # Trades collection
        trades_col = db.get_collection('trades')
        if 'trades' not in db.list_collection_names():
            db.create_collection('trades')
        trades_col.create_index([('timestamp', -1)])
        trades_col.create_index([('status', 1)])
        trades_col.create_index([('pair', 1)])
        
        # Performance collection
        perf_col = db.get_collection('performance')
        if 'performance' not in db.list_collection_names():
            db.create_collection('performance')
        perf_col.create_index([('timestamp', -1)])
        perf_col.create_index([('metric', 1)])
        
        # Alerts collection
        alerts_col = db.get_collection('alerts')
        if 'alerts' not in db.list_collection_names():
            db.create_collection('alerts')
        alerts_col.create_index([('timestamp', -1)])
        alerts_col.create_index([('status', 1)])
        
        # Strategies collection
        strat_col = db.get_collection('strategies')
        if 'strategies' not in db.list_collection_names():
            db.create_collection('strategies')
        strat_col.create_index([('name', 1)], unique=True)
        
        # Market data collection
        market_col = db.get_collection('market_data')
        if 'market_data' not in db.list_collection_names():
            db.create_collection('market_data')
        market_col.create_index([('timestamp', -1)])
        market_col.create_index([('pair', 1)])
        market_col.create_index([('timeframe', 1)])
        
        # Account collection
        account_col = db.get_collection('account')
        if 'account' not in db.list_collection_names():
            db.create_collection('account')
        account_col.create_index([('timestamp', -1)])
        
        logger.info("Database collections initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing collections: {str(e)}")
        raise

# Database operations with retry
@retry_with_backoff(retries=3)
def insert_trade(trade_data: Dict[str, Any]) -> str:
    """
    Insert a new trade into the database.
    
    Args:
        trade_data (dict): Trade data to insert
        
    Returns:
        str: ID of the inserted trade
    """
    try:
        db = get_database()
        
        # Ensure required fields
        if not all(k in trade_data for k in ['symbol', 'side', 'quantity', 'price', 'status']):
            raise ValueError("Missing required trade data fields")
            
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
            
        # Insert trade
        result = db.trades.insert_one(trade_data)
        
        if result and result.inserted_id:
            logger.info(f"Trade inserted successfully with ID: {result.inserted_id}")
            return str(result.inserted_id)
        else:
            logger.error("Failed to insert trade - no ID returned")
            raise Exception("Trade insertion failed")
            
    except Exception as e:
        logger.error(f"Error inserting trade: {str(e)}")
        raise

@retry_with_backoff(retries=3)
def get_active_trades():
    """Get all active trades with retry"""
    try:
        db = get_database()
        return list(db.trades.find({'status': 'active'}))
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}")
        raise

@retry_with_backoff(retries=3)
def get_trade_history(limit: int = 50):
    """Get trade history with retry"""
    try:
        db = get_database()
        return list(db.trades.find({'status': 'closed'}).sort('timestamp', -1).limit(limit))
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        raise

@retry_with_backoff(retries=3)
def update_market_data(symbol: str, timeframe: str, data: Dict[str, Any]):
    """Update market data with retry"""
    try:
        db = get_database()
        data['timestamp'] = datetime.utcnow()
        data['symbol'] = symbol
        data['timeframe'] = timeframe
        result = db.market_data.update_one(
            {'symbol': symbol, 'timeframe': timeframe},
            {'$set': data},
            upsert=True
        )
        return result.modified_count or result.upserted_id
    except Exception as e:
        logger.error(f"Error updating market data: {str(e)}")
        raise

@retry_with_backoff(retries=3)
def get_market_data(symbol: str, timeframe: str, limit: int = 500):
    """Get market data with retry"""
    try:
        db = get_database()
        return list(db.market_data.find(
            {'symbol': symbol, 'timeframe': timeframe}
        ).sort('timestamp', -1).limit(limit))
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
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