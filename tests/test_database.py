"""
Tests for database operations.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pymongo
from datetime import datetime
from config.database import (
    get_database,
    close_connections,
    init_collections,
    insert_trade,
    get_active_trades,
    get_trade_history,
    update_market_data,
    get_market_data,
    _local_client,
    _remote_client
)
from bson.objectid import ObjectId

class TestDatabase(unittest.TestCase):
    @patch('config.database.retry_with_backoff')
    def test_error_handling(self, mock_retry):
        """Test database error handling."""
        # Make retry decorator just run the function once without retrying
        def mock_decorator(retries):
            def wrapper(func):
                def wrapped(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapped
            return wrapper
        mock_retry.side_effect = mock_decorator
        
        # Reset global clients to ensure clean state
        global _local_client, _remote_client
        from config.database import _local_client, _remote_client
        _local_client = None
        _remote_client = None
        
        # Test connection failure
        with patch('config.database.MongoClient') as mock_client:
            # Create a mock instance that raises ConnectionFailure on any operation
            mock_instance = MagicMock()
            mock_instance.side_effect = pymongo.errors.ConnectionFailure("Connection error")
            mock_client.side_effect = pymongo.errors.ConnectionFailure("Connection error")
            
            with self.assertRaises(pymongo.errors.ConnectionFailure):
                get_database()
                
        # Test operation timeout
        with patch('config.database.MongoClient') as mock_client:
            # Create a mock instance that raises ServerSelectionTimeoutError on any operation
            mock_instance = MagicMock()
            mock_instance.side_effect = pymongo.errors.ServerSelectionTimeoutError("Timeout")
            mock_client.side_effect = pymongo.errors.ServerSelectionTimeoutError("Timeout")
            
            with self.assertRaises(pymongo.errors.ServerSelectionTimeoutError):
                get_database()