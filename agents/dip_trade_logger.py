"""
Database logging module for dip trades.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection
from config.database import get_database

logger = logging.getLogger(__name__)

class DipTradeLogger:
    """
    Logger for dip trade events and analytics.
    Stores trade data and ML predictions in MongoDB.
    """
    
    def __init__(self):
        """Initialize the dip trade logger."""
        self.db = get_database()
        self._ensure_collections()
        
    def _ensure_collections(self) -> None:
        """Ensure required collections exist with proper indexes."""
        try:
            # Dip trades collection
            if 'dip_trades' not in self.db.list_collection_names():
                dip_trades = self.db.create_collection('dip_trades')
                dip_trades.create_index([('timestamp', DESCENDING)])
                dip_trades.create_index([('symbol', 1)])
                dip_trades.create_index([('status', 1)])
                
            # Dip predictions collection
            if 'dip_predictions' not in self.db.list_collection_names():
                predictions = self.db.create_collection('dip_predictions')
                predictions.create_index([('timestamp', DESCENDING)])
                predictions.create_index([('symbol', 1)])
                
            # Dip analytics collection
            if 'dip_analytics' not in self.db.list_collection_names():
                analytics = self.db.create_collection('dip_analytics')
                analytics.create_index([('timestamp', DESCENDING)])
                analytics.create_index([('symbol', 1)])
                
            logger.info("Dip trade collections initialized")
            
        except Exception as e:
            logger.error(f"Error ensuring collections: {str(e)}")
            
    def log_dip_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a dip trade event.
        
        Args:
            trade_data: Trade details including entry price, size, etc.
            
        Returns:
            str: ID of inserted document
        """
        try:
            trade_data['timestamp'] = datetime.now()
            result = self.db.dip_trades.insert_one(trade_data)
            logger.info(f"Logged dip trade: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging dip trade: {str(e)}")
            return ""
            
    def log_prediction(self, symbol: str, prediction_data: Dict[str, Any]) -> str:
        """
        Log a dip prediction event.
        
        Args:
            symbol: Trading pair symbol
            prediction_data: Prediction details and metadata
            
        Returns:
            str: ID of inserted document
        """
        try:
            doc = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                **prediction_data
            }
            result = self.db.dip_predictions.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            return ""
            
    def update_trade_status(self, trade_id: str, status: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a dip trade.
        
        Args:
            trade_id: Trade document ID
            status: New status
            metadata: Additional metadata to update
            
        Returns:
            bool: Success status
        """
        try:
            update_doc = {
                '$set': {
                    'status': status,
                    'last_updated': datetime.now()
                }
            }
            
            if metadata:
                update_doc['$set'].update(metadata)
                
            result = self.db.dip_trades.update_one(
                {'_id': trade_id},
                update_doc
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating trade status: {str(e)}")
            return False
            
    def log_analytics(self, symbol: str, analytics_data: Dict[str, Any]) -> str:
        """
        Log dip trading analytics data.
        
        Args:
            symbol: Trading pair symbol
            analytics_data: Analytics metrics and data
            
        Returns:
            str: ID of inserted document
        """
        try:
            doc = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                **analytics_data
            }
            result = self.db.dip_analytics.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging analytics: {str(e)}")
            return ""
            
    def get_recent_trades(self, symbol: Optional[str] = None, 
                         limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent dip trades.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of trades to return
            
        Returns:
            List[Dict[str, Any]]: Recent trades
        """
        try:
            query = {'symbol': symbol} if symbol else {}
            return list(self.db.dip_trades
                       .find(query)
                       .sort('timestamp', DESCENDING)
                       .limit(limit))
        except Exception as e:
            logger.error(f"Error fetching recent trades: {str(e)}")
            return []
            
    def get_prediction_accuracy(self, symbol: str, 
                              window_hours: int = 24) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            symbol: Trading pair symbol
            window_hours: Time window for calculation
            
        Returns:
            Dict[str, float]: Accuracy metrics
        """
        try:
            since = datetime.now() - timedelta(hours=window_hours)
            
            # Get predictions
            predictions = list(self.db.dip_predictions.find({
                'symbol': symbol,
                'timestamp': {'$gte': since}
            }))
            
            # Get actual trades
            trades = list(self.db.dip_trades.find({
                'symbol': symbol,
                'timestamp': {'$gte': since}
            }))
            
            # Calculate metrics
            total_predictions = len(predictions)
            true_positives = sum(1 for p in predictions 
                               if p.get('is_dip') and 
                               any(t['entry_time'] - p['timestamp'] < timedelta(hours=1)
                                   for t in trades))
            
            return {
                'total_predictions': total_predictions,
                'true_positives': true_positives,
                'accuracy': true_positives / total_predictions if total_predictions > 0 else 0,
                'window_hours': window_hours
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {str(e)}")
            return {
                'total_predictions': 0,
                'true_positives': 0,
                'accuracy': 0,
                'window_hours': window_hours
            }
            
    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dip trading performance metrics.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            query = {'symbol': symbol} if symbol else {}
            
            # Get completed trades
            trades = list(self.db.dip_trades.find({
                **query,
                'status': 'closed'
            }))
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'total_profit': 0
                }
                
            # Calculate metrics
            profitable_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
            total_profit = sum(t.get('profit', 0) for t in trades)
            
            return {
                'total_trades': len(trades),
                'win_rate': profitable_trades / len(trades),
                'avg_profit': total_profit / len(trades),
                'total_profit': total_profit
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0
            } 