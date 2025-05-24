"""
Automated model retraining module.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os
from agents.ml_dip_predictor import MLDipPredictor
from agents.dip_trade_logger import DipTradeLogger

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Automated model training and evaluation module.
    Handles periodic retraining and performance monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer."""
        self.config = config
        self.logger = DipTradeLogger()
        self.model_dir = config.get('model_dir', 'models')
        self.performance_threshold = config.get('performance_threshold', 0.7)
        self.min_samples = config.get('min_training_samples', 1000)
        self._ensure_model_dir()
        
    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
    async def check_and_retrain(self, symbol: str) -> bool:
        """
        Check if model retraining is needed and perform if necessary.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: Whether retraining was performed
        """
        try:
            # Check current model performance
            current_performance = await self._evaluate_current_model(symbol)
            
            # Get training data
            training_data = await self._get_training_data(symbol)
            if len(training_data) < self.min_samples:
                logger.info(f"Insufficient training data for {symbol}")
                return False
                
            # Decide if retraining is needed
            if self._should_retrain(current_performance):
                logger.info(f"Initiating model retraining for {symbol}")
                await self._perform_retraining(symbol, training_data)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in check_and_retrain: {str(e)}")
            return False
            
    async def _evaluate_current_model(self, symbol: str) -> Dict[str, float]:
        """Evaluate current model performance."""
        try:
            # Get recent predictions and outcomes
            predictions = await self._get_recent_predictions(symbol)
            if not predictions:
                return {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
                
            y_true = [p['actual_dip'] for p in predictions]
            y_pred = [p['predicted_dip'] for p in predictions]
            
            return {
                'f1_score': float(f1_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred)),
                'recall': float(recall_score(y_true, y_pred))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
            
    async def _get_training_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical data for training."""
        try:
            # Get price history and dip events
            trades = self.logger.get_recent_trades(symbol, limit=1000)
            price_history = await self._get_price_history(symbol)
            
            # Prepare training data
            training_data = []
            for trade in trades:
                # Get market context around trade
                context = self._get_market_context(
                    price_history,
                    trade['timestamp']
                )
                if context is not None:
                    training_data.append({
                        'features': context,
                        'is_dip': True,
                        'profit': trade.get('profit', 0)
                    })
                    
            # Add negative samples (non-dip periods)
            negative_samples = self._generate_negative_samples(
                price_history,
                [t['timestamp'] for t in trades]
            )
            training_data.extend(negative_samples)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            return []
            
    def _should_retrain(self, performance: Dict[str, float]) -> bool:
        """Determine if retraining is needed."""
        # Check if performance is below threshold
        if performance['f1_score'] < self.performance_threshold:
            return True
            
        # Check time since last training
        last_training = self._get_last_training_time()
        if not last_training:
            return True
            
        training_interval = timedelta(
            hours=self.config.get('retraining_interval_hours', 24)
        )
        return datetime.now() - last_training > training_interval
        
    async def _perform_retraining(self, symbol: str, 
                                training_data: List[Dict[str, Any]]) -> None:
        """Perform model retraining."""
        try:
            # Prepare features and labels
            X = np.array([d['features'] for d in training_data])
            y = np.array([d['is_dip'] for d in training_data])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train new model
            predictor = MLDipPredictor(model_dir=self.model_dir)
            predictor.train_models(X_train, y_train)
            
            # Evaluate new model
            cv_scores = cross_val_score(
                predictor.anomaly_detector,
                X_train, y_train,
                cv=5
            )
            
            # Save model and metadata
            self._save_model_metadata(symbol, {
                'timestamp': datetime.now(),
                'cv_scores_mean': float(np.mean(cv_scores)),
                'cv_scores_std': float(np.std(cv_scores)),
                'training_samples': len(training_data)
            })
            
            logger.info(f"Model retraining completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            
    async def _get_price_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical price data."""
        try:
            # Get price data from database
            return list(self.logger.db.market_data.find(
                {'symbol': symbol}
            ).sort('timestamp', -1).limit(5000))
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return []
            
    def _get_market_context(self, price_history: List[Dict[str, Any]],
                           timestamp: datetime) -> Optional[np.ndarray]:
        """Extract market context around a timestamp."""
        try:
            # Find relevant price data window
            window_start = timestamp - timedelta(hours=24)
            window_end = timestamp
            
            window_data = [
                p for p in price_history
                if window_start <= p['timestamp'] <= window_end
            ]
            
            if len(window_data) < 24:  # Need at least 24 hours of data
                return None
                
            # Sort by timestamp
            window_data.sort(key=lambda x: x['timestamp'])
            
            # Extract features
            prices = np.array([float(p['price']) for p in window_data])
            volumes = np.array([float(p['volume']) for p in window_data])
            
            # Calculate technical features
            price_changes = np.diff(prices) / prices[:-1]  # Price changes
            volume_changes = np.diff(volumes) / volumes[:-1]  # Volume changes
            volatility = np.std(price_changes)  # Volatility
            
            # Create feature vector
            features = np.array([
                np.mean(price_changes),  # Average price change
                np.std(price_changes),   # Price volatility
                np.mean(volume_changes), # Average volume change
                np.std(volume_changes),  # Volume volatility
                np.corrcoef(prices[:-1], volumes[:-1])[0, 1],  # Price-volume correlation
                volatility  # Overall volatility
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting market context: {str(e)}")
            return None
            
    def _generate_negative_samples(self, price_history: List[Dict[str, Any]],
                                 dip_timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Generate negative (non-dip) samples for training."""
        try:
            if not price_history:
                return []
                
            # Get time range
            times = [p['timestamp'] for p in price_history]
            start_time = min(times)
            end_time = max(times)
            
            # Generate random non-dip timestamps
            num_samples = len(dip_timestamps)  # Generate same number as positive samples
            negative_samples = []
            attempts = 0
            max_attempts = num_samples * 3
            
            while len(negative_samples) < num_samples and attempts < max_attempts:
                attempts += 1
                timestamp = self._get_random_non_dip_time(start_time, end_time, dip_timestamps)
                
                # Get market context for this timestamp
                context = self._get_market_context(price_history, timestamp)
                if context is not None:
                    negative_samples.append({
                        'features': context,
                        'is_dip': False,
                        'timestamp': timestamp
                    })
                    
            return negative_samples
            
        except Exception as e:
            logger.error(f"Error generating negative samples: {str(e)}")
            return []
            
    def _get_random_non_dip_time(self, start: datetime, end: datetime,
                                dip_times: List[datetime]) -> datetime:
        """Get random timestamp avoiding dip periods."""
        while True:
            # Generate random timestamp
            time_delta = end - start
            random_seconds = np.random.randint(0, time_delta.total_seconds())
            candidate_time = start + timedelta(seconds=random_seconds)
            
            # Check if it's far enough from dip times
            if all(abs((t - candidate_time).total_seconds()) > 3600
                   for t in dip_times):
                return candidate_time
                
    def _get_last_training_time(self) -> Optional[datetime]:
        """Get timestamp of last model training."""
        try:
            metadata_file = os.path.join(self.model_dir, 'training_metadata.json')
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                return metadata.get('last_training_time')
            return None
        except Exception as e:
            logger.error(f"Error getting last training time: {str(e)}")
            return None
            
    def _save_model_metadata(self, symbol: str, metadata: Dict[str, Any]) -> None:
        """Save model training metadata."""
        try:
            metadata_file = os.path.join(self.model_dir, 'training_metadata.json')
            existing_metadata = {}
            
            if os.path.exists(metadata_file):
                existing_metadata = joblib.load(metadata_file)
                
            existing_metadata.update({
                symbol: metadata,
                'last_training_time': datetime.now()
            })
            
            joblib.dump(existing_metadata, metadata_file)
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}") 