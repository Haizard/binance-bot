"""
Machine Learning based dip prediction and analysis module.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

logger = logging.getLogger(__name__)

class MLDipPredictor:
    """
    ML-based dip prediction and analysis module.
    Uses ensemble methods to predict dips and validate recovery patterns.
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the ML predictor."""
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.recovery_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.is_trained = False
        self._ensure_model_dir()
        
    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
    def prepare_features(self, price_history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Prepare feature matrix from price history.
        
        Args:
            price_history: List of price data points
            
        Returns:
            np.ndarray: Feature matrix
        """
        df = pd.DataFrame(price_history)
        
        # Calculate technical features
        df['returns'] = df['price'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        df['price_ma'] = df['price'].rolling(window=20).mean()
        df['price_std'] = df['price'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_z_score'] = (df['price'] - df['price_ma']) / df['price_std']
        
        # Calculate momentum indicators
        df['rsi'] = self._calculate_rsi(df['price'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['price'])
        
        # Drop NaN values from calculations
        df = df.dropna()
        
        # Select features for ML
        features = [
            'returns', 'volume_ratio', 'price_z_score',
            'rsi', 'macd', 'macd_signal'
        ]
        
        return df[features].values
        
    def train_models(self, price_history: List[Dict[str, Any]], known_dips: List[Dict[str, Any]]) -> None:
        """
        Train ML models on historical data.
        
        Args:
            price_history: Historical price data
            known_dips: List of known dip events for training
        """
        try:
            X = self.prepare_features(price_history)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X)
            
            # Prepare recovery prediction data
            recovery_X = []
            recovery_y = []
            
            for dip in known_dips:
                dip_start = dip['timestamp']
                dip_end = dip_start + timedelta(hours=24)
                dip_data = [p for p in price_history 
                           if dip_start <= p['timestamp'] <= dip_end]
                
                if dip_data:
                    features = self.prepare_features(dip_data)
                    recovery_pct = (dip_data[-1]['price'] - dip_data[0]['price']) / dip_data[0]['price']
                    
                    recovery_X.append(features[-1])
                    recovery_y.append(recovery_pct)
            
            if recovery_X and recovery_y:
                self.recovery_predictor.fit(recovery_X, recovery_y)
            
            self.is_trained = True
            self._save_models()
            logger.info("ML models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
            
    def predict_dip(self, current_data: List[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Predict if current market state indicates a dip.
        
        Args:
            current_data: Recent market data
            
        Returns:
            Tuple[bool, float, dict]: (is_dip, confidence, metadata)
        """
        try:
            X = self.prepare_features(current_data)
            if not self.is_trained:
                self._load_models()
                
            # Get anomaly scores
            anomaly_scores = self.anomaly_detector.score_samples(X)
            
            # Predict recovery potential
            recovery_potential = self.recovery_predictor.predict(X[-1].reshape(1, -1))[0]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(anomaly_scores[-1], recovery_potential)
            
            # Determine if it's a dip
            is_dip = anomaly_scores[-1] < -0.5 and recovery_potential > 0
            
            metadata = {
                'anomaly_score': float(anomaly_scores[-1]),
                'recovery_potential': float(recovery_potential),
                'technical_indicators': {
                    'rsi': float(X[-1][3]),  # RSI is 4th feature
                    'macd': float(X[-1][4])  # MACD is 5th feature
                }
            }
            
            return is_dip, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error in dip prediction: {str(e)}")
            return False, 0.0, {}
            
    def _calculate_confidence(self, anomaly_score: float, recovery_potential: float) -> float:
        """Calculate confidence score for dip prediction."""
        # Normalize scores to 0-1 range
        anomaly_conf = 1 / (1 + np.exp(anomaly_score))  # Sigmoid transformation
        recovery_conf = max(0, min(1, recovery_potential))
        
        # Weighted combination
        return 0.7 * anomaly_conf + 0.3 * recovery_conf
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD technical indicator."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
        
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
            joblib.dump(self.anomaly_detector, os.path.join(self.model_dir, 'anomaly_detector.joblib'))
            joblib.dump(self.recovery_predictor, os.path.join(self.model_dir, 'recovery_predictor.joblib'))
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            
    def _load_models(self) -> None:
        """Load trained models from disk."""
        try:
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
            self.anomaly_detector = joblib.load(os.path.join(self.model_dir, 'anomaly_detector.joblib'))
            self.recovery_predictor = joblib.load(os.path.join(self.model_dir, 'recovery_predictor.joblib'))
            self.is_trained = True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}") 