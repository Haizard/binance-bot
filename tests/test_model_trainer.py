"""
Tests for automated model training module.
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from agents.model_trainer import ModelTrainer
from tests.mock_data_agent import MockDataAgent

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for model files
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'model_dir': self.test_dir,
            'performance_threshold': 0.7,
            'min_training_samples': 100,
            'retraining_interval_hours': 24
        }
        self.trainer = ModelTrainer(self.config)
        self.data_agent = MockDataAgent()
        
        # Set up mock data
        self.data_agent.set_mock_price('BTCUSDT', 50000.0)
        self.data_agent.add_mock_trade({
            'timestamp': datetime.now(),
            'symbol': 'BTCUSDT',
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'profit': 0.02
        })
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    async def test_check_and_retrain(self):
        """Test model retraining check and execution."""
        # Mock dependencies
        self.trainer._evaluate_current_model = AsyncMock(
            return_value={'f1_score': 0.6}  # Below threshold
        )
        self.trainer._get_training_data = AsyncMock(
            return_value=[{'features': np.random.rand(10), 'is_dip': True}] * 150
        )
        self.trainer._perform_retraining = AsyncMock()
        
        # Test retraining execution
        result = await self.trainer.check_and_retrain('BTCUSDT')
        self.assertTrue(result)
        self.trainer._perform_retraining.assert_called_once()
        
        # Test with insufficient data
        self.trainer._get_training_data = AsyncMock(return_value=[])
        result = await self.trainer.check_and_retrain('BTCUSDT')
        self.assertFalse(result)
        
    async def test_evaluate_current_model(self):
        """Test model performance evaluation."""
        # Mock recent predictions
        mock_predictions = [
            {'actual_dip': True, 'predicted_dip': True},
            {'actual_dip': False, 'predicted_dip': False},
            {'actual_dip': True, 'predicted_dip': False}
        ]
        self.trainer._get_recent_predictions = AsyncMock(
            return_value=mock_predictions
        )
        
        metrics = await self.trainer._evaluate_current_model('BTCUSDT')
        
        # Verify metrics calculation
        self.assertIn('f1_score', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        
    async def test_get_training_data(self):
        """Test training data preparation."""
        # Mock trade data
        mock_trades = [
            {
                'timestamp': datetime.now(),
                'price': 50000,
                'volume': 1.5,
                'profit': 0.02
            }
        ]
        self.trainer.logger.get_recent_trades = Mock(
            return_value=mock_trades
        )
        
        # Mock price history
        mock_price_history = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'price': 50000 - i * 100,
                'volume': 1.5
            }
            for i in range(100)
        ]
        self.trainer._get_price_history = AsyncMock(
            return_value=mock_price_history
        )
        
        training_data = await self.trainer._get_training_data('BTCUSDT')
        
        # Verify training data structure
        self.assertTrue(len(training_data) > 0)
        self.assertIn('features', training_data[0])
        self.assertIn('is_dip', training_data[0])
        
    def test_should_retrain(self):
        """Test retraining decision logic."""
        # Test performance threshold
        performance = {'f1_score': 0.6}  # Below threshold
        self.assertTrue(self.trainer._should_retrain(performance))
        
        performance = {'f1_score': 0.8}  # Above threshold
        self.trainer._get_last_training_time = Mock(
            return_value=datetime.now()
        )
        self.assertFalse(self.trainer._should_retrain(performance))
        
        # Test training interval
        self.trainer._get_last_training_time = Mock(
            return_value=datetime.now() - timedelta(hours=25)
        )
        self.assertTrue(self.trainer._should_retrain(performance))
        
    async def test_perform_retraining(self):
        """Test model retraining process."""
        # Prepare test data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        training_data = [
            {'features': x, 'is_dip': y_i}
            for x, y_i in zip(X, y)
        ]
        
        await self.trainer._perform_retraining('BTCUSDT', training_data)
        
        # Verify model files were created
        metadata_file = os.path.join(self.test_dir, 'training_metadata.json')
        self.assertTrue(os.path.exists(metadata_file))
        
    def test_get_market_context(self):
        """Test market context extraction."""
        # Create sample price history
        price_history = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'price': 50000 - i * 100,
                'volume': 1.5
            }
            for i in range(48)  # 2 days of hourly data
        ]
        
        context = self.trainer._get_market_context(
            price_history,
            datetime.now() - timedelta(hours=24)
        )
        
        # Verify context features
        self.assertIsInstance(context, np.ndarray)
        
    def test_generate_negative_samples(self):
        """Test negative sample generation."""
        # Create sample price history
        price_history = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'price': 50000 - i * 100,
                'volume': 1.5
            }
            for i in range(100)
        ]
        
        # Create sample dip timestamps
        dip_times = [
            datetime.now() - timedelta(hours=i*24)
            for i in range(3)
        ]
        
        samples = self.trainer._generate_negative_samples(
            price_history,
            dip_times
        )
        
        # Verify negative samples
        self.assertTrue(len(samples) > 0)
        for sample in samples:
            self.assertFalse(sample['is_dip'])
            
    def test_get_random_non_dip_time(self):
        """Test random time selection for negative samples."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        dip_times = [
            datetime.now() - timedelta(hours=i*24)
            for i in range(3)
        ]
        
        time = self.trainer._get_random_non_dip_time(
            start, end, dip_times
        )
        
        # Verify selected time
        self.assertTrue(start <= time <= end)
        for dip_time in dip_times:
            self.assertTrue(abs((time - dip_time).total_seconds()) > 3600)
            
    def test_save_model_metadata(self):
        """Test model metadata saving."""
        symbol = 'BTCUSDT'
        metadata = {
            'timestamp': datetime.now(),
            'cv_scores_mean': 0.85,
            'cv_scores_std': 0.05,
            'training_samples': 1000
        }
        
        self.trainer._save_model_metadata(symbol, metadata)
        
        # Verify metadata file
        metadata_file = os.path.join(self.test_dir, 'training_metadata.json')
        self.assertTrue(os.path.exists(metadata_file))

if __name__ == '__main__':
    unittest.main() 