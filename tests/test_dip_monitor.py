"""
Tests for dip trading monitor dashboard.
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dashboards.dip_monitor import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    run_backtest,
    get_model_performance_history,
    get_sentiment_history
)
from tests.mock_data_agent import MockDataAgent

class TestDipMonitor(unittest.TestCase):
    """Test cases for dip monitor dashboard."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_agent = MockDataAgent()
        
        # Set up mock data
        self.data_agent.set_mock_price('BTCUSDT', 50000.0)
        for i in range(10):
            self.data_agent.add_mock_trade({
                'timestamp': datetime.now() - timedelta(hours=i),
                'symbol': 'BTCUSDT',
                'entry_price': 50000 - i * 100,
                'exit_price': 50100 - i * 100,
                'profit': [0.02, -0.01, 0.015, -0.02, 0.01][i % 5]
            })
        
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Test with sample returns
        returns = pd.Series([0.05, -0.03, 0.02, -0.05, 0.03])
        max_dd = calculate_max_drawdown(returns)
        
        # Verify drawdown calculation
        self.assertIsInstance(max_dd, float)
        self.assertLess(max_dd, 0)  # Drawdown should be negative
        
        # Test with no drawdown
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        max_dd = calculate_max_drawdown(returns)
        self.assertEqual(max_dd, 0)
        
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test with sample returns
        returns = pd.Series([0.05, -0.03, 0.02, -0.05, 0.03])
        sharpe = calculate_sharpe_ratio(returns)
        
        # Verify Sharpe ratio
        self.assertIsInstance(sharpe, float)
        
        # Test with constant returns (undefined Sharpe)
        returns = pd.Series([0.02] * 5)
        sharpe = calculate_sharpe_ratio(returns)
        self.assertTrue(np.isnan(sharpe) or np.isinf(sharpe))
        
    @patch('dashboards.dip_monitor.logger')
    def test_run_backtest(self, mock_logger):
        """Test backtesting functionality."""
        # Mock price data with a clear dip pattern
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(100)]
        prices = []
        base_price = 50000.0
        
        # Generate price data with dips
        for i in range(100):
            if i % 20 == 0:  # Create a dip every 20 hours
                prices.append(base_price * 0.97)  # 3% dip
            elif i % 20 == 1:  # Recovery after dip
                prices.append(base_price * 0.99)  # 2% recovery
            else:
                prices.append(base_price)
                
        mock_price_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': [1.5] * 100
        })
        mock_logger.db.market_data.find.return_value = mock_price_data.to_dict('records')
        
        # Run backtest
        results = run_backtest(
            'BTCUSDT',
            datetime.now() - timedelta(days=7),
            datetime.now(),
            {
                'min_dip': 2.0,  # Look for 2% dips
                'recovery_target': 1.0,  # Take profit at 1% gain
                'stop_loss': 1.0  # Stop loss at 1% loss
            }
        )
        
        # Verify backtest results
        self.assertIsNotNone(results)
        self.assertIn('total_trades', results)
        self.assertIn('win_rate', results)
        self.assertIn('avg_profit', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('trades', results)
        self.assertTrue(len(results['trades']) > 0)
        
    @patch('dashboards.dip_monitor.logger')
    def test_get_model_performance_history(self, mock_logger):
        """Test model performance history retrieval."""
        # Mock performance metrics
        mock_metrics = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'f1_score': 0.8 - i * 0.01,
                'precision': 0.85 - i * 0.01,
                'recall': 0.75 - i * 0.01
            }
            for i in range(24)
        ]
        mock_logger.db.dip_analytics.find.return_value = mock_metrics
        
        # Get performance history
        history = get_model_performance_history('BTCUSDT', 24)
        
        # Verify performance data
        self.assertIsInstance(history, pd.DataFrame)
        self.assertTrue(len(history) > 0)
        self.assertIn('f1_score', history.columns)
        self.assertIn('precision', history.columns)
        self.assertIn('recall', history.columns)
        
    @patch('dashboards.dip_monitor.logger')
    def test_get_sentiment_history(self, mock_logger):
        """Test sentiment history retrieval."""
        # Mock sentiment data
        mock_sentiment = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'symbol': 'BTCUSDT',
                'combined_score': 0.5 - i * 0.02,
                'social_score': 0.6 - i * 0.02,
                'news_score': 0.4 - i * 0.02,
                'technical_score': 0.5 - i * 0.02
            }
            for i in range(24)
        ]
        mock_logger.db.dip_analytics.find.return_value = mock_sentiment
        
        # Get sentiment history
        history = get_sentiment_history('BTCUSDT', 24)
        
        # Verify sentiment data
        self.assertIsInstance(history, pd.DataFrame)
        self.assertTrue(len(history) > 0)
        self.assertIn('combined_score', history.columns)
        self.assertIn('social_score', history.columns)
        self.assertIn('news_score', history.columns)
        self.assertIn('technical_score', history.columns)
        
    def test_backtest_edge_cases(self):
        """Test backtesting edge cases."""
        # Test with no price data
        results = run_backtest(
            'BTCUSDT',
            datetime.now(),
            datetime.now(),
            {
                'min_dip': 2.0,
                'recovery_target': 1.0,
                'stop_loss': 1.0
            }
        )
        self.assertIsNone(results)
        
        # Test with invalid parameters
        results = run_backtest(
            'BTCUSDT',
            datetime.now(),
            datetime.now() - timedelta(days=1),  # End before start
            {
                'min_dip': 2.0,
                'recovery_target': 1.0,
                'stop_loss': 1.0
            }
        )
        self.assertIsNone(results)
        
    def test_performance_metrics_edge_cases(self):
        """Test performance metrics edge cases."""
        # Test max drawdown with empty returns
        empty_returns = pd.Series([])
        max_dd = calculate_max_drawdown(empty_returns)
        self.assertEqual(max_dd, 0)
        
        # Test Sharpe ratio with empty returns
        sharpe = calculate_sharpe_ratio(empty_returns)
        self.assertTrue(np.isnan(sharpe))
        
        # Test with single return value
        single_return = pd.Series([0.05])
        max_dd = calculate_max_drawdown(single_return)
        sharpe = calculate_sharpe_ratio(single_return)
        self.assertEqual(max_dd, 0)
        self.assertTrue(np.isnan(sharpe))

if __name__ == '__main__':
    unittest.main() 