"""
Test configuration settings.
"""
import os
from datetime import timedelta

# Test environment settings
TEST_CONFIG = {
    'model_dir': 'test_models',
    'performance_threshold': 0.7,
    'min_training_samples': 100,
    'retraining_interval_hours': 24,
    'twitter_bearer_token': 'test_token',
    'reddit_client_id': 'test_id',
    'reddit_client_secret': 'test_secret',
    'news_api_key': 'test_key'
}

# Test trading parameters
TEST_TRADE_PARAMS = {
    'min_dip': 2.0,
    'recovery_target': 1.0,
    'stop_loss': 1.0,
    'volume_surge_threshold': 2.0,
    'cooldown_period': timedelta(hours=4)
}

# Test database settings
TEST_DB_CONFIG = {
    'host': 'localhost',
    'port': 27017,
    'db_name': 'test_trading_bot'
}

# Test symbols
TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

# Test intervals
TEST_INTERVALS = ['1m', '5m', '15m', '1h', '4h', '1d']

# Test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
if not os.path.exists(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR)

# Test model paths
TEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'test_models')
if not os.path.exists(TEST_MODEL_DIR):
    os.makedirs(TEST_MODEL_DIR)

# Test logging settings
TEST_LOG_CONFIG = {
    'level': 'DEBUG',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(TEST_DATA_DIR, 'test.log')
}

# Test API settings
TEST_API_CONFIG = {
    'base_url': 'http://test.api.endpoint',
    'timeout': 5,
    'max_retries': 3
}

# Test sentiment analysis settings
TEST_SENTIMENT_CONFIG = {
    'min_confidence': 0.6,
    'source_weights': {
        'social': 0.3,
        'news': 0.3,
        'technical': 0.4
    }
}

# Test backtesting settings
TEST_BACKTEST_CONFIG = {
    'start_balance': 10000,
    'position_size': 0.1,
    'max_positions': 3,
    'fee_rate': 0.001
}

# Test model training settings
TEST_TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'early_stopping_rounds': 10
}

# Test feature settings
TEST_FEATURE_CONFIG = {
    'price_features': [
        'close',
        'high',
        'low',
        'volume'
    ],
    'technical_indicators': [
        'rsi',
        'macd',
        'bollinger_bands'
    ],
    'window_sizes': [14, 26, 50]
}

# Test risk management settings
TEST_RISK_CONFIG = {
    'max_drawdown': 0.1,
    'max_risk_per_trade': 0.02,
    'max_daily_trades': 5,
    'max_position_size': 0.2
}

# Test monitoring settings
TEST_MONITOR_CONFIG = {
    'update_interval': 60,  # seconds
    'alert_thresholds': {
        'drawdown': 0.05,
        'profit': 0.1,
        'volume_surge': 3.0
    }
}

# Test validation thresholds
TEST_VALIDATION_THRESHOLDS = {
    'min_accuracy': 0.7,
    'min_precision': 0.7,
    'min_recall': 0.7,
    'min_f1': 0.7,
    'max_false_positives': 0.2
} 