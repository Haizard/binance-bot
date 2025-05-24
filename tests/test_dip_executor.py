"""
Unit tests for dip detection and ML prediction logic.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd
from agents.dip_executor import DipExecutorModule
from agents.ml_dip_predictor import MLDipPredictor

@pytest.fixture
def dip_executor():
    """Create a DipExecutorModule instance for testing."""
    trade_agent = Mock()
    trade_agent._is_trading_enabled = Mock(return_value=True)
    trade_agent._is_within_trading_hours = Mock(return_value=True)
    executor = DipExecutorModule(trade_agent)
    
    # Set up test configuration with Decimal values
    executor.dip_config = {
        'min_dip_percent': Decimal('2.0'),
        'recovery_percent': Decimal('0.5'),
        'volume_increase_factor': Decimal('1.5'),
        'max_position_size': Decimal('0.1'),
        'price_window': 24,
        'enabled_pairs': ['BTCUSDT'],
        'cooldown_period': 4
    }
    return executor

def test_is_dip_trading_enabled(dip_executor):
    """Test dip trading enabled check."""
    # Test enabled pair
    assert dip_executor._is_dip_trading_enabled('BTCUSDT')
    
    # Test disabled pair
    assert not dip_executor._is_dip_trading_enabled('ETHUSDT')
    
    # Test when trading is disabled
    dip_executor.trade_agent._is_trading_enabled.return_value = False
    assert not dip_executor._is_dip_trading_enabled('BTCUSDT')

def test_is_in_cooldown(dip_executor):
    """Test cooldown period check."""
    symbol = 'BTCUSDT'
    
    # Test no cooldown
    assert not dip_executor._is_in_cooldown(symbol)
    
    # Test active cooldown
    dip_executor.dip_states[symbol] = {
        'last_trade_time': datetime.now(),
        'dip_percent': 2.5
    }
    assert dip_executor._is_in_cooldown(symbol)
    
    # Test expired cooldown
    dip_executor.dip_states[symbol] = {
        'last_trade_time': datetime.now() - timedelta(hours=5),
        'dip_percent': 2.5
    }
    assert not dip_executor._is_in_cooldown(symbol)

@pytest.mark.asyncio
async def test_verify_volume_surge(dip_executor):
    """Test volume surge verification."""
    symbol = 'BTCUSDT'
    
    # Test case with volume surge
    dip_executor.price_history[symbol] = [
        {'volume': Decimal('100')},
        {'volume': Decimal('100')},
        {'volume': Decimal('100')},
        {'volume': Decimal('100')},
        {'volume': Decimal('200')}  # 2x surge
    ]
    assert await dip_executor._verify_volume_surge(symbol)
    
    # Test case without volume surge
    dip_executor.price_history[symbol] = [
        {'volume': Decimal('100')},
        {'volume': Decimal('100')},
        {'volume': Decimal('100')},
        {'volume': Decimal('100')},
        {'volume': Decimal('110')}  # Only 10% increase
    ]
    assert not await dip_executor._verify_volume_surge(symbol)

@pytest.mark.asyncio
async def test_verify_recovery_signs(dip_executor):
    """Test recovery signs verification."""
    symbol = 'BTCUSDT'
    
    # Test case with recovery
    dip_executor.price_history[symbol] = [
        {'price': Decimal('100')},
        {'price': Decimal('95')},
        {'price': Decimal('90')},
        {'price': Decimal('92')},
        {'price': Decimal('94')}  # Recovering
    ]
    assert await dip_executor._verify_recovery_signs(symbol)
    
    # Test case without recovery
    dip_executor.price_history[symbol] = [
        {'price': Decimal('100')},
        {'price': Decimal('98')},
        {'price': Decimal('95')},
        {'price': Decimal('93')},
        {'price': Decimal('90')}  # Still falling
    ]
    assert not await dip_executor._verify_recovery_signs(symbol)

@pytest.mark.asyncio
@patch('agents.dip_executor.DipExecutorModule._execute_dip_trade')
@patch('agents.dip_executor.DipExecutorModule._is_dip_trading_enabled')
@patch('agents.dip_executor.DipExecutorModule._is_in_cooldown')
@patch('agents.dip_executor.DipExecutorModule._verify_volume_surge')
@patch('agents.dip_executor.DipExecutorModule._verify_recovery_signs')
async def test_check_for_dip(mock_recovery, mock_volume, mock_cooldown, mock_enabled, mock_execute, dip_executor):
    """Test complete dip detection flow."""
    symbol = 'BTCUSDT'
    
    # Mock all the checks to return True
    mock_enabled.return_value = True
    mock_cooldown.return_value = False
    mock_volume.return_value = True
    mock_recovery.return_value = True
    
    # Set up price history with a significant dip (>2%)
    dip_executor.price_history[symbol] = [
        {'price': Decimal('100'), 'volume': Decimal('100'), 'timestamp': datetime.now()},
        {'price': Decimal('99'), 'volume': Decimal('100'), 'timestamp': datetime.now()},
        {'price': Decimal('95'), 'volume': Decimal('150'), 'timestamp': datetime.now()},
        {'price': Decimal('92'), 'volume': Decimal('200'), 'timestamp': datetime.now()},  # 8% drop
        {'price': Decimal('94'), 'volume': Decimal('180'), 'timestamp': datetime.now()}
    ]
    
    await dip_executor._check_for_dip(symbol)
    mock_execute.assert_called_once()
    
    # Test with insufficient price drop
    dip_executor.price_history[symbol] = [
        {'price': Decimal('100'), 'volume': Decimal('100'), 'timestamp': datetime.now()},
        {'price': Decimal('99'), 'volume': Decimal('100'), 'timestamp': datetime.now()},
        {'price': Decimal('98.5'), 'volume': Decimal('100'), 'timestamp': datetime.now()},
        {'price': Decimal('98'), 'volume': Decimal('100'), 'timestamp': datetime.now()},
        {'price': Decimal('99'), 'volume': Decimal('100'), 'timestamp': datetime.now()}
    ]
    
    mock_execute.reset_mock()
    await dip_executor._check_for_dip(symbol)
    mock_execute.assert_not_called()

@pytest.fixture
def ml_predictor():
    """Create an MLDipPredictor instance for testing."""
    return MLDipPredictor()

def test_prepare_features(ml_predictor):
    """Test feature preparation."""
    # Create sample price history
    price_history = [
        {
            'timestamp': datetime.now() - timedelta(hours=i),
            'price': 100 - i,
            'volume': 100 + i
        }
        for i in range(30)
    ]
    
    features = ml_predictor.prepare_features(price_history)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 6  # 6 features

def test_calculate_technical_indicators(ml_predictor):
    """Test technical indicator calculations."""
    prices = pd.Series([100, 102, 98, 103, 97, 105])
    
    # Test RSI
    rsi = ml_predictor._calculate_rsi(prices)
    assert isinstance(rsi, pd.Series)
    assert all(0 <= x <= 100 for x in rsi.dropna())
    
    # Test MACD
    macd, signal = ml_predictor._calculate_macd(prices)
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)

@patch('agents.ml_dip_predictor.MLDipPredictor._load_models')
def test_predict_dip(mock_load, ml_predictor):
    """Test dip prediction."""
    # Create sample current data
    current_data = [
        {
            'timestamp': datetime.now() - timedelta(minutes=i),
            'price': 100 - i,
            'volume': 100 + i
        }
        for i in range(30)
    ]
    
    # Mock trained models
    ml_predictor.is_trained = True
    ml_predictor.anomaly_detector.score_samples = Mock(
        return_value=np.array([-0.8])  # Strong anomaly
    )
    ml_predictor.recovery_predictor.predict = Mock(
        return_value=np.array([0.02])  # 2% recovery predicted
    )
    
    is_dip, confidence, metadata = ml_predictor.predict_dip(current_data)
    
    assert is_dip
    assert confidence > 0
    assert 'anomaly_score' in metadata
    assert 'recovery_potential' in metadata
    assert 'technical_indicators' in metadata 