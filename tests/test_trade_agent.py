"""
Tests for trade agent functionality.
"""
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from datetime import datetime, timedelta
from agents.trade_agent import TradeAgent
from agents.base_agent import BaseAgent

@pytest_asyncio.fixture
async def trade_agent():
    """Fixture to create a trade agent for testing."""
    config = {
        'max_open_trades': 3,
        'trade_size_usd': 1000,
        'max_loss_percent': 2.0,
        'take_profit_percent': 1.5,
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT'],
        'trading_enabled': True
    }
    agent = TradeAgent(config)
    return agent

@pytest.mark.asyncio
async def test_execute_trade(trade_agent):
    """Test trade execution."""
    # Enable trading
    trade_agent.config['trading_enabled'] = True
    trade_agent.trade_size_usd = 100000  # Set a high limit to allow test trades
    
    trade_params = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 0.1,
        'price': 50000.0
    }
    
    # Test successful trade
    with patch.object(trade_agent, '_place_order') as mock_place_order, \
         patch.object(trade_agent, '_is_within_risk_limits') as mock_risk_check, \
         patch.object(trade_agent, 'validate_trade_params') as mock_validate:
        mock_place_order.return_value = {
            'orderId': '12345',
            'status': 'FILLED'
        }
        mock_risk_check.return_value = True
        mock_validate.return_value = True
        
        result = await trade_agent.execute_trade(trade_params)
        assert result['success']
        assert result['order_id'] == '12345'
        assert len(trade_agent.open_trades) == 1
    
    # Reset state before next test
    trade_agent.open_trades = []
    
    # Test invalid parameters
    invalid_params = trade_params.copy()
    invalid_params['quantity'] = -0.1
    with patch.object(trade_agent, 'validate_trade_params') as mock_validate:
        mock_validate.return_value = False
        result = await trade_agent.execute_trade(invalid_params)
        assert not result['success']
        assert 'invalid trade parameters' in result['message'].lower()
    
    # Test max open trades limit
    trade_agent.open_trades = [
        {'symbol': 'BTCUSDT', 'quantity': 0.1},
        {'symbol': 'ETHUSDT', 'quantity': 1.0},
        {'symbol': 'BNBUSDT', 'quantity': 2.0}
    ]
    
    result = await trade_agent.execute_trade(trade_params)
    assert not result['success']
    assert 'max open trades limit' in result['message'].lower()
    
    # Reset state before testing position size limit
    trade_agent.open_trades = []
    
    # Test position size limit
    large_params = trade_params.copy()
    large_params['quantity'] = 1.0  # 50,000 USD position
    trade_agent.trade_size_usd = 1000  # Set a low limit to trigger the check
    
    with patch.object(trade_agent, '_is_within_risk_limits') as mock_risk_check, \
         patch.object(trade_agent, 'validate_trade_params') as mock_validate:
        mock_risk_check.return_value = True
        mock_validate.return_value = True
        
        result = await trade_agent.execute_trade(large_params)
        assert not result['success']
        assert 'position size exceeds limit' in result['message'].lower()
    
    # Test failed order placement
    trade_agent.open_trades = []  # Reset open trades
    trade_agent.trade_size_usd = 100000  # Reset trade size limit
    
    with patch.object(trade_agent, '_place_order') as mock_place_order, \
         patch.object(trade_agent, '_is_within_risk_limits') as mock_risk_check, \
         patch.object(trade_agent, 'validate_trade_params') as mock_validate:
        mock_place_order.return_value = None  # Simulate failed order
        mock_risk_check.return_value = True
        mock_validate.return_value = True
        
        result = await trade_agent.execute_trade(trade_params)
        assert not result['success']
        assert 'failed to place order' in result['message'].lower()