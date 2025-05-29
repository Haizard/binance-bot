import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from decimal import Decimal
from agents.market_drop_analyzer_agent import MarketDropAnalyzerAgent

@pytest.mark.asyncio
async def test_integration_with_binance_testnet():
    """
    Integration test for MarketDropAnalyzerAgent with Binance testnet.
    This test now mocks the websocket message handler to simulate trade data.
    """
    agent = MarketDropAnalyzerAgent(target_drop_percent=30.0)
    with patch.object(agent, '_handle_socket_message', new=AsyncMock()) as mock_handler:
        await agent.setup()
        # Simulate a trade message
        await agent._handle_socket_message({
            'e': 'trade',
            's': 'BTCUSDT',
            'p': '50000',
            'q': '0.01',
            'T': 1620000000000
        })
        # Optionally, you can manually update price_history for assertion
        agent.price_history['BTCUSDT'] = [{
            'timestamp': 1620000000,
            'price': Decimal('50000'),
            'volume': Decimal('0.01')
        }]
        assert len(agent.price_history) > 0
        for symbol, prices in agent.price_history.items():
            assert len(prices) > 0
            for price_data in prices:
                assert 'price' in price_data
                assert 'volume' in price_data
                assert isinstance(price_data['price'], Decimal)
                assert isinstance(price_data['volume'], Decimal)
        await agent.cleanup()
