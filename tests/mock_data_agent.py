"""
Mock data agent for testing.
"""
from typing import Dict, Any, List
from datetime import datetime

class MockDataAgent:
    """Mock data agent that simulates DataAgent functionality."""
    
    def __init__(self):
        """Initialize mock data agent."""
        self.price_data = {}
        self.trades = []
        
    def get_current_price(self, symbol: str) -> float:
        """Get mock current price."""
        return self.price_data.get(symbol, {'price': 50000.0})['price']
        
    def get_price_history(self, symbol: str, interval: str = '1h',
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get mock price history."""
        return [
            {
                'timestamp': datetime.now(),
                'price': 50000.0,
                'volume': 1.5
            }
        ]
        
    def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get mock trade history."""
        return self.trades
        
    def set_mock_price(self, symbol: str, price: float):
        """Set mock price for testing."""
        self.price_data[symbol] = {'price': price}
        
    def add_mock_trade(self, trade: Dict[str, Any]):
        """Add mock trade for testing."""
        self.trades.append(trade) 