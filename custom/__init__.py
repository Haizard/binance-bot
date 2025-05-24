"""
🌙 Moon Dev's Custom Strategies
Built with love by Moon Dev 🚀
"""

from src.strategies.base_strategy import BaseStrategy

# Import built-in example strategy
from src.strategies.example_strategy import ExampleStrategy

# List of available strategies
AVAILABLE_STRATEGIES = {
    'example': ExampleStrategy
}

def get_strategy(name: str) -> BaseStrategy:
    """Get a strategy by name
    
    Args:
        name: Name of the strategy to get
        
    Returns:
        BaseStrategy: The requested strategy instance
        
    Raises:
        ValueError: If strategy not found
    """
    if name not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Strategy '{name}' not found. Available strategies: {list(AVAILABLE_STRATEGIES.keys())}")
    
    return AVAILABLE_STRATEGIES[name]()

__all__ = ['get_strategy', 'AVAILABLE_STRATEGIES']