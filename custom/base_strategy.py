"""
Base strategy module for trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base strategy."""
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and return trading signals.
        
        Args:
            data: Market data dictionary containing price, volume, etc.
            
        Returns:
            Dictionary containing analysis results and trading signals.
        """
        pass
        
    @abstractmethod
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Trading signal to validate.
            
        Returns:
            True if signal is valid, False otherwise.
        """
        pass
        
    def get_name(self) -> str:
        """Get strategy name."""
        return self.name
        
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.config
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update strategy configuration."""
        self.config.update(new_config) 