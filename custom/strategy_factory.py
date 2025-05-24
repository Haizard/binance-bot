"""
🌙 Moon Dev's Strategy Factory
Factory for creating strategy instances from custom strategy files
"""

import os
import sys
import importlib.util
from typing import Dict, List, Union, Optional, Type, Any
from pathlib import Path
from termcolor import cprint
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from src.strategies.python_base_strategy import PythonBaseStrategy
from custom.strategy_adapter import (
    StrategyAdapter,
    DirectionalChangeStrategy,
    RSIPCAStrategy,
    VolatilityHawkesStrategy,
    TrendLineStrategy,
    HeadShouldersStrategy,
    MarketProfileStrategy,
    HarmonicPatternsStrategy,
    FlagsPennantsStrategy,
    PipPatternMinerStrategy,
    MarketStructureStrategy,
    MeanReversionStrategy,
    VolatilityStrategy,
    TrendFollowingStrategy,
    TVLIndicatorStrategy,
    IntramarketDifferenceStrategy,
    PermutationEntropyStrategy,
    VSAStrategy,
    RollingWindowStrategy,
    PIPStrategy,
    WFPIPMinerStrategy
)

# Strategy directories to search in
STRATEGY_DIRS = [
    "custom",
    "examples"
]

# Files that should not be treated as strategy modules
EXCLUDED_FILES = {
    '__init__.py',
    'strategy_adapter.py',
    'strategy_factory.py',
    'test_',  # Any file starting with test_
    'example_',  # Any file starting with example_
    'utils.py',
    'config.py',
    'setup.py',
    'requirements.txt',
    'README.md',
    '.gitignore',
    'LICENSE',
}

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, **kwargs):
        self.parameters = kwargs
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals from market data."""
        pass
        
    @abstractmethod
    def get_weight(self) -> float:
        """Get the strategy weight for signal combination."""
        pass

class MovingAverageCrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20, weight: float = 1.0):
        super().__init__(fast_period=fast_period, slow_period=slow_period, weight=weight)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        
        # Calculate moving averages
        fast_ma = SMAIndicator(close=df['close'], window=self.parameters['fast_period'])
        slow_ma = SMAIndicator(close=df['close'], window=self.parameters['slow_period'])
        
        df['fast_ma'] = fast_ma.sma_indicator()
        df['slow_ma'] = slow_ma.sma_indicator()
        
        # Generate signals
        df['signal'] = np.where(df['fast_ma'] > df['slow_ma'], 1, -1)
        
        # Calculate additional metrics
        current_signal = df['signal'].iloc[-1]
        signal_changed = df['signal'].iloc[-1] != df['signal'].iloc[-2]
        
        return {
            'signal': current_signal,
            'signal_changed': signal_changed,
            'metrics': {
                'fast_ma': df['fast_ma'].iloc[-1],
                'slow_ma': df['slow_ma'].iloc[-1]
            }
        }
        
    def get_weight(self) -> float:
        return self.parameters.get('weight', 1.0)

class RSIStrategy(BaseStrategy):
    """Relative Strength Index Strategy."""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30, weight: float = 1.0):
        super().__init__(period=period, overbought=overbought, oversold=oversold, weight=weight)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        
        # Calculate RSI
        rsi = RSIIndicator(close=df['close'], window=self.parameters['period'])
        df['rsi'] = rsi.rsi()
        
        # Generate signals
        current_rsi = df['rsi'].iloc[-1]
        signal = 0
        if current_rsi > self.parameters['overbought']:
            signal = -1  # Sell signal
        elif current_rsi < self.parameters['oversold']:
            signal = 1   # Buy signal
            
        return {
            'signal': signal,
            'metrics': {
                'rsi': current_rsi,
                'overbought': self.parameters['overbought'],
                'oversold': self.parameters['oversold']
            }
        }
        
    def get_weight(self) -> float:
        return self.parameters.get('weight', 1.0)

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, weight: float = 1.0):
        super().__init__(period=period, std_dev=std_dev, weight=weight)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        
        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['close'], window=self.parameters['period'], window_dev=self.parameters['std_dev'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        
        # Generate signals
        current_price = df['close'].iloc[-1]
        signal = 0
        if current_price > df['bb_upper'].iloc[-1]:
            signal = -1  # Sell signal (overbought)
        elif current_price < df['bb_lower'].iloc[-1]:
            signal = 1   # Buy signal (oversold)
            
        return {
            'signal': signal,
            'metrics': {
                'price': current_price,
                'upper_band': df['bb_upper'].iloc[-1],
                'lower_band': df['bb_lower'].iloc[-1],
                'middle_band': df['bb_middle'].iloc[-1]
            }
        }
        
    def get_weight(self) -> float:
        return self.parameters.get('weight', 1.0)

class StrategyFactory:
    """Factory for creating strategy instances from custom strategy files

    This factory provides methods for:
    1. Creating strategy instances from built-in adapters
    2. Loading strategy classes from Python files
    3. Getting a list of available strategies
    """

    _strategies = {
        # Basic technical analysis strategies
        'moving_average_cross': MovingAverageCrossStrategy,
        'rsi': RSIStrategy,
        'bollinger_bands': BollingerBandsStrategy,
        
        # Advanced pattern recognition strategies
        'directional_change': DirectionalChangeStrategy,
        'rsi_pca': RSIPCAStrategy,
        'volatility_hawkes': VolatilityHawkesStrategy,
        'trendline': TrendLineStrategy,
        'head_shoulders': HeadShouldersStrategy,
        'market_profile': MarketProfileStrategy,
        'harmonic_patterns': HarmonicPatternsStrategy,
        'flags_pennants': FlagsPennantsStrategy,
        
        # Market structure and analysis strategies
        'pip_pattern_miner': PipPatternMinerStrategy,
        'market_structure': MarketStructureStrategy,
        'mean_reversion': MeanReversionStrategy,
        'volatility': VolatilityStrategy,
        'trend_following': TrendFollowingStrategy,
        'tvl_indicator': TVLIndicatorStrategy,
        
        # Advanced analysis strategies
        'intramarket_difference': IntramarketDifferenceStrategy,
        'permutation_entropy': PermutationEntropyStrategy,
        'vsa': VSAStrategy,
        'rolling_window': RollingWindowStrategy,
        'pip': PIPStrategy,
        'wf_pip_miner': WFPIPMinerStrategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> Union[BaseStrategy, PythonBaseStrategy]:
        """Create a strategy instance by name

        Args:
            strategy_name: Name of the strategy to create
            **kwargs: Additional arguments to pass to the strategy constructor

        Returns:
            Strategy instance
        """
        # First try built-in strategies
        if strategy_name in cls._strategies:
            strategy_class = cls._strategies[strategy_name]
            return strategy_class(**kwargs)
            
        # If not found in built-ins, try loading from file
        strategy_class = cls.load_strategy_class(strategy_name)
        if strategy_class:
            return strategy_class(**kwargs)
            
        raise ValueError(f"Strategy '{strategy_name}' not found")

    @staticmethod
    def should_load_file(filename: str) -> bool:
        """Check if a file should be loaded as a strategy

        Args:
            filename: Name of the file to check

        Returns:
            bool: True if file should be loaded, False otherwise
        """
        # Check against excluded files
        for excluded in EXCLUDED_FILES:
            if filename.startswith(excluded) or filename == excluded:
                return False
                
        # Must be a .py file
        if not filename.endswith('.py'):
            return False
            
        # Don't load files in __pycache__ or venv
        if '__pycache__' in filename or 'venv' in filename:
            return False
            
        return True

    @staticmethod
    def load_strategy_class(strategy_name: str) -> Optional[Type[PythonBaseStrategy]]:
        """Load a strategy class from a Python file

        Args:
            strategy_name: Name of the strategy to load

        Returns:
            Strategy class or None if not found
        """
        try:
            # Check each strategy directory
            for strategy_dir in STRATEGY_DIRS:
                strategy_path = os.path.join(project_root, strategy_dir)
                
                # Try with .py extension
                if not strategy_name.endswith(".py"):
                    strategy_name += ".py"

                # Check in directory
                file_path = os.path.join(strategy_path, strategy_name)
                if os.path.exists(file_path):
                    # Check if we should load this file
                    if not StrategyFactory.should_load_file(os.path.basename(file_path)):
                        continue

                    # Load module
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find strategy class
                    strategy_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, PythonBaseStrategy) and attr != PythonBaseStrategy:
                            strategy_class = attr
                            break

                    if strategy_class:
                        return strategy_class

            return None

        except Exception as e:
            cprint(f"⚠️ Error loading strategy '{strategy_name}': {str(e)}", "yellow")
            return None

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """Register a new strategy class."""
        if not (issubclass(strategy_class, BaseStrategy) or issubclass(strategy_class, PythonBaseStrategy)):
            raise ValueError("Strategy must inherit from BaseStrategy or PythonBaseStrategy")
        cls._strategies[name] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        strategies = list(cls._strategies.keys())
        
        # Add strategies from files
        for strategy_dir in STRATEGY_DIRS:
            strategy_path = os.path.join(project_root, strategy_dir)
            if os.path.exists(strategy_path):
                for file in os.listdir(strategy_path):
                    if cls.should_load_file(file):
                        strategy_name = os.path.splitext(file)[0]
                        if strategy_name not in strategies:
                            strategies.append(strategy_name)
        
        return strategies

    @classmethod
    def create_all_strategies(cls, **kwargs) -> Dict[str, Union[BaseStrategy, PythonBaseStrategy]]:
        """Create instances of all available strategies

        Args:
            **kwargs: Default parameters to pass to all strategies

        Returns:
            Dictionary of strategy name to strategy instance
        """
        strategies = {}
        for strategy_name in cls.get_available_strategies():
            try:
                strategies[strategy_name] = cls.create_strategy(strategy_name, **kwargs)
            except Exception as e:
                cprint(f"Error creating strategy '{strategy_name}': {str(e)}", "red")

        return strategies
