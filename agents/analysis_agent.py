"""
Market data analysis and signal generation agent implementation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from custom.strategy_factory import StrategyFactory
from status_manager import StatusManager
import asyncio

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """
    Agent responsible for analyzing market data and generating trading signals.
    Uses custom strategies for analysis and signal generation.
    """
    def __init__(self, message_broker: Any = None):
        """
        Initialize the AnalysisAgent.
        
        Args:
            message_broker: Message broker instance for inter-agent communication
        """
        super().__init__("Analysis", message_broker)
        self.data_cache = {}  # Cache for market data
        self.analysis_config = {}  # Configuration for analysis parameters
        self.active_symbols = set()  # Set of symbols being analyzed
        self.strategies = {}  # Active strategies

    async def setup(self) -> None:
        """Set up the analysis agent."""
        # Subscribe to necessary topics
        await self.subscribe("config.update")
        await self.subscribe("data.historical.response.Analysis")
        
        # Subscribe to market data updates dynamically based on symbols
        
        # Request initial configuration
        await self.send_message("config.get.request", {
            'sender': self.name,
            'include_keys': False
        })
        
        StatusManager().update("Analysis", {"message": "Setup completed, waiting for data"})
        logger.info("AnalysisAgent setup completed")
        # Start heartbeat
        asyncio.create_task(self._heartbeat())

    async def _heartbeat(self):
        while True:
            # Show which symbols are being analyzed and last signal if available
            symbol_status = []
            for symbol in self.active_symbols:
                last = self.data_cache.get(symbol, [])
                if last:
                    last_data = last[-1]
                    symbol_status.append(f"{symbol}")
            msg = "AnalysisAgent alive"
            if symbol_status:
                msg += " | Analyzing: " + ", ".join(symbol_status)
            StatusManager().update("Analysis", {"message": msg})
            await asyncio.sleep(5)

    async def cleanup(self) -> None:
        """Clean up the analysis agent."""
        # Unsubscribe from all topics
        for topic in list(self.subscriptions):
            await self.unsubscribe(topic)
        
        # Clear data caches
        self.data_cache.clear()
        self.active_symbols.clear()
        self.strategies.clear()
        
        logger.info("AnalysisAgent cleanup completed")

    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming messages.
        
        Args:
            message (dict): Message to process
        """
        topic = message['topic']
        data = message['message'].get('data', {})
        
        if topic == "config.update":
            await self._handle_config_update(data)
        elif topic.startswith("data.update."):
            await self._handle_market_data_update(topic.split('.')[-1], data)
        elif topic == "data.historical.response.Analysis":
            await self._handle_historical_data(data)

    async def _handle_config_update(self, data: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        config = data.get('config', {})
        
        # Update analysis parameters
        analysis_config = config.get('Analysis', {})
        if analysis_config:
            self.analysis_config = analysis_config
            
            # Update strategies based on configuration
            await self._update_strategies()
        
        # Handle trading pair updates - iterate through symbols list
        trading_config = config.get('Trading', {}) or config.get('trading', {})
        if trading_config:
            symbols = trading_config.get('symbols', [])
            for symbol in symbols:
                if symbol and symbol not in self.active_symbols:
                    await self._subscribe_to_symbol(symbol)

    async def _handle_market_data_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Handle real-time market data updates.
        
        Args:
            symbol (str): Trading pair symbol
            data (dict): Market data update
        """
        if not data:
            return
            
        try:
            StatusManager().update("Analysis", {"message": f"Analyzing {symbol} market data..."})
            # Update data cache
            if symbol not in self.data_cache:
                self.data_cache[symbol] = []
            
            self.data_cache[symbol].append(data)
            logger.debug(f"[AnalysisAgent] Added data for {symbol}. Cache size: {len(self.data_cache[symbol])}")
            
            # Maintain cache size
            max_cache_size = self.analysis_config.get('max_cache_size', 100)
            if len(self.data_cache[symbol]) > max_cache_size:
                self.data_cache[symbol] = self.data_cache[symbol][-max_cache_size:]
            
            # Perform analysis
            analysis_result = await self._analyze_market_data(symbol)
            if analysis_result:
                # Show strategies and signals
                strategies = analysis_result.get('strategy_signals', [])
                strat_msgs = []
                for strat in strategies:
                    strat_msgs.append(f"{strat['strategy']}: {strat['signal']}")
                strat_msg = ", ".join(strat_msgs)
                StatusManager().update("Analysis", {"message": f"Signal: {analysis_result['combined_signal']['signal']} for {symbol} | {strat_msg}"})
                logger.info(f"[AnalysisAgent] Generated signal for {symbol}: {analysis_result['combined_signal']['signal']}")
                logger.debug(f"[AnalysisAgent] Strategy signals for {symbol}: {strat_msg}")
                # Broadcast analysis results
                await self.send_message(f"analysis.update.{symbol}", {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis_result
                })
                StatusManager().update("Analysis", {"message": f"Analysis done for {symbol} | {strat_msg}"})
                
        except Exception as e:
            StatusManager().update("Analysis", {"message": f"Error analyzing {symbol}: {str(e)}"})
            logger.error(f"Error processing market data for {symbol}: {str(e)}")

    async def _handle_historical_data(self, data: Dict[str, Any]) -> None:
        """Handle historical data response."""
        symbol = data.get('symbol')
        historical_data = data.get('data', [])
        
        if not symbol or not historical_data:
            return
            
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(historical_data)
            logger.debug(f"[AnalysisAgent] Received historical data for {symbol}. Rows: {len(df)}")
            
            # Perform historical analysis
            analysis_result = await self._analyze_historical_data(symbol, df)
            if analysis_result:
                logger.info(f"[AnalysisAgent] Analyzed historical data for {symbol}.")
            else:
                logger.warning(f"[AnalysisAgent] Historical data analysis for {symbol} returned no result.")
            
            # Broadcast historical analysis results
            await self.send_message(f"analysis.historical.{symbol}", {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_result
            })
            
        except Exception as e:
            logger.error(f"Error analyzing historical data for {symbol}: {str(e)}")

    async def _subscribe_to_symbol(self, symbol: str) -> None:
        """
        Subscribe to market data for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
        """
        if symbol in self.active_symbols:
            return
            
        try:
            # Subscribe to market data updates
            await self.subscribe(f"data.update.{symbol}")
            self.active_symbols.add(symbol)
            
            # Request historical data for initial analysis
            await self.send_message("data.historical.request", {
                'symbol': symbol,
                'sender': self.name,
                'limit': self.analysis_config.get('historical_data_limit', 100)
            })
            
            logger.info(f"Subscribed to {symbol} market data")
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {str(e)}")

    async def _update_strategies(self) -> None:
        """Update strategies based on configuration."""
        try:
            # Get configured strategies
            strategy_configs = self.analysis_config.get('strategies', {})
            
            # Clear existing strategies
            self.strategies.clear()
            
            # Initialize each configured strategy
            for strategy_name, config in strategy_configs.items():
                try:
                    # Create strategy instance with configuration
                    strategy = StrategyFactory.create_strategy(
                        strategy_name,
                        **config.get('parameters', {})
                    )
                    self.strategies[strategy_name] = {
                        'instance': strategy,
                        'weight': config.get('weight', 1.0),
                        'enabled': config.get('enabled', True)
                    }
                    StatusManager().update("Analysis", {"message": f"Initialized strategy: {strategy_name}"})
                    logger.info(f"Initialized strategy: {strategy_name}")
                    
                except Exception as e:
                    StatusManager().update("Analysis", {"message": f"Error initializing strategy {strategy_name}: {str(e)}"})
                    logger.error(f"Error initializing strategy {strategy_name}: {str(e)}")
                    
        except Exception as e:
            StatusManager().update("Analysis", {"message": f"Error updating strategies: {str(e)}"})
            logger.error(f"Error updating strategies: {str(e)}")

    async def _analyze_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze real-time market data using configured strategies.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Analysis results including signals and metrics
        """
        if symbol not in self.data_cache or not self.data_cache[symbol]:
            return None
            
        try:
            # Convert cached data to DataFrame
            df = pd.DataFrame(self.data_cache[symbol])
            
            # Get analysis results from each strategy
            analysis_results = []
            total_weight = 0
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    result = strategy['instance'].generate_signals(df)
                    weight = strategy['weight']
                    
                    analysis_results.append({
                        'strategy': strategy_name,
                        'signal': result['signal'],
                        'weight': weight,
                        'metrics': result.get('metrics', {}),
                        'metadata': {
                            'signal_changed': result.get('signal_changed', False),
                            'confidence': result.get('confidence', 1.0)
                        }
                    })
                    
                    total_weight += weight
                    
                except Exception as e:
                    logger.error(f"Error in strategy {strategy_name}: {str(e)}")
                    
            if not analysis_results:
                return None
                
            # Combine signals from all strategies
            combined_signal = self._combine_signals(analysis_results, total_weight)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'combined_signal': combined_signal,
                'strategy_signals': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market data for {symbol}: {str(e)}")
            return None

    async def _analyze_historical_data(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze historical market data using configured strategies.
        
        Args:
            symbol (str): Trading pair symbol
            df (pd.DataFrame): Historical market data
            
        Returns:
            dict: Analysis results including signals and metrics
        """
        if df.empty:
            return None
            
        try:
            analysis_results = []
            total_weight = 0
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    result = strategy['instance'].generate_signals(df)
                    weight = strategy['weight']
                    
                    analysis_results.append({
                        'strategy': strategy_name,
                        'signal': result['signal'],
                        'weight': weight,
                        'metrics': result.get('metrics', {}),
                        'metadata': {
                            'signal_changed': result.get('signal_changed', False),
                            'confidence': result.get('confidence', 1.0)
                        }
                    })
                    
                    total_weight += weight
                    
                except Exception as e:
                    logger.error(f"Error in strategy {strategy_name}: {str(e)}")
                    
            if not analysis_results:
                return None
                
            # Combine signals from all strategies
            combined_signal = self._combine_signals(analysis_results, total_weight)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'combined_signal': combined_signal,
                'strategy_signals': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical data for {symbol}: {str(e)}")
            return None

    def _combine_signals(self, signals: List[Dict[str, Any]], total_weight: float) -> Dict[str, Any]:
        """
        Combine signals from multiple strategies using weighted average.
        
        Args:
            signals (list): List of strategy signals and weights
            total_weight (float): Sum of all strategy weights
            
        Returns:
            dict: Combined signal information
        """
        if not signals or total_weight == 0:
            return {
                'signal': 0,
                'confidence': 0,
                'weighted_signal': 0
            }
            
        weighted_signal = 0
        total_confidence = 0
        
        for signal_info in signals:
            weight = signal_info['weight']
            signal = signal_info['signal']
            confidence = signal_info['metadata'].get('confidence', 1.0)
            
            # Calculate weighted contribution
            weighted_signal += (signal * weight * confidence)
            total_confidence += (weight * confidence)
            
        # Normalize weighted signal
        normalized_signal = weighted_signal / total_weight
        
        # Calculate average confidence
        avg_confidence = total_confidence / total_weight
        
        # Determine final signal (-1, 0, or 1)
        if normalized_signal > 0.2:  # Bullish threshold
            final_signal = 1
        elif normalized_signal < -0.2:  # Bearish threshold
            final_signal = -1
        else:
            final_signal = 0  # Neutral
            
        return {
            'signal': final_signal,
            'confidence': avg_confidence,
            'weighted_signal': normalized_signal
        } 