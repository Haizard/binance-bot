"""
Main entry point for the trading bot.
Implements an agent-based architecture for modular and maintainable trading operations.
"""
import os
import sys
import logging
import asyncio
import argparse
from typing import List
from agents import BaseAgent, MessageBroker, ConfigAgent, DataAgent, AnalysisAgent, TradeAgent, PerformanceAgent
import traceback
import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBotOrchestrator:
    """
    Main orchestrator class that manages all trading bot agents.
    """
    def __init__(self):
        """Initialize the trading bot orchestrator."""
        self.message_broker = MessageBroker()
        self.agents: List[BaseAgent] = []
        self.running = False

    async def initialize(self) -> None:
        """Initialize all agents and systems."""
        try:
            # Start message broker
            logger.info("Starting message broker...")
            asyncio.create_task(self.message_broker.start())

            # Initialize config agent first
            logger.info("Initializing ConfigAgent...")
            config_agent = ConfigAgent(self.message_broker)
            self.agents.append(config_agent)

            # Initialize data agent
            logger.info("Initializing DataAgent...")
            data_agent = DataAgent(self.message_broker)
            self.agents.append(data_agent)

            # Initialize analysis agent
            logger.info("Initializing AnalysisAgent...")
            analysis_agent = AnalysisAgent(self.message_broker)
            self.agents.append(analysis_agent)

            # Initialize trade agent
            logger.info("Initializing TradeAgent...")
            trade_agent = TradeAgent(self.message_broker)
            self.agents.append(trade_agent)

            # Initialize performance agent
            logger.info("Initializing PerformanceAgent...")
            performance_agent = PerformanceAgent(self.message_broker)
            self.agents.append(performance_agent)

            logger.info("All agents initialized")

        except Exception as e:
            logger.error(f"Error initializing trading bot: {str(e)}\n{traceback.format_exc()}")
            raise

    async def start(self) -> None:
        """Start all agents and begin trading operations."""
        if self.running:
            logger.warning("Trading bot is already running")
            return

        try:
            self.running = True
            logger.info("Starting trading bot...")

            # Start all agents
            agent_tasks = [
                asyncio.create_task(agent.start())
                for agent in self.agents
            ]

            # Wait for all agents to complete
            await asyncio.gather(*agent_tasks)

        except Exception as e:
            logger.error(f"Error running trading bot: {str(e)}\n{traceback.format_exc()}")
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop all agents and cleanup."""
        logger.info("Stopping trading bot...")
        self.running = False

        # Stop all agents in reverse order
        for agent in reversed(self.agents):
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping {agent.name}: {str(e)}")

        # Stop message broker
        try:
            await self.message_broker.stop()
        except Exception as e:
            logger.error(f"Error stopping message broker: {str(e)}")

        logger.info("Trading bot stopped")

async def main() -> None:
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--config', type=str, default='config.ini', help='Path to config file')
    args = parser.parse_args()

    try:
        # Create and initialize the trading bot
        bot = TradingBotOrchestrator()
        await bot.initialize()

        # Handle shutdown gracefully
        def signal_handler():
            asyncio.create_task(bot.stop())

        # Register signal handlers only if not on Windows
        if sys.platform != "win32":
            for sig in [signal.SIGINT, signal.SIGTERM]:
                asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

        # Start the bot
        await bot.start()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Ensure cleanup
        await bot.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
