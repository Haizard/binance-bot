"""
Base Agent class that provides common functionality for all trading bot agents.
"""
import asyncio
import json
import logging
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents in the trading bot system.
    Provides common functionality and required interface for agent communication.
    """
    def __init__(self, name: str, message_broker: Any = None):
        """
        Initialize the base agent.
        
        Args:
            name (str): Unique name identifier for the agent
            message_broker: Message broker instance for inter-agent communication
        """
        self.name = name
        self.message_broker = message_broker
        self.running = False
        self.subscriptions = set()
        self._message_queue = asyncio.Queue()
        logger.info(f"Initialized {self.name} agent")

    async def start(self) -> None:
        """Start the agent's main processing loop."""
        if self.running:
            logger.warning(f"{self.name} agent is already running")
            return
        
        self.running = True
        logger.info(f"Starting {self.name} agent")
        await self.setup()
        
        try:
            await self._run_loop()
        except Exception as e:
            logger.error(f"Error in {self.name} agent: {str(e)}")
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the agent's processing loop."""
        logger.info(f"Stopping {self.name} agent")
        self.running = False
        await self.cleanup()

    @abstractmethod
    async def setup(self) -> None:
        """
        Setup method to be implemented by each agent.
        Called when the agent starts.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup method to be implemented by each agent.
        Called when the agent stops.
        """
        pass

    async def send_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Send a message to other agents through the message broker.
        
        Args:
            topic (str): Message topic/channel
            message (dict): Message content
        """
        if self.message_broker:
            try:
                await self.message_broker.publish(topic, {
                    'sender': self.name,
                    'data': message
                })
                logger.debug(f"{self.name} sent message on topic {topic}")
            except Exception as e:
                logger.error(f"Error sending message from {self.name}: {str(e)}")

    async def receive_message(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Receive a message from the message queue.
        
        Args:
            timeout (float, optional): Maximum time to wait for a message
            
        Returns:
            dict: Received message or None if timeout
        """
        try:
            message = await asyncio.wait_for(self._message_queue.get(), timeout)
            return message
        except asyncio.TimeoutError:
            return None

    async def subscribe(self, topic: str) -> None:
        """
        Subscribe to a message topic.
        
        Args:
            topic (str): Topic to subscribe to
        """
        if self.message_broker:
            await self.message_broker.subscribe(topic, self._handle_message)
            self.subscriptions.add(topic)
            logger.debug(f"{self.name} subscribed to {topic}")

    async def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a message topic.
        
        Args:
            topic (str): Topic to unsubscribe from
        """
        if self.message_broker and topic in self.subscriptions:
            await self.message_broker.unsubscribe(topic, self._handle_message)
            self.subscriptions.remove(topic)
            logger.debug(f"{self.name} unsubscribed from {topic}")

    async def _handle_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Handle incoming messages from subscribed topics.
        
        Args:
            topic (str): Message topic
            message (dict): Message content
        """
        await self._message_queue.put({
            'topic': topic,
            'message': message
        })

    async def _run_loop(self) -> None:
        """Main processing loop for the agent."""
        while self.running:
            try:
                message = await self.receive_message(timeout=1.0)
                if message:
                    await self.process_message(message)
            except Exception as e:
                logger.error(f"Error in {self.name} processing loop: {str(e)}")
                continue

    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process received messages. To be implemented by each agent.
        
        Args:
            message (dict): Message to process
        """
        pass

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} Agent" 