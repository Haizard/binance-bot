"""
Message broker implementation for inter-agent communication.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Set

logger = logging.getLogger(__name__)

class MessageBroker:
    """
    A simple pub/sub message broker for inter-agent communication.
    Handles message routing between agents using an in-memory event system.
    """
    def __init__(self):
        """Initialize the message broker."""
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._message_queue = asyncio.Queue()
        self.running = False
        logger.info("Message broker initialized")

    async def start(self) -> None:
        """Start the message broker's processing loop."""
        if self.running:
            logger.warning("Message broker is already running")
            return

        self.running = True
        logger.info("Starting message broker")
        
        try:
            await self._process_messages()
        except Exception as e:
            logger.error(f"Error in message broker: {str(e)}")
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the message broker's processing loop."""
        logger.info("Stopping message broker")
        self.running = False

    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Publish a message to a topic.
        
        Args:
            topic (str): Topic to publish to
            message (dict): Message content
        """
        if topic in self._subscribers:
            try:
                await self._message_queue.put({
                    'topic': topic,
                    'message': message
                })
                logger.debug(f"Message queued for topic {topic}")
            except Exception as e:
                logger.error(f"Error publishing message to {topic}: {str(e)}")

    async def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Subscribe to a topic with a callback function.
        
        Args:
            topic (str): Topic to subscribe to
            callback (callable): Function to call when message is received
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = set()
        self._subscribers[topic].add(callback)
        logger.debug(f"New subscription to topic {topic}")

    async def unsubscribe(self, topic: str, callback: Callable) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic (str): Topic to unsubscribe from
            callback (callable): Callback function to remove
        """
        if topic in self._subscribers and callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)
            if not self._subscribers[topic]:
                del self._subscribers[topic]
            logger.debug(f"Unsubscribed from topic {topic}")

    async def _process_messages(self) -> None:
        """Process messages in the queue and distribute to subscribers."""
        while self.running:
            try:
                message_data = await self._message_queue.get()
                topic = message_data['topic']
                message = message_data['message']

                if topic in self._subscribers:
                    for callback in self._subscribers[topic]:
                        try:
                            await callback(topic, message)
                        except Exception as e:
                            logger.error(f"Error in subscriber callback: {str(e)}")
                            continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                continue

    def get_topics(self) -> List[str]:
        """
        Get list of active topics.
        
        Returns:
            list: List of active topics
        """
        return list(self._subscribers.keys())

    def get_subscriber_count(self, topic: str) -> int:
        """
        Get number of subscribers for a topic.
        
        Args:
            topic (str): Topic to check
            
        Returns:
            int: Number of subscribers
        """
        return len(self._subscribers.get(topic, set())) 