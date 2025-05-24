"""
Tests for sentiment analysis module.
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime
from agents.sentiment_analyzer import SentimentAnalyzer
from tests.mock_data_agent import MockDataAgent

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'twitter_bearer_token': 'test_token',
            'reddit_client_id': 'test_id',
            'reddit_client_secret': 'test_secret',
            'news_api_key': 'test_key'
        }
        self.analyzer = SentimentAnalyzer(self.config)
        self.data_agent = MockDataAgent()
        
    @patch('tweepy.Client')
    @patch('praw.Reddit')
    def test_setup_apis(self, mock_reddit, mock_twitter):
        """Test API client setup."""
        analyzer = SentimentAnalyzer(self.config)
        
        # Verify Twitter setup
        mock_twitter.assert_called_once_with(
            bearer_token='test_token',
            wait_on_rate_limit=True
        )
        
        # Verify Reddit setup
        mock_reddit.assert_called_once_with(
            client_id='test_id',
            client_secret='test_secret',
            user_agent="Crypto Sentiment Bot 1.0"
        )
        
    @patch('agents.sentiment_analyzer.SentimentAnalyzer.analyze_social_sentiment')
    @patch('agents.sentiment_analyzer.SentimentAnalyzer.analyze_news_sentiment')
    @patch('agents.sentiment_analyzer.SentimentAnalyzer.get_trading_view_sentiment')
    async def test_get_combined_sentiment(self, mock_tv, mock_news, mock_social):
        """Test combined sentiment calculation."""
        # Mock individual sentiment results
        mock_social.return_value = {
            'score': 0.5,
            'confidence': 0.8,
            'source': 'social'
        }
        mock_news.return_value = {
            'score': -0.2,
            'confidence': 0.6,
            'source': 'news'
        }
        mock_tv.return_value = {
            'score': 0.3,
            'confidence': 0.9,
            'source': 'technical'
        }
        
        result = await self.analyzer.get_combined_sentiment('BTCUSDT')
        
        # Verify combined score calculation
        expected_score = (0.5 * 0.3) + (-0.2 * 0.3) + (0.3 * 0.4)
        self.assertAlmostEqual(result['combined_score'], expected_score)
        
        # Verify metadata
        self.assertIn('source_weights', result['metadata'])
        self.assertIn('confidence', result['metadata'])
        
    @patch('tweepy.Client')
    async def test_analyze_social_sentiment(self, mock_twitter):
        """Test social media sentiment analysis."""
        # Mock Twitter response
        mock_tweets = Mock()
        mock_tweets.data = [
            Mock(text="Bitcoin is amazing! #BTC"),
            Mock(text="Crypto markets looking bearish"),
            Mock(text="Great time to buy $BTC")
        ]
        mock_twitter.return_value.search_recent_tweets.return_value = mock_tweets
        
        # Mock Reddit posts
        mock_reddit_posts = [
            Mock(title="BTC Analysis", selftext="Positive outlook"),
            Mock(title="Bear market", selftext="Negative sentiment")
        ]
        self.analyzer.reddit.subreddit = Mock()
        self.analyzer.reddit.subreddit().search.return_value = mock_reddit_posts
        
        result = await self.analyzer.analyze_social_sentiment('BTC')
        
        # Verify sentiment calculation
        self.assertIn('score', result)
        self.assertIn('confidence', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['source'], 'social')
        
    @patch('aiohttp.ClientSession.get')
    async def test_analyze_news_sentiment(self, mock_get):
        """Test news sentiment analysis."""
        # Mock news API response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Bitcoin reaches new high',
                    'description': 'Positive market momentum'
                },
                {
                    'title': 'Crypto market analysis',
                    'description': 'Mixed signals in trading'
                }
            ]
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await self.analyzer.analyze_news_sentiment('BTC')
        
        # Verify sentiment calculation
        self.assertIn('score', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['source'], 'news')
        
    @patch('aiohttp.ClientSession.get')
    async def test_get_trading_view_sentiment(self, mock_get):
        """Test TradingView sentiment analysis."""
        # Mock TradingView API response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'oscillators': {'buy': 3, 'sell': 1},
            'moving_averages': {'buy': 5, 'sell': 2},
            'summary': {'neutral': 2}
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await self.analyzer.get_trading_view_sentiment('BTCUSDT')
        
        # Verify sentiment calculation
        self.assertIn('score', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['source'], 'technical')
        
        # Verify signal counting
        self.assertEqual(result['metadata']['buy_signals'], 8)
        self.assertEqual(result['metadata']['sell_signals'], 3)
        
    def test_calculate_confidence(self):
        """Test confidence score calculation."""
        social = {'confidence': 0.8}
        news = {'confidence': 0.6}
        technical = {'confidence': 0.9}
        
        confidence = self.analyzer._calculate_confidence(
            social, news, technical
        )
        
        # Verify confidence calculation
        expected_confidence = (0.8 + 0.6 + 0.9) / 3
        self.assertAlmostEqual(confidence, expected_confidence)
        
    async def test_error_handling(self):
        """Test error handling in sentiment analysis."""
        # Test with invalid API keys
        analyzer = SentimentAnalyzer({})
        result = await analyzer.get_combined_sentiment('BTCUSDT')
        
        # Verify graceful error handling
        self.assertEqual(result['social_sentiment']['score'], 0)
        self.assertEqual(result['news_sentiment']['score'], 0)
        self.assertEqual(result['technical_sentiment']['score'], 0)

if __name__ == '__main__':
    unittest.main() 