"""
Sentiment analysis module for crypto market sentiment.
"""
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from textblob import TextBlob
import tweepy
import praw
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes market sentiment from multiple sources:
    - Twitter/X posts
    - Reddit discussions
    - News articles
    - Trading view technical indicators
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize sentiment analyzer with API keys."""
        self.config = config
        self._setup_apis()
        
    def _setup_apis(self):
        """Set up API clients."""
        try:
            # Twitter/X API setup
            self.twitter = tweepy.Client(
                bearer_token=self.config.get('twitter_bearer_token'),
                wait_on_rate_limit=True
            )
            
            # Reddit API setup
            self.reddit = praw.Reddit(
                client_id=self.config.get('reddit_client_id'),
                client_secret=self.config.get('reddit_client_secret'),
                user_agent="Crypto Sentiment Bot 1.0"
            )
            
            # News API setup
            self.news_api_key = self.config.get('news_api_key')
            
        except Exception as e:
            logger.error(f"Error setting up APIs: {str(e)}")
            
    async def get_combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined sentiment analysis from all sources.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dict containing sentiment scores and metadata
        """
        base_asset = symbol.replace('USDT', '').replace('USD', '')
        
        # Gather sentiment from different sources concurrently
        tasks = [
            self.analyze_social_sentiment(base_asset),
            self.analyze_news_sentiment(base_asset),
            self.get_trading_view_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks)
        social_sentiment, news_sentiment, tv_sentiment = results
        
        # Combine sentiment scores with weights
        weights = {
            'social': 0.3,
            'news': 0.3,
            'technical': 0.4
        }
        
        combined_score = (
            social_sentiment['score'] * weights['social'] +
            news_sentiment['score'] * weights['news'] +
            tv_sentiment['score'] * weights['technical']
        )
        
        return {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'combined_score': combined_score,
            'social_sentiment': social_sentiment,
            'news_sentiment': news_sentiment,
            'technical_sentiment': tv_sentiment,
            'metadata': {
                'source_weights': weights,
                'confidence': self._calculate_confidence(social_sentiment, news_sentiment, tv_sentiment)
            }
        }
        
    async def analyze_social_sentiment(self, asset: str) -> Dict[str, Any]:
        """Analyze sentiment from social media."""
        try:
            # Get Twitter/X posts
            tweets = self.twitter.search_recent_tweets(
                query=f"#{asset} OR {asset} crypto",
                max_results=100
            )
            
            # Get Reddit posts
            subreddits = ['cryptocurrency', 'bitcoin', 'cryptomarkets']
            reddit_posts = []
            for sub in subreddits:
                reddit_posts.extend(
                    self.reddit.subreddit(sub).search(asset, time_filter='day', limit=50)
                )
            
            # Analyze sentiment
            tweet_sentiments = [
                TextBlob(tweet.text).sentiment.polarity
                for tweet in tweets.data if tweets.data
            ]
            
            reddit_sentiments = [
                TextBlob(post.title + " " + post.selftext).sentiment.polarity
                for post in reddit_posts
            ]
            
            # Combine scores
            all_sentiments = tweet_sentiments + reddit_sentiments
            if not all_sentiments:
                return {'score': 0, 'confidence': 0, 'source': 'social'}
                
            score = np.mean(all_sentiments)
            confidence = len(all_sentiments) / 150  # Normalize by expected max
            
            return {
                'score': float(score),
                'confidence': min(1.0, float(confidence)),
                'source': 'social',
                'metadata': {
                    'tweet_count': len(tweet_sentiments),
                    'reddit_count': len(reddit_sentiments),
                    'sentiment_std': float(np.std(all_sentiments))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {str(e)}")
            return {'score': 0, 'confidence': 0, 'source': 'social'}
            
    async def analyze_news_sentiment(self, asset: str) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        try:
            # Fetch news articles
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{asset} crypto",
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    news_data = await response.json()
            
            if 'articles' not in news_data:
                return {'score': 0, 'confidence': 0, 'source': 'news'}
                
            # Analyze sentiment of headlines and descriptions
            sentiments = []
            for article in news_data['articles']:
                text = f"{article['title']} {article['description']}"
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
                
            if not sentiments:
                return {'score': 0, 'confidence': 0, 'source': 'news'}
                
            score = np.mean(sentiments)
            confidence = len(sentiments) / 50  # Normalize by expected max
            
            return {
                'score': float(score),
                'confidence': min(1.0, float(confidence)),
                'source': 'news',
                'metadata': {
                    'article_count': len(sentiments),
                    'sentiment_std': float(np.std(sentiments))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'score': 0, 'confidence': 0, 'source': 'news'}
            
    async def get_trading_view_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get technical sentiment from TradingView indicators."""
        try:
            # TradingView's technical analysis endpoint
            url = f"https://scanner.tradingview.com/{symbol}/technical"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
            if not data:
                return {'score': 0, 'confidence': 0, 'source': 'technical'}
                
            # Extract oscillator and MA signals
            oscillators = data.get('oscillators', {})
            moving_avgs = data.get('moving_averages', {})
            
            # Calculate technical sentiment
            buy_signals = (
                oscillators.get('buy', 0) +
                moving_avgs.get('buy', 0)
            )
            sell_signals = (
                oscillators.get('sell', 0) +
                moving_avgs.get('sell', 0)
            )
            
            total_signals = buy_signals + sell_signals
            if total_signals == 0:
                return {'score': 0, 'confidence': 0, 'source': 'technical'}
                
            score = (buy_signals - sell_signals) / total_signals
            confidence = total_signals / 30  # Normalize by expected max signals
            
            return {
                'score': float(score),
                'confidence': min(1.0, float(confidence)),
                'source': 'technical',
                'metadata': {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'neutral_signals': data.get('summary', {}).get('neutral', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting technical sentiment: {str(e)}")
            return {'score': 0, 'confidence': 0, 'source': 'technical'}
            
    def _calculate_confidence(self, social: Dict[str, Any], 
                            news: Dict[str, Any], 
                            technical: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        confidences = [
            social['confidence'],
            news['confidence'],
            technical['confidence']
        ]
        return float(np.mean(confidences)) 