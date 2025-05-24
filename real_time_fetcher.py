import yaml
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def initialize_binance_client(config):
    """Initialize Binance client with API credentials"""
    try:
        api_key = config['api']['api_key']
        api_secret = config['api']['api_secret']
        
        if not api_key or not api_secret:
            raise ValueError("Please set your Binance API credentials in config.yaml")
        
        client = Client(api_key, api_secret)
        
        # Synchronize time with Binance server with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get server time
                server_time = client.get_server_time()
                local_time = int(time.time() * 1000)
                time_offset = server_time['serverTime'] - local_time
                
                # Set the timestamp offset in the client
                client.timestamp_offset = time_offset
                logger.info(f"Successfully synchronized time with Binance server. Offset: {time_offset}ms")
                
                # Test connection
                client.get_system_status()
                logger.info("Successfully connected to Binance API")
                
                return client
            except BinanceAPIException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(1)
                else:
                    raise
                    
    except Exception as e:
        logger.error(f"Error initializing Binance client: {str(e)}")
        return None

class RealTimeDataFetcher:
    def __init__(self):
        self.config = load_config()
        self.client = initialize_binance_client(self.config)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Initialize last update times
        self.last_updates = {}
        for timeframe in self.config['trading']['timeframes']:
            self.last_updates[timeframe] = 0
    
    def get_appropriate_limit(self, timeframe):
        """Get appropriate number of klines based on timeframe"""
        if timeframe in ['1m', '3m', '5m']:
            return 1000  # More data points for lower timeframes
        return 500
    
    def should_update(self, timeframe):
        """Check if we should update this timeframe"""
        now = time.time()
        update_intervals = {
            '1m': 30,      # Update every 30 seconds
            '3m': 60,      # Update every 1 minute
            '5m': 60,      # Update every 1 minute
            '15m': 120,    # Update every 2 minutes
            '30m': 300,    # Update every 5 minutes
            '1h': 600,     # Update every 10 minutes
            '2h': 1200,    # Update every 20 minutes
            '4h': 3600,    # Update every hour
            '6h': 3600,    # Update every hour
            '8h': 3600,    # Update every hour
            '12h': 7200,   # Update every 2 hours
            '1d': 14400,   # Update every 4 hours
        }
        
        interval = update_intervals.get(timeframe, 3600)  # Default to 1 hour
        return (now - self.last_updates.get(timeframe, 0)) >= interval
    
    def update_klines(self, symbol, timeframe):
        """Update klines data for a specific symbol and timeframe"""
        try:
            if not self.should_update(timeframe):
                return
            
            logger.info(f"Updating {symbol} {timeframe} klines...")
            
            # Get klines data
            klines = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=self.get_appropriate_limit(timeframe)
            )
            
            # Process klines data
            klines_data = []
            for k in klines:
                klines_data.append({
                    'timestamp': pd.to_datetime(k[0], unit='ms'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': pd.to_datetime(k[6], unit='ms'),
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                    'taker_buy_base': float(k[9]),
                    'taker_buy_quote': float(k[10])
                })
            
            # Save to CSV
            df = pd.DataFrame(klines_data)
            filename = f'data/{symbol}_{timeframe}_klines.csv'
            df.to_csv(filename, index=False)
            
            # Update last update time
            self.last_updates[timeframe] = time.time()
            
            logger.info(f"Successfully updated {symbol} {timeframe} klines")
            
        except Exception as e:
            logger.error(f"Error updating {symbol} {timeframe} klines: {str(e)}")
    
    def update_account_data(self):
        """Update account balance and trade data"""
        try:
            if not self.client:
                logger.error("Binance client not initialized")
                return
            
            # Get account information with automatic timestamp adjustment
            account = self.client.get_account()
            
            # Process balances
            balances = []
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                if free > 0 or locked > 0:
                    balances.append({
                        'asset': asset['asset'],
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    })
            
            # Save to CSV
            df = pd.DataFrame(balances)
            df.to_csv('data/account_balance.csv', index=False)
            logger.info("Successfully updated account data")
            
            # Update trade history
            for symbol in self.config['trading']['symbols']:
                trades = self.client.get_my_trades(symbol=symbol)
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_df.to_csv(f'data/{symbol}_trades.csv', index=False)
                    logger.info(f"Successfully updated {symbol} trades")
                
        except Exception as e:
            logger.error(f"Error updating account data: {str(e)}")
    
    def run(self):
        """Main run loop"""
        logger.info("Starting real-time data fetcher...")
        
        while True:
            try:
                # Update klines for all symbols and timeframes
                for symbol in self.config['trading']['symbols']:
                    for timeframe in self.config['trading']['timeframes']:
                        self.update_klines(symbol, timeframe)
                
                # Update account data
                self.update_account_data()
                
                # Sleep for 1 second before next iteration
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Stopping real-time data fetcher...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait 5 seconds before retrying

if __name__ == "__main__":
    try:
        fetcher = RealTimeDataFetcher()
        fetcher.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise 