import yaml
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime, timedelta
import os
import time

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def initialize_binance_client(config):
    """Initialize Binance client with API credentials"""
    api_key = config['api']['api_key']
    api_secret = config['api']['api_secret']
    
    if not api_key or not api_secret:
        raise ValueError("Please set your Binance API credentials in config.yaml")
    
    client = Client(api_key, api_secret)
    
    # Synchronize time with Binance server
    server_time = client.get_server_time()
    local_time = int(time.time() * 1000)
    time_offset = server_time['serverTime'] - local_time
    client.timestamp_offset = time_offset
    
    return client

def fetch_historical_trades(client, symbol, start_time=None):
    """Fetch historical trades from Binance"""
    if start_time is None:
        start_time = datetime.now() - timedelta(days=30)  # Last 30 days by default
    
    try:
        trades = client.get_my_trades(symbol=symbol, startTime=int(start_time.timestamp() * 1000))
        
        trade_data = []
        for trade in trades:
            trade_data.append({
                'id': trade['id'],
                'symbol': trade['symbol'],
                'side': trade['isBuyer'] and 'BUY' or 'SELL',
                'price': float(trade['price']),
                'quantity': float(trade['qty']),
                'commission': float(trade['commission']),
                'commission_asset': trade['commissionAsset'],
                'time': pd.to_datetime(trade['time'], unit='ms'),
                'quote_qty': float(trade['quoteQty']),
                'realized_pnl': float(trade.get('realizedPnl', 0))
            })
        
        return pd.DataFrame(trade_data)
    except BinanceAPIException as e:
        print(f"Error fetching trades for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_account_balance(client):
    """Fetch current account balance"""
    try:
        account = client.get_account()
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
        
        return pd.DataFrame(balances)
    except BinanceAPIException as e:
        print(f"Error fetching account balance: {str(e)}")
        return pd.DataFrame()

def fetch_open_orders(client, symbol=None):
    """Fetch all open orders"""
    try:
        orders = client.get_open_orders(symbol=symbol)
        
        order_data = []
        for order in orders:
            order_data.append({
                'id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'price': float(order['price']),
                'original_quantity': float(order['origQty']),
                'executed_quantity': float(order['executedQty']),
                'status': order['status'],
                'time': pd.to_datetime(order['time'], unit='ms')
            })
        
        return pd.DataFrame(order_data)
    except BinanceAPIException as e:
        print(f"Error fetching open orders for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_klines_data(client, symbol, interval='1d', limit=500):
    """Fetch klines (candlestick) data"""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
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
        
        return pd.DataFrame(klines_data)
    except BinanceAPIException as e:
        print(f"Error fetching klines data for {symbol}: {str(e)}")
        return pd.DataFrame()

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize Binance client with time synchronization
        print("Initializing Binance client and synchronizing time...")
        client = initialize_binance_client(config)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Fetch data for configured symbols
        for symbol in config['trading']['symbols']:
            print(f"\nFetching data for {symbol}...")
            
            # Fetch historical trades
            print("Fetching historical trades...")
            trades_df = fetch_historical_trades(client, symbol)
            if not trades_df.empty:
                trades_df.to_csv(f'data/{symbol}_trades.csv', index=False)
                print(f"Saved {len(trades_df)} trades")
            
            # Fetch open orders
            print("Fetching open orders...")
            orders_df = fetch_open_orders(client, symbol)
            if not orders_df.empty:
                orders_df.to_csv(f'data/{symbol}_open_orders.csv', index=False)
                print(f"Saved {len(orders_df)} open orders")
            
            # Fetch klines data for different timeframes
            for timeframe in config['trading']['timeframes']:
                print(f"Fetching {timeframe} klines data...")
                klines_df = fetch_klines_data(client, symbol, interval=timeframe)
                if not klines_df.empty:
                    klines_df.to_csv(f'data/{symbol}_{timeframe}_klines.csv', index=False)
                    print(f"Saved {len(klines_df)} klines")
        
        # Fetch account balance
        print("\nFetching account balance...")
        balance_df = fetch_account_balance(client)
        if not balance_df.empty:
            balance_df.to_csv('data/account_balance.csv', index=False)
            print(f"Saved balance data for {len(balance_df)} assets")
        
        print("\nData fetching completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 