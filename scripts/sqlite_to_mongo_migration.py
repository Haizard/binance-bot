import sqlite3
from pymongo import MongoClient, ASCENDING
import os

# SQLite DB path
SQLITE_DB_PATH = 'kline_cache.db'  # Use the same path as your check script
# MongoDB connection
MONGO_URL = os.getenv('MONGODB_URL', 'mongodb+srv://haithammisape:hrz123@binance.5hz1tvp.mongodb.net/?retryWrites=true&w=majority&appName=binance')
MONGO_DB_NAME = 'binance'

# Connect to SQLite
sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
sqlite_cursor = sqlite_conn.cursor()

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URL)
mongo_db = mongo_client[MONGO_DB_NAME]
klines_collection = mongo_db['klines']
klines_collection.create_index([('symbol', ASCENDING), ('interval', ASCENDING), ('open_time', ASCENDING)], unique=True)

# Fetch all klines from SQLite
def fetch_all_klines():
    sqlite_cursor.execute('SELECT symbol, interval, open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore FROM klines')
    return sqlite_cursor.fetchall()

# Insert klines into MongoDB
def insert_klines_to_mongo(klines):
    for i, k in enumerate(klines):
        doc = {
            'symbol': k[0],
            'interval': k[1],
            'open_time': int(k[2]),
            'open': k[3],
            'high': k[4],
            'low': k[5],
            'close': k[6],
            'volume': k[7],
            'close_time': int(k[8]),
            'quote_asset_volume': k[9],
            'number_of_trades': int(k[10]),
            'taker_buy_base_asset_volume': k[11],
            'taker_buy_quote_asset_volume': k[12],
            'ignore': k[13]
        }
        klines_collection.update_one(
            {'symbol': doc['symbol'], 'interval': doc['interval'], 'open_time': doc['open_time']},
            {'$set': doc},
            upsert=True
        )
        if (i+1) % 1000 == 0:
            print(f"Inserted {i+1} klines...")
    print(f"Migration complete. Inserted {len(klines)} klines.")

if __name__ == '__main__':
    print("Fetching klines from SQLite...")
    all_klines = fetch_all_klines()
    print(f"Fetched {len(all_klines)} klines. Migrating to MongoDB...")
    insert_klines_to_mongo(all_klines)
    print("Done.") 