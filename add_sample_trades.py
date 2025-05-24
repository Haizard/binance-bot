from pymongo import MongoClient
from datetime import datetime, timedelta
import random

def main():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/tradingbot")
    db = client.tradingbot
    
    # Sample trades
    symbols = ["BTC/USDT", "ETH/USDT"]
    sides = ["buy", "sell"]
    
    # Create some trades over the last few days
    trades = []
    for i in range(20):
        # Random trade data
        symbol = random.choice(symbols)
        side = random.choice(sides)
        entry_price = random.uniform(25000, 30000) if symbol == "BTC/USDT" else random.uniform(1800, 2000)
        quantity = random.uniform(0.1, 1.0)
        pnl = random.uniform(-500, 1000)
        
        # Create trade with timestamp within last 3 days
        trade = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": entry_price * (1 + random.uniform(-0.05, 0.05)),
            "quantity": quantity,
            "realized_pnl": pnl,
            "status": "closed",
            "timestamp": datetime.now() - timedelta(days=random.uniform(0, 3))
        }
        trades.append(trade)
    
    # Add a few active trades
    for i in range(3):
        symbol = random.choice(symbols)
        trade = {
            "symbol": symbol,
            "side": random.choice(sides),
            "entry_price": random.uniform(25000, 30000) if symbol == "BTC/USDT" else random.uniform(1800, 2000),
            "quantity": random.uniform(0.1, 1.0),
            "realized_pnl": 0,
            "status": "active",
            "timestamp": datetime.now() - timedelta(hours=random.uniform(1, 12))
        }
        trades.append(trade)
    
    # Insert the trades
    result = db.trades.insert_many(trades)
    print(f"Inserted {len(result.inserted_ids)} trades")
    
    # Verify the trades were added
    total_trades = db.trades.count_documents({})
    active_trades = db.trades.count_documents({"status": "active"})
    print(f"\nTotal trades in database: {total_trades}")
    print(f"Active trades: {active_trades}")

if __name__ == "__main__":
    main() 