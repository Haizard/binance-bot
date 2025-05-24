from pymongo import MongoClient
from datetime import datetime

def main():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/tradingbot")
    db = client.tradingbot
    
    # Sample strategy document
    sample_strategy = {
        "name": "Simple Moving Average Crossover",
        "type": "Technical",
        "pairs": ["BTC/USDT", "ETH/USDT"],
        "performance": 15.5,
        "status": "active",
        "entry_conditions": ["SMA(20) crosses above SMA(50)"],
        "exit_conditions": ["SMA(20) crosses below SMA(50)"],
        "timeframes": ["1h", "4h"],
        "risk": {
            "stop_loss": 2.0,
            "take_profit": 5.0,
            "max_position": 1000.0
        },
        "timestamp": datetime.now()
    }
    
    # Insert the sample strategy
    result = db.strategies.insert_one(sample_strategy)
    print(f"Inserted sample strategy with ID: {result.inserted_id}")
    
    # Verify the strategy was added
    strategies = list(db.strategies.find())
    print(f"\nFound {len(strategies)} strategies:")
    for strategy in strategies:
        print(f"Strategy: {strategy}")

if __name__ == "__main__":
    main() 