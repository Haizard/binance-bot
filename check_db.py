from pymongo import MongoClient

def main():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/tradingbot")
    db = client.tradingbot
    
    # Get all strategies
    strategies = list(db.strategies.find())
    print(f"Found {len(strategies)} strategies:")
    for strategy in strategies:
        print(f"Strategy: {strategy}")

if __name__ == "__main__":
    main() 