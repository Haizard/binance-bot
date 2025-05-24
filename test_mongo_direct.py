from pymongo import MongoClient
import sys
import time

# Direct connection string format (using multiple hosts)
MONGODB_URI = "mongodb://haithammisape:hrz123@ac-yvwqxvl-shard-00-00.pzhq7bm.mongodb.net:27017,ac-yvwqxvl-shard-00-01.pzhq7bm.mongodb.net:27017,ac-yvwqxvl-shard-00-02.pzhq7bm.mongodb.net:27017/trading_bot?ssl=true&replicaSet=atlas-qc7888-shard-0&authSource=admin&retryWrites=true&w=majority"

def test_connection():
    client = None
    try:
        print("Attempting to connect to MongoDB...")
        print("Using direct connection method...")
        
        # Create MongoDB client with specific options
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=30000,  # 30 seconds
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            maxPoolSize=1,
            retryWrites=True,
            ssl=True
        )
        
        # Force a connection attempt
        print("\nTesting connection...")
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB!")
        
        # Get database
        print("\nAccessing database...")
        db = client.trading_bot
        
        # List collections
        print("Retrieving collections...")
        collections = db.list_collection_names()
        print(f"✓ Available collections: {collections}")
        
        # Try to perform a simple operation
        if 'trades' in collections:
            doc_count = db.trades.count_documents({})
            print(f"✓ Number of documents in trades collection: {doc_count}")
        
    except Exception as e:
        print("\n✗ Connection Error:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        sys.exit(1)
    finally:
        if client:
            client.close()
            print("\nConnection closed.")

if __name__ == "__main__":
    print("=== MongoDB Direct Connection Test ===")
    test_connection() 