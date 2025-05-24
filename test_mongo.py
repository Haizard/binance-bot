from pymongo import MongoClient
import sys
import dns.resolver

# Configure DNS resolver to use Google's DNS
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

# MongoDB connection string
MONGODB_URI = "mongodb+srv://haithammisape:hrz123@algobot.pzhq7bm.mongodb.net/?retryWrites=true&w=majority&appName=algobot"

def test_connection():
    client = None
    try:
        # Create a MongoDB client with more specific options
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000,  # Increased timeout
            connectTimeoutMS=20000,
            socketTimeoutMS=20000,
            maxPoolSize=1
        )
        
        # Force a connection attempt
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        # Get database
        db = client.trading_bot
        
        # List collections
        collections = db.list_collection_names()
        print(f"\nAvailable collections: {collections}")
        
        # Try to perform a simple operation
        doc_count = db.trades.count_documents({})
        print(f"\nNumber of documents in trades collection: {doc_count}")
        
    except Exception as e:
        print(f"\nError details:", file=sys.stderr)
        print(f"Type: {type(e).__name__}", file=sys.stderr)
        print(f"Message: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client:
            client.close()
            print("\nConnection closed.")

if __name__ == "__main__":
    print("Testing MongoDB connection...")
    print(f"Using connection string: {MONGODB_URI}")
    test_connection() 