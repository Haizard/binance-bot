import dns.resolver
from pymongo import MongoClient
import sys
import os

# Set environment variables to bypass proxy
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# Configure DNS resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']  # Google DNS
dns.resolver.default_resolver.timeout = 20
dns.resolver.default_resolver.lifetime = 20

# MongoDB connection string (direct connection)
MONGODB_URI = "mongodb://haithammisape:hrz123@ac-biem6go-shard-00-00.pzhq7bm.mongodb.net:27017,ac-biem6go-shard-00-01.pzhq7bm.mongodb.net:27017,ac-biem6go-shard-00-02.pzhq7bm.mongodb.net:27017/admin?replicaSet=atlas-qc7888-shard-0&tls=true&authSource=admin"

def test_connection():
    client = None
    try:
        print("=== MongoDB Connection Test ===")
        print("Using Google DNS servers (8.8.8.8, 8.8.4.4)")
        print("Bypassing proxy settings")
        print("Using direct connection string")
        print("\nAttempting to connect to MongoDB...")
        
        # Create MongoDB client with optimized settings
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            tlsAllowInvalidCertificates=True,  # Allow invalid certificates for testing
            retryWrites=True,
            maxPoolSize=1
        )
        
        # Force a connection attempt
        print("\nTesting connection...")
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB!")
        
        # Get database info
        print("\nRetrieving database information...")
        dbs = client.list_database_names()
        print(f"✓ Available databases: {dbs}")
        
        # Connect to specific database
        db = client.get_database()
        print(f"\nConnected to database: {db.name}")
        
        # List collections
        collections = db.list_collection_names()
        print(f"✓ Available collections: {collections}")
        
        if collections:
            print("\nCollection statistics:")
            for collection in collections:
                count = db[collection].count_documents({})
                print(f"- {collection}: {count} documents")
        
    except Exception as e:
        print("\n✗ Connection Error:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        
        # Additional error information
        if hasattr(e, 'details'):
            print("\nDetailed error information:")
            print(e.details)
        sys.exit(1)
    finally:
        if client:
            client.close()
            print("\nConnection closed.")

if __name__ == "__main__":
    test_connection() 