from pymongo import MongoClient
import sys
import os

# Bypass any proxy settings
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# Direct connection string (using the cluster's direct hostnames)
MONGODB_URI = "mongodb://haithammisape:hrz123@ac-yvwqxvl-shard-00-00.pzhq7bm.mongodb.net:27017,ac-yvwqxvl-shard-00-01.pzhq7bm.mongodb.net:27017,ac-yvwqxvl-shard-00-02.pzhq7bm.mongodb.net:27017/admin?replicaSet=atlas-qc7888-shard-0&tls=true&authSource=admin"

def test_connection():
    client = None
    try:
        print("=== MongoDB Direct Connection Test ===")
        print("Using direct connection string (bypassing DNS SRV)")
        print("\nAttempting to connect to MongoDB...")
        
        # Create MongoDB client with optimized settings
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            retryWrites=True,
            directConnection=False
        )
        
        # Force a connection attempt
        print("\nTesting connection...")
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB!")
        
        # Get database info
        print("\nRetrieving database information...")
        dbs = client.list_database_names()
        print(f"✓ Available databases: {dbs}")
        
        # List collections in each database
        for db_name in dbs:
            if db_name not in ['admin', 'local']:  # Skip system databases
                db = client[db_name]
                collections = db.list_collection_names()
                print(f"\nCollections in {db_name}:")
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