from pymongo import MongoClient
import dns.resolver
import sys
import os

# Configure DNS resolver to use multiple DNS providers
dns_servers = [
    '1.1.1.1',  # Cloudflare
    '8.8.8.8',  # Google
    '9.9.9.9'   # Quad9
]

# Configure DNS resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = dns_servers
dns.resolver.default_resolver.timeout = 10
dns.resolver.default_resolver.lifetime = 10

# Set environment variables
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# MongoDB connection string
MONGODB_URI = "mongodb+srv://haithammisape:hrz123@algobot.pzhq7bm.mongodb.net/?retryWrites=true&w=majority&appName=algobot"

def test_connection():
    client = None
    try:
        print("=== MongoDB Connection Test with Custom DNS ===")
        print(f"Using DNS servers: {', '.join(dns_servers)}")
        print("\nAttempting to connect to MongoDB...")
        
        # Create MongoDB client with optimized settings
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
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
        
        # Try to get more DNS information
        try:
            print("\nTrying to resolve MongoDB hostnames...")
            hosts = [
                "algobot.pzhq7bm.mongodb.net",
                "ac-yvwqxvl-shard-00-00.pzhq7bm.mongodb.net",
                "ac-yvwqxvl-shard-00-01.pzhq7bm.mongodb.net",
                "ac-yvwqxvl-shard-00-02.pzhq7bm.mongodb.net"
            ]
            for host in hosts:
                try:
                    print(f"\nResolving {host}...")
                    answers = dns.resolver.resolve(host, 'A')
                    for rdata in answers:
                        print(f"✓ {host} -> {rdata.address}")
                except Exception as dns_error:
                    print(f"✗ Failed to resolve {host}: {str(dns_error)}")
        except Exception as dns_test_error:
            print(f"Error during DNS testing: {str(dns_test_error)}")
        
        sys.exit(1)
    finally:
        if client:
            client.close()
            print("\nConnection closed.")

if __name__ == "__main__":
    test_connection() 