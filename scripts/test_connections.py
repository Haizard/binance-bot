"""
Test both local and remote MongoDB connections.
"""
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import get_database, close_connections, get_connection_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_connections():
    """Test database connections"""
    try:
        # Try to get database connection (will try remote first, then local)
        db = get_database()
        
        # Get connection info
        conn_info = get_connection_info()
        logger.info("\nConnection Information:")
        logger.info(f"Status: {'Connected' if conn_info['is_connected'] else 'Disconnected'}")
        logger.info(f"Using Local: {'Yes' if conn_info['using_local'] else 'No'}")
        logger.info(f"Database: {conn_info['database_name']}")
        logger.info(f"URI: {conn_info['uri'].split('@')[-1]}")  # Hide credentials
        
        # Test basic operations
        logger.info("\nTesting database operations...")
        
        # List collections
        collections = db.list_collection_names()
        logger.info(f"Available collections: {collections}")
        
        # Get document counts
        for collection in collections:
            count = db[collection].count_documents({})
            logger.info(f"Collection '{collection}' has {count} documents")
        
        logger.info("\nDatabase connection test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing database connections: {str(e)}")
        raise
    finally:
        close_connections()

if __name__ == "__main__":
    logger.info("Starting database connection tests...")
    test_connections() 