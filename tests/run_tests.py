"""
Test runner script.
"""
import unittest
import sys
import os
import logging
from datetime import datetime
import pytest
from test_config import TEST_LOG_CONFIG

def setup_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=TEST_LOG_CONFIG['level'],
        format=TEST_LOG_CONFIG['format'],
        filename=TEST_LOG_CONFIG['file']
    )

def run_tests():
    """Run all tests."""
    # Set up logging
    setup_logging()
    
    # Log test start
    logging.info(f"Starting tests at {datetime.now()}")
    
    try:
        # Run tests with pytest
        pytest.main([
            '-v',  # verbose output
            '--asyncio-mode=strict',  # strict async mode
            '--tb=short',  # shorter traceback format
            '--cov=agents',  # coverage for agents module
            '--cov=dashboards',  # coverage for dashboards module
            '--cov-report=term-missing',  # show missing lines
            'tests/'  # test directory
        ])
        
    except Exception as e:
        logging.error(f"Error running tests: {str(e)}")
        return 1
        
    finally:
        # Log test end
        logging.info(f"Finished tests at {datetime.now()}")
        
    return 0

if __name__ == '__main__':
    sys.exit(run_tests()) 