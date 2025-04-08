#!/usr/bin/env python
"""
Test runner script to execute all unit tests in the tests directory.
This provides a convenient way to run all tests with a single command.
"""

import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    """Discover and run all tests in the tests directory."""
    # Start from the current directory
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Discover all tests in the tests directory
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    # Run tests and exit with appropriate exit code
    sys.exit(run_tests()) 