import unittest
import pandas as pd
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from src import data_handler

class TestDataHandler(unittest.TestCase):
    """Unit tests for the data_handler module."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Reset data store before each test
        data_handler._data_store = {
            "df": None,
            "pandas_ai": None,
            "llm": None,
            "csv_path": None,
            "schema_context": "",
            "api_key": None,
            "model": "test-model"
        }
        
        # Create a temporary CSV file for testing
        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.temp_csv_path = self.temp_csv.name
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'value': [10.5, 20.3, 15.7, 8.2, 12.9]
        })
        
        # Write DataFrame to CSV
        self.test_df.to_csv(self.temp_csv_path, index=False)
        self.temp_csv.close()
        
        # Create temporary state file
        self.temp_state = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.temp_state_path = self.temp_state.name
        self.temp_state.close()
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files
        if os.path.exists(self.temp_csv_path):
            os.unlink(self.temp_csv_path)
        
        if os.path.exists(self.temp_state_path):
            os.unlink(self.temp_state_path)
    
    def test_load_csv(self):
        """Test loading CSV file."""
        # Load the test CSV file
        df = data_handler.load_csv(self.temp_csv_path)
        
        # Verify DataFrame loaded correctly
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ['id', 'name', 'value'])
        self.assertEqual(df['name'][1], 'Bob')
        
        # Verify data store updated
        self.assertIsNotNone(data_handler._data_store['df'])
        self.assertEqual(data_handler._data_store['csv_path'], self.temp_csv_path)
    
    def test_get_dataframe(self):
        """Test getting DataFrame."""
        # Set DataFrame in data store
        data_handler._data_store['df'] = self.test_df
        
        # Get DataFrame
        df = data_handler.get_dataframe()
        
        # Verify DataFrame retrieved correctly
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ['id', 'name', 'value'])
    
    def test_set_api_key(self):
        """Test setting API key."""
        # Set API key
        result = data_handler.set_api_key('test-api-key')
        
        # Verify API key set correctly
        self.assertTrue(result)
        self.assertEqual(data_handler._data_store['api_key'], 'test-api-key')
    
    def test_set_model(self):
        """Test setting model."""
        # Set model
        result = data_handler.set_model('new-test-model')
        
        # Verify model set correctly
        self.assertTrue(result)
        self.assertEqual(data_handler._data_store['model'], 'new-test-model')
    
    def test_schema_context(self):
        """Test schema context functions."""
        # Set schema context
        test_context = "This is a test schema context"
        result = data_handler.set_schema_context(test_context)
        
        # Verify schema context set correctly
        self.assertTrue(result)
        self.assertEqual(data_handler._data_store['schema_context'], test_context)
        
        # Get schema context
        context = data_handler.get_schema_context()
        
        # Verify schema context retrieved correctly
        self.assertEqual(context, test_context)
    
    @patch('src.data_handler.init_pandasai')
    def test_save_load_state(self, mock_init_pandasai):
        """Test saving and loading state."""
        # Set up mock
        mock_init_pandasai.return_value = True
        
        # Set up test data
        data_handler._data_store = {
            "df": self.test_df,
            "pandas_ai": None,
            "llm": None,
            "csv_path": self.temp_csv_path,
            "schema_context": "Test context",
            "api_key": "test-api-key",
            "model": "test-model"
        }
        
        # Save state
        result = data_handler.save_state(self.temp_state_path)
        
        # Verify state saved correctly
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.temp_state_path))
        
        # Reset data store
        data_handler._data_store = {
            "df": None,
            "pandas_ai": None,
            "llm": None,
            "csv_path": None,
            "schema_context": "",
            "api_key": None,
            "model": "default-model"
        }
        
        # Load state
        result = data_handler.load_state(self.temp_state_path)
        
        # Verify state loaded correctly
        self.assertTrue(result)
        self.assertEqual(data_handler._data_store['csv_path'], self.temp_csv_path)
        self.assertEqual(data_handler._data_store['schema_context'], "Test context")
        self.assertEqual(data_handler._data_store['api_key'], "test-api-key")
        self.assertEqual(data_handler._data_store['model'], "test-model")
        
        # Verify PandasAI initialization attempted
        mock_init_pandasai.assert_called_once()

if __name__ == '__main__':
    unittest.main() 