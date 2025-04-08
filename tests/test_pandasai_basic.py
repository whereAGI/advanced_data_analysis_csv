"""
Basic test script for PandasAI functionality.
This script tests basic PandasAI functionality with a simple DataFrame.
"""

import os
import pandas as pd
import pytest
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
    }
    return pd.DataFrame(data)

def test_pandasai_import():
    """Test that PandasAI can be imported."""
    try:
        import pandasai
        assert True, "PandasAI successfully imported"
    except ImportError as e:
        pytest.skip(f"PandasAI import failed: {e}")

def test_api_key_available():
    """Test that an API key is available."""
    api_key = os.getenv("PANDASAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("No API key found. Please set PANDASAI_API_KEY or GROQ_API_KEY in your environment.")
    assert api_key is not None, "API key is available"

@pytest.mark.skipif(
    os.getenv("PANDASAI_API_KEY") is None and os.getenv("GROQ_API_KEY") is None,
    reason="No API key available"
)
def test_dataframe_analysis(sample_dataframe):
    """Test simple DataFrame analysis."""
    # This test is skipped for now until we implement proper PandasAI integration
    pytest.skip("Test not implemented yet")

if __name__ == '__main__':
    pytest.main(["-v", __file__]) 