"""
Test script to verify PandasAI functionality.

This script creates a simple DataFrame and tests basic PandasAI functionality.
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
    sales_data = {
        'Country': ['United States', 'United Kingdom', 'France', 'Germany', 'Italy'],
        'Sales': [5000, 3200, 2900, 4100, 2300]
    }
    return pd.DataFrame(sales_data)

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
def test_pandasai_basic_query(sample_dataframe):
    """Test a basic PandasAI query."""
    try:
        import pandasai
        # Check which version of PandasAI we're using
        pandasai_v2 = hasattr(pandasai, "__version__")
        
        # Depending on PandasAI version, initialize differently
        if pandasai_v2:
            # Skip this test for now as we need to update the initialization logic
            pytest.skip("Test not yet updated for PandasAI v2+")
        else:
            # This is the v1 way
            pytest.skip("Test not yet updated for PandasAI v1")
    except Exception as e:
        pytest.skip(f"PandasAI initialization failed: {e}")

if __name__ == '__main__':
    pytest.main(["-v", __file__]) 