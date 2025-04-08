import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import local modules
from src.config import logger, APP_DATA_DIR, STATE_FILE_PATH
import src.data_handler as data_handler

# Set page config
st.set_page_config(
    page_title="Data Chat Home",
    page_icon="🏠",
    layout="wide"
)

def main():
    """Main function for the Home page"""
    st.title("🏠 Data Chat Home")
    
    # Show welcome message
    st.markdown("""
    ## Welcome to Data Chat!
    
    This application helps you analyze your data using natural language queries powered by PandasAI and Groq.
    
    ### Getting Started:
    
    1. Go to the **⚙️ Settings** page to:
       - Upload your CSV data
       - Set your API key in the API Settings tab
       - Choose your preferred model
       - Add schema context (optional)
       
    2. Then visit the **💬 Chat** page to:
       - Ask questions about your data
       - Get instant visualizations and insights
       
    ### Features:
    
    - **Natural Language Queries**: Ask questions in plain English
    - **Automatic Visualizations**: Get charts and graphs based on your data
    - **Schema Context**: Provide additional information to improve AI understanding
    - **Data Persistence**: Your settings and data are saved between sessions
    
    ### Example Questions:
    
    - "Show me the distribution of values in column X"
    - "What's the average of column Y grouped by column Z?"
    - "Find the top 5 rows with the highest values in column A"
    - "Plot the trend of column B over time"
    """)
    
    # Display current state information
    st.sidebar.header("Current State")
    
    # Check if data is loaded
    df = data_handler.get_dataframe()
    if df is not None:
        st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        st.sidebar.warning("❌ No data loaded")
        
    # Check if API key is set
    if data_handler._data_store.get("api_key"):
        st.sidebar.success("✅ API key configured")
    else:
        st.sidebar.warning("❌ API key not set")
        
    # Check if schema context is provided
    schema_context = data_handler.get_schema_context()
    if schema_context:
        st.sidebar.success("✅ Schema context provided")
    else:
        st.sidebar.info("ℹ️ No schema context provided (optional)")
        
    # Check if PandasAI is initialized
    if data_handler.get_pandasai():
        st.sidebar.success("✅ PandasAI initialized")
    else:
        st.sidebar.info("ℹ️ PandasAI not initialized")
        
    # Check if state has been saved
    if os.path.exists(STATE_FILE_PATH):
        st.sidebar.success("✅ Settings saved to disk")
    else:
        st.sidebar.info("ℹ️ Settings not saved yet")

if __name__ == "__main__":
    # Try to load saved state
    if os.path.exists(STATE_FILE_PATH):
        try:
            data_handler.load_state(STATE_FILE_PATH)
            st.sidebar.success("Settings loaded from saved state")
        except Exception as e:
            logger.error(f"Error loading saved state: {e}", exc_info=True)
    
    main() 