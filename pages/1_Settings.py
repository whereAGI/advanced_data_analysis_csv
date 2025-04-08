import streamlit as st
import pandas as pd
import os
import io
from pathlib import Path

# Import local modules
from src.config import logger, APP_DATA_DIR, STATE_FILE_PATH
import src.data_handler as data_handler

# Set page config
st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

def main():
    """Main function for the Settings page"""
    st.title("⚙️ Settings")

    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["Data Upload", "API Settings", "Schema Context"])

    # Tab 1: Data Upload
    with tab1:
        handle_data_upload()

    # Tab 2: API Settings
    with tab2:
        handle_api_settings()

    # Tab 3: Schema Context
    with tab3:
        handle_schema_context()

    # Save settings button (outside tabs)
    if st.button("Save Settings", type="primary"):
        save_settings()

def handle_data_upload():
    """Handle data upload functionality"""
    st.header("Upload Your Data")
    st.write("Upload a CSV file to analyze with PandasAI")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Process uploaded file
    if uploaded_file is not None:
        # Save the file to a temporary location
        try:
            # Create directory if it doesn't exist
            os.makedirs(APP_DATA_DIR, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(APP_DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Load the CSV
            df = data_handler.load_csv(file_path)
            if df is not None:
                # Display success message
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(5))
                
                # Display data info
                st.subheader("Data Information")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                
                # Display data types
                st.subheader("Data Types")
                st.write(df.dtypes)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            logger.error(f"Error loading file: {e}", exc_info=True)
    
    # Show loaded file info
    if data_handler.get_dataframe() is not None:
        df = data_handler.get_dataframe()
        st.info(f"Current loaded dataset has {df.shape[0]} rows and {df.shape[1]} columns")

def handle_api_settings():
    """Handle API key and model settings"""
    st.header("API Settings")
    
    # API Key input
    api_key = st.text_input(
        "API Key (Groq or OpenAI)",
        value=data_handler._data_store.get("api_key", ""),
        type="password"
    )
    
    if api_key:
        data_handler.set_api_key(api_key)
    
    # Model selection
    available_models = data_handler.get_available_models()
    available_models.append("Custom Model")  # Add custom model option
    
    current_model = data_handler._data_store.get("model", "deepseek-r1-distill-llama-70b")
    
    # Create a selection index that defaults to the current model if it exists
    default_index = 0
    if current_model in available_models:
        default_index = available_models.index(current_model)
    # If using a custom model not in the list, select the "Custom Model" option
    elif "Custom Model" in available_models:
        default_index = available_models.index("Custom Model")
    
    model_selection = st.selectbox(
        "Select Model",
        options=available_models,
        index=default_index
    )
    
    # If Custom Model is selected, show a text input for the model name
    if model_selection == "Custom Model":
        custom_model = st.text_input(
            "Enter Custom Model Name",
            value=current_model if current_model not in available_models[:-1] else "",
            help="Enter the full name of the Groq model you want to use"
        )
        if custom_model:
            data_handler.set_model(custom_model)
    else:
        data_handler.set_model(model_selection)
    
    # Test connection button
    if st.button("Test Connection"):
        test_api_connection()

def test_api_connection():
    """Test API connection"""
    try:
        # Initialize PandasAI
        success = data_handler.init_pandasai()
        
        if success:
            st.success("Connection successful! Your API key is working.")
        else:
            st.error("Failed to initialize PandasAI. Please check your API key and try again.")
    except Exception as e:
        st.error(f"Error testing connection: {e}")
        logger.error(f"Error testing connection: {e}", exc_info=True)

def handle_schema_context():
    """Handle schema context settings"""
    st.header("Schema Context")
    st.write("Provide additional context about your data schema to help the AI understand it better.")
    
    # Get current schema context
    current_context = data_handler.get_schema_context()
    
    # Schema context input
    schema_context = st.text_area(
        "Schema Context",
        value=current_context,
        height=300,
        help="Describe your data schema, column meanings, and any relevant business context."
    )
    
    # Preview of schema context with data
    if schema_context:
        data_handler.set_schema_context(schema_context)
        
        # Show preview of how it will be used
        st.subheader("Preview")
        st.info("This is how your context will be used in queries:")
        
        example_query = "Show me the top 5 users"
        enhanced_query = f"Schema Information: {schema_context}\n\nUser Query: {example_query}"
        
        st.code(enhanced_query)

def save_settings():
    """Save all settings to disk"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        
        # Save the state
        success = data_handler.save_state(STATE_FILE_PATH)
        
        if success:
            st.success("Settings saved successfully!")
        else:
            st.error("Failed to save settings. Please try again.")
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        logger.error(f"Error saving settings: {e}", exc_info=True)

if __name__ == "__main__":
    # Try to load saved state
    if os.path.exists(STATE_FILE_PATH):
        try:
            data_handler.load_state(STATE_FILE_PATH)
            st.sidebar.success("Settings loaded from saved state")
        except Exception as e:
            logger.error(f"Error loading saved state: {e}", exc_info=True)
    
    main() 