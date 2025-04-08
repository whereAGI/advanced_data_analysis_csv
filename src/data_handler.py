import pandas as pd
import streamlit as st # Needed for UploadedFile type hint if desired, though not strictly necessary for runtime
from io import StringIO # To handle the uploaded file buffer
import logging
import os
import json
import re
import shutil
from pathlib import Path
from src.config import logger, PANDASAI_API_KEY, VISUALIZATION_CONFIG # Import the configured logger and PandasAI settings
import traceback
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
import time

# Import PandasAI
try:
    # Try to import from both PandasAI v1 and v2
    try:
        # PandasAI v2
        import pandasai
        from pandasai.smart_dataframe import SmartDataframe
        from pandasai.llm import OpenAI
        # Check if we're using PandasAI v2+
        PANDASAI_V2 = hasattr(pandasai, "__version__")
        logger.info(f"PandasAI imported successfully. Version: {pandasai.__version__}")
        PANDASAI_AVAILABLE = True
    except ImportError as e1:
        logger.error(f"Error importing PandasAI v2: {e1}")
        # PandasAI v1
        try:
            from pandasai.llm.base import LLM
            from pandasai import PandasAI
            from pandasai.llm.openai import OpenAI
            PANDASAI_V2 = False
            logger.info("PandasAI v1 imported successfully")
            PANDASAI_AVAILABLE = True
        except ImportError as e2:
            logger.error(f"Error importing PandasAI v1: {e2}")
            raise
except ImportError as e:
    logger.error(f"PandasAI import error: {e}")
    PANDASAI_AVAILABLE = False
    PANDASAI_V2 = False
    # Define a stub class for type hints
    class SmartDataframe:
        pass
    class PandasAI:
        pass
    
# Import custom Groq LLM if available
try:
    from src.groq_llm import GroqLLM
    GROQ_LLM_AVAILABLE = True
except ImportError:
    GROQ_LLM_AVAILABLE = False

# Constants for file paths
UPLOADS_DIR = "uploads"
CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# Optional: Type hinting for the input
# from streamlit.runtime.uploaded_file_manager import UploadedFile

# Global variable to store loaded data and configuration
_data_store = {
    "df": None,
    "pandas_ai": None,
    "llm": None,
    "csv_path": None,
    "schema_context": "",
    "api_key": None,
    # "model": "deepseek-r1-distill-llama-70b",  # Set default model
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # Set default model
}

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame
    """
    try:
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        _data_store["df"] = df
        _data_store["csv_path"] = file_path
        
        # Attempt to extract schema relationships automatically
        auto_schema_context = infer_schema_relationships(df)
        if auto_schema_context and not _data_store.get("schema_context"):
            _data_store["schema_context"] = auto_schema_context
            logger.info("Automatically inferred schema relationships")
        
        # Convert to PandasAI format if available
        if PANDASAI_AVAILABLE:
            init_pandasai()
            
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        logger.error(traceback.format_exc())
        raise

def get_dataframe() -> pd.DataFrame:
    """
    Get the loaded DataFrame
    
    Returns:
        Pandas DataFrame or None if not loaded
    """
    return _data_store["df"]

def init_pandasai() -> bool:
    """
    Initialize PandasAI with the current DataFrame and LLM
    
    Returns:
        True if initialization successful, False otherwise
    """
    if not PANDASAI_AVAILABLE:
        logger.warning("PandasAI not available. Install with pip install pandasai>=2.0.0")
        return False
        
    if _data_store["df"] is None:
        logger.warning("DataFrame not loaded. Cannot initialize PandasAI")
        return False
        
    try:
        logger.info("Initializing PandasAI")
        
        # Use the stored API key
        api_key = _data_store.get("api_key")
        if not api_key:
            logger.warning("API key not set. Cannot initialize PandasAI")
            return False
            
        model = _data_store.get("model", "deepseek-r1-distill-llama-70b")
        logger.info(f"Using model: {model}")
        logger.info(f"API key length: {len(api_key)}")
        
        # Create LLM based on availability
        if GROQ_LLM_AVAILABLE:
            logger.info(f"Using GroqLLM with model: {model}")
            llm = GroqLLM(api_token=api_key, model=model)
            logger.info("GroqLLM created successfully")
        else:
            # Fallback to generic OpenAI interface
            logger.info("Using default OpenAI interface")
            llm = OpenAI(api_token=api_key)
            logger.info("OpenAI LLM created successfully")
            
        # Store LLM
        _data_store["llm"] = llm
        
        # Create PandasAI instance based on version
        if PANDASAI_V2:
            logger.info("Using PandasAI v2")
            # Configure PandasAI with custom settings
            config = {
                "llm": llm,
                "enable_cache": True,
                "custom_prompts": {
                    "generate_python_code": """
                    Generate Python code to answer this question: {prompt}

                    Rules:
                    1. Generate executable Python code
                    2. Return a dictionary with exactly this format:
                       result = {{"type": "dataframe", "value": df}}
                    3. Use proper pandas syntax
                    4. Handle all edge cases
                    5. Return empty DataFrame if no results found
                    6. Include RELATED data based on the schema that provides context
                    7. Look for relationships in the schema and include relevant columns from related entities
                    8. If retrieving specific entities, also include any contextually relevant attributes

                    Code:
                    """
                },
                "save_charts": True,
                "verbose": True,
                "enforce_output_type": True,
                "sanitize_output": True
            }
            
            # Store the config for SmartDataframe
            _data_store["pandas_ai"] = config
        else:
            logger.info("Using PandasAI v1")
            # PandasAI V1 accepts dataframe on initialization
            pandas_ai = PandasAI(llm)
            _data_store["pandas_ai"] = pandas_ai
            
        logger.info("PandasAI initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing PandasAI: {e}")
        logger.error(traceback.format_exc())
        return False

def get_pandasai() -> Optional[Union[SmartDataframe, 'PandasAI']]:
    """
    Get the PandasAI instance
    
    Returns:
        PandasAI instance or None if not initialized
    """
    return _data_store.get("pandas_ai")

def get_available_models() -> List[str]:
    """
    Get available models
    
    Returns:
        List of model names
    """
    # Try to get models from GroqLLM if available
    if GROQ_LLM_AVAILABLE:
        try:
            # Create a temporary instance to get models
            temp_llm = GroqLLM(api_token="temp")
            models = temp_llm.get_models()
            logger.info(f"Got {len(models)} models from GroqLLM")
            return models
        except Exception as e:
            logger.error(f"Error getting models from GroqLLM: {e}")
            
    # Fallback model list
    return [
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]

def set_api_key(api_key: str) -> bool:
    """
    Set the API key
    
    Args:
        api_key: The API key
        
    Returns:
        True if successful, False otherwise
    """
    _data_store["api_key"] = api_key
    logger.info("API key set successfully")
    
    # Re-initialize PandasAI if already initialized
    if _data_store["pandas_ai"] is not None and _data_store["df"] is not None:
        return init_pandasai()
    
    return True

def set_model(model: str) -> bool:
    """
    Set the model
    
    Args:
        model: The model name
        
    Returns:
        True if successful, False otherwise
    """
    _data_store["model"] = model
    logger.info(f"Model set to: {model}")
    
    # Re-initialize PandasAI if already initialized
    if _data_store["pandas_ai"] is not None and _data_store["df"] is not None:
        return init_pandasai()
    
    return True

def get_schema_context() -> str:
    """
    Get the schema context
    
    Returns:
        Schema context string
    """
    return _data_store.get("schema_context", "")

def set_schema_context(context: str) -> bool:
    """
    Set the schema context
    
    Args:
        context: The schema context
        
    Returns:
        True if successful, False otherwise
    """
    _data_store["schema_context"] = context
    logger.info("Schema context set successfully")
    return True

def save_state(file_path: str) -> bool:
    """
    Save the current state to a file
    
    Args:
        file_path: Path to save the state
        
    Returns:
        True if successful, False otherwise
    """
    try:
        state = {
            "csv_path": _data_store.get("csv_path"),
            "schema_context": _data_store.get("schema_context"),
            "api_key": _data_store.get("api_key"),
            "model": _data_store.get("model"),
        }
        
        with open(file_path, "w") as f:
            json.dump(state, f)
            
        logger.info(f"State saved to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        logger.error(traceback.format_exc())
        return False

def load_state(file_path: str) -> bool:
    """
    Load state from a file
    
    Args:
        file_path: Path to the state file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, "r") as f:
            state = json.load(f)
            
        # Set values from state
        if "api_key" in state:
            _data_store["api_key"] = state["api_key"]
            
        if "model" in state:
            _data_store["model"] = state["model"]
            
        if "schema_context" in state:
            _data_store["schema_context"] = state["schema_context"]
            
        # Load CSV if path exists and is valid
        if "csv_path" in state and state["csv_path"] and os.path.exists(state["csv_path"]):
            try:
                load_csv(state["csv_path"])
            except Exception as e:
                logger.error(f"Error loading CSV from saved state: {e}")
        
        logger.info(f"State loaded from: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        logger.error(traceback.format_exc())
        return False

def run_pandasai_query(query: str) -> Dict[str, Any]:
    """
    Run a query using PandasAI with improved rate limit handling
    
    Args:
        query: The query to run
        
    Returns:
        Query results (structured for display)
    """
    try:
        # Get the dataframe
        df = get_dataframe()
        if df is None:
            return {"type": "error", "value": "No DataFrame loaded"}
        
        # Get Schema context
        schema_context = _data_store.get("schema_context", "")
        
        # Add any schema context to enhance the query
        enhanced_query = query
        if schema_context:
            enhanced_query = f"""Question: {query}

Schema Context:
{schema_context}

Please use the schema context to include related data that might be relevant."""
        
        # Log the query
        logger.info(f"Running PandasAI query: {query}")
        
        # Check if PandasAI v2 is being used
        if PANDASAI_V2:
            # PandasAI v2 uses SmartDataframe
            logger.info("Running query with PandasAI v2 (SmartDataframe)")
            config = _data_store.get("pandas_ai")
            if config is None:
                return {"type": "error", "value": "PandasAI not initialized"}
                
            llm = config.get("llm")
            if llm is None:
                return {"type": "error", "value": "LLM not initialized"}
            
            # Set up retry handling for rate limits
            max_retries = 3
            retry_count = 0
            backoff_time = 30  # Initial backoff time in seconds
            
            while retry_count < max_retries:
                try:
                    # Disable cache globally
                    config["enable_cache"] = False
                    
                    # Modify the custom prompt to include more explicit instructions and constraints
                    config["custom_prompts"] = {
                        "generate_python_code": """
                        Generate Python code to answer this question: {prompt}

                        Rules:
                        1. Generate ONLY executable Python code
                        2. Import pandas as pd at the beginning
                        3. Use ONLY the 'df' variable which is already available
                        4. Return a dictionary with exactly this format: 
                           result = {{"type": "dataframe", "value": result_df}}
                        5. Use proper pandas syntax
                        6. Handle all edge cases
                        7. Return empty DataFrame if no results found
                        8. Include RELATED data based on the schema that provides context
                        9. DO NOT use undefined variables
                        10. DO NOT use lists of dataframes or try to concatenate multiple dataframes
                        11. ONLY work with the single 'df' variable already provided
                        12. Handle datetime columns properly with pd.to_datetime() when needed

                        Code:
                        """
                    }
                    
                    # Create SmartDataframe with the config
                    smart_df = SmartDataframe(df, config=config)
                    
                    # Regular query handling for all queries - with rate limit handling
                    try:
                        logger.info("Executing SmartDataframe chat query...")
                        result = smart_df.chat(enhanced_query)
                        logger.info(f"Query result type: {type(result)}")
                        
                        # Success - break out of the retry loop
                        break
                        
                    except Exception as query_error:
                        error_message = str(query_error)
                        
                        # Check if this is a rate limit error
                        if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
                            retry_count += 1
                            logger.warning(f"Rate limit hit during query execution. Retry {retry_count}/{max_retries}")
                            
                            # If we haven't exhausted retries, wait and try again
                            if retry_count < max_retries:
                                wait_time = backoff_time * retry_count
                                logger.info(f"Waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                                continue
                        
                        # For non-rate limit errors or if we've exhausted retries, re-raise
                        raise query_error
                
                except Exception as e:
                    error_message = str(e)
                    
                    # Check if this is a rate limit error
                    if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
                        retry_count += 1
                        logger.warning(f"Rate limit hit during SmartDataframe setup. Retry {retry_count}/{max_retries}")
                        
                        # If we haven't exhausted retries, wait and try again
                        if retry_count < max_retries:
                            wait_time = backoff_time * retry_count
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                    
                    # If this is not a rate limit error or we've exhausted retries, log and return error
                    logger.error(f"SmartDataframe chat error: {e}")
                    logger.error(traceback.format_exc())
                    return {"type": "error", "value": str(e)}
            
            # Check if we exhausted retries
            if retry_count >= max_retries:
                logger.error("Maximum retries exceeded for query due to rate limits")
                return {
                    "type": "error", 
                    "value": "Unable to execute query due to persistent rate limit errors. Please try again later."
                }
            
            # Various result types handling
            if isinstance(result, str):
                # Extract code if result is a string containing thinking steps
                thinking_match = re.search(r"<think>(.*?)</think>", result, re.DOTALL)
                dict_match = re.search(r"{'type':\s*'code',\s*'value':\s*'(.*?)'}", result, re.DOTALL)
                
                if thinking_match:
                    # Extract only the part after thinking
                    clean_result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL)
                    if "Python code:" in clean_result:
                        clean_result = clean_result.split("Python code:")[1].strip()
                    # Return as string
                    return {"type": "string", "value": clean_result}
                elif dict_match:
                    # Handle dictionary format response
                    code = dict_match.group(1).replace('\\n', '\n')
                    # Try to evaluate the code
                    try:
                        local_vars = {"df": df.copy(), "pd": pd}
                        exec(code, globals(), local_vars)
                        if 'result' in local_vars:
                            return local_vars['result']
                    except Exception as e:
                        logger.error(f"Error executing extracted code: {e}")
                    return {"type": "string", "value": result}
                else:
                    return {"type": "string", "value": result}
            elif isinstance(result, pd.DataFrame):
                # Check if we should analyze the DataFrame
                df_result = analyze_result_dataframe(result, query, schema_context)
                return df_result
            elif isinstance(result, dict) and "type" in result and "value" in result:
                # If this is a DataFrame result, analyze it
                if result["type"] == "dataframe" and isinstance(result["value"], pd.DataFrame):
                    result = analyze_result_dataframe(result["value"], query, schema_context, result)
                return result
            else:
                return {"type": "string", "value": str(result)}
        else:
            # PandasAI v1 accepts dataframe first, then query
            logger.info("Running query with PandasAI v1")
            pandas_ai = _data_store["pandas_ai"]
            
            # Set up retry handling for rate limits
            max_retries = 3
            retry_count = 0
            backoff_time = 30  # Initial backoff time in seconds
            
            while retry_count < max_retries:
                try:
                    # Add a wrapper around the query to encourage including related data
                    enhanced_query_v1 = f"""{enhanced_query}

Important: When answering, please:
1. Include related data fields that provide context
2. Return not just the exact answer but also contextually relevant attributes
3. If the query is for specific items, include related attributes that help understand the data better
4. Look for relationships between data columns and include any related information"""

                    # Execute the query with rate limit awareness
                    logger.info("Executing PandasAI run query...")
                    result = pandas_ai.run(df, enhanced_query_v1)
                    logger.info(f"Query result type: {type(result)}")
                    
                    # If we got here, the query was successful
                    break
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # Check if this is a rate limit error
                    if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
                        retry_count += 1
                        logger.warning(f"Rate limit hit during query. Retry {retry_count}/{max_retries}")
                        
                        # If we haven't exhausted retries, wait and try again
                        if retry_count < max_retries:
                            wait_time = backoff_time * retry_count
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                    
                    # For non-rate limit errors or if we've exhausted retries, log and return error
                    logger.error(f"PandasAI run error: {e}")
                    return {"type": "error", "value": str(e)}
            
            # Check if we exhausted retries
            if retry_count >= max_retries:
                logger.error("Maximum retries exceeded for query due to rate limits")
                return {
                    "type": "error", 
                    "value": "Unable to execute query due to persistent rate limit errors. Please try again later."
                }
            
            # Ensure result is in correct format
            if isinstance(result, (str, int, float)):
                return {"type": "string", "value": str(result)}
            elif isinstance(result, pd.DataFrame):
                # Check if we should analyze the DataFrame
                df_result = analyze_result_dataframe(result, query, schema_context)
                return df_result
            elif isinstance(result, dict) and "type" in result and "value" in result:
                # If this is a DataFrame result, analyze it
                if result["type"] == "dataframe" and isinstance(result["value"], pd.DataFrame):
                    result = analyze_result_dataframe(result["value"], query, schema_context, result)
                return result
            else:
                return {"type": "string", "value": str(result)}
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error running PandasAI query: {error_message}")
        logger.error(traceback.format_exc())
        
        # Check if this is a rate limit error
        if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
            return {
                "type": "error", 
                "value": "Rate limit exceeded. The API allows a limited number of tokens per minute. Please wait a moment before trying again."
            }
            
        return {"type": "error", "value": str(e)}

def analyze_result_dataframe(df: pd.DataFrame, query: str, schema_context: str, existing_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze a result DataFrame and add analysis to the response with improved rate limit handling
    
    Args:
        df: The pandas DataFrame to analyze
        query: The original user query
        schema_context: The schema context information
        existing_result: An existing result dictionary to augment (optional)
        
    Returns:
        Enhanced result with analysis
    """
    try:
        # Get the LLM to use for analysis
        llm = _data_store.get("llm")
        if not llm or not hasattr(llm, "analyze_dataframe"):
            logger.warning("LLM not available or doesn't support DataFrame analysis")
            if existing_result:
                return existing_result
            return {"type": "dataframe", "value": df}
        
        # Set up retry handling for rate limits
        max_retries = 3
        retry_count = 0
        backoff_time = 30  # Initial backoff time in seconds
        
        # Generate analysis with retry for rate limits
        logger.info("Analyzing result DataFrame...")
        analysis_result = None
        
        while retry_count < max_retries:
            try:
                analysis_result = llm.analyze_dataframe(df, query, schema_context)
                # If we got here, the analysis was successful
                break
            except Exception as e:
                error_message = str(e)
                
                # Check if this is a rate limit error
                if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
                    retry_count += 1
                    logger.warning(f"Rate limit hit during DataFrame analysis. Retry {retry_count}/{max_retries}")
                    
                    # If we haven't exhausted retries, wait and try again
                    if retry_count < max_retries:
                        wait_time = backoff_time * retry_count
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                
                # For non-rate limit errors or if we've exhausted retries, log and re-raise
                logger.error(f"Error during DataFrame analysis: {error_message}")
                raise
        
        # Check if we exhausted retries
        if retry_count >= max_retries:
            logger.error("Maximum retries exceeded for DataFrame analysis due to rate limits")
            if existing_result:
                existing_result["analysis"] = "Unable to provide analysis due to API rate limits. The analysis is temporarily unavailable."
                return existing_result
            else:
                return {
                    "type": "dataframe", 
                    "value": df, 
                    "analysis": "Unable to provide analysis due to API rate limits. Please try again later."
                }
        
        # Create or augment result
        if existing_result:
            result = existing_result
        else:
            result = {"type": "dataframe", "value": df}
        
        # Add analysis and reasoning to result
        if isinstance(analysis_result, dict):
            # New format with separate analysis and reasoning
            result["analysis"] = analysis_result.get("analysis", "")
            result["reasoning"] = analysis_result.get("reasoning", "")
            
            # Propagate data truncation information if available
            for field in ["data_truncated", "full_size", "sample_size", "estimated_tokens", "token_limit"]:
                if field in analysis_result:
                    result[field] = analysis_result[field]
                    
        else:
            # Handle legacy format (string) for backward compatibility
            result["analysis"] = analysis_result
            result["reasoning"] = ""
        
        return result
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error analyzing result DataFrame: {error_message}")
        
        # Check if this is a rate limit error
        if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
            if existing_result:
                existing_result["analysis"] = "Unable to provide analysis due to API rate limits. The analysis is temporarily unavailable."
                return existing_result
            else:
                return {
                    "type": "dataframe", 
                    "value": df, 
                    "analysis": "Unable to provide analysis due to API rate limits. Please try again later."
                }
                
        if existing_result:
            return existing_result
        return {"type": "dataframe", "value": df}

def save_uploaded_file(uploaded_file) -> str:
    """
    Saves an uploaded file to the uploads directory.
    
    Args:
        uploaded_file: The file object from st.file_uploader.
        
    Returns:
        The path where the file was saved, or None if saving failed.
    """
    if uploaded_file is None:
        logger.warning("save_uploaded_file called with None object.")
        return None
    
    try:
        # Ensure uploads directory exists
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Create a safe filename
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        
        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file {uploaded_file.name}: {e}", exc_info=True)
        return None

def save_app_state(df_path=None, schema_context=None, auto_initialize_llm=None, set_session_state=True):
    """
    Saves the current application state to a config file.
    
    Args:
        df_path: Path to the saved DataFrame CSV file
        schema_context: The schema context provided by the user
        auto_initialize_llm: Whether to auto-initialize the LLM on startup
        set_session_state: Whether to update session state with these values
    """
    try:
        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # Create the config dictionary
        config = {}
        
        if df_path is not None:
            config["df_path"] = df_path
        
        if schema_context is not None:
            config["schema_context"] = schema_context
            
        if auto_initialize_llm is not None:
            config["auto_initialize_llm"] = auto_initialize_llm
        
        # Save to config file
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved application state to {CONFIG_FILE}")
        
        # Update session state if requested
        if set_session_state:
            if df_path is not None:
                st.session_state["saved_df_path"] = df_path
            if schema_context is not None:
                st.session_state["last_schema_context"] = schema_context
            if auto_initialize_llm is not None:
                st.session_state["auto_initialize_llm"] = auto_initialize_llm
        
        return True
    except Exception as e:
        logger.error(f"Failed to save application state: {e}", exc_info=True)
        return False

def load_app_state(set_session_state=True):
    """
    Loads the saved application state from the config file.
    
    Args:
        set_session_state: Whether to update session state with loaded values
        
    Returns:
        A dictionary containing the loaded state, or empty dict if no state found
    """
    if not os.path.exists(CONFIG_FILE):
        logger.info(f"No saved state found at {CONFIG_FILE}")
        return {}
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        logger.info(f"Loaded application state from {CONFIG_FILE}: {config}")
        
        # Update session state if requested
        if set_session_state:
            if "df_path" in config and os.path.exists(config["df_path"]):
                st.session_state["saved_df_path"] = config["df_path"]
            
            if "schema_context" in config:
                st.session_state["last_schema_context"] = config["schema_context"]
                
            if "auto_initialize_llm" in config:
                st.session_state["auto_initialize_llm"] = config["auto_initialize_llm"]
        
        return config
    except Exception as e:
        logger.error(f"Failed to load application state: {e}", exc_info=True)
        return {}

def load_saved_df():
    """
    Loads a DataFrame from a previously saved CSV file.
    
    Returns:
        A pandas DataFrame if loading is successful, otherwise None.
    """
    if "saved_df_path" not in st.session_state or not st.session_state["saved_df_path"]:
        logger.info("No saved DataFrame path in session state")
        return None
    
    df_path = st.session_state["saved_df_path"]
    
    if not os.path.exists(df_path):
        logger.warning(f"Saved DataFrame path {df_path} does not exist")
        return None
    
    try:
        df = pd.read_csv(df_path)
        logger.info(f"Successfully loaded saved DataFrame from {df_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load saved DataFrame from {df_path}: {e}", exc_info=True)
        return None

def clear_saved_state():
    """
    Clears all saved state including uploaded files and config.
    """
    try:
        # Delete the config file
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
            logger.info(f"Deleted config file {CONFIG_FILE}")
        
        # Clear the uploads directory
        if os.path.exists(UPLOADS_DIR):
            for file in os.listdir(UPLOADS_DIR):
                file_path = os.path.join(UPLOADS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"Cleared uploads directory {UPLOADS_DIR}")
        
        # Clear session state
        if "saved_df_path" in st.session_state:
            del st.session_state["saved_df_path"]
        if "last_schema_context" in st.session_state:
            del st.session_state["last_schema_context"]
        if "df" in st.session_state:
            del st.session_state["df"]
        if "df_loaded" in st.session_state:
            del st.session_state["df_loaded"]
        if "settings_schema_context_input" in st.session_state:
            del st.session_state["settings_schema_context_input"]
        if "auto_initialize_llm" in st.session_state:
            del st.session_state["auto_initialize_llm"]
        if "llm" in st.session_state:
            del st.session_state["llm"]
        if "pai_df" in st.session_state:
            del st.session_state["pai_df"]
        if "pai_df_description" in st.session_state:
            del st.session_state["pai_df_description"]
        
        logger.info("Cleared all saved state")
        return True
    except Exception as e:
        logger.error(f"Failed to clear saved state: {e}", exc_info=True)
        return False

def infer_schema_relationships(df: pd.DataFrame) -> str:
    """
    Infer potential schema relationships from DataFrame columns
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Schema context string with potential relationships
    """
    try:
        if df is None or df.empty:
            return ""
            
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        
        # Look for potential ID columns
        id_columns = [col for col in columns if 'id' in col.lower() or '_id' in col.lower()]
        
        # Look for potential datetime columns
        datetime_columns = []
        for col in columns:
            try:
                if 'date' in col.lower() or 'time' in col.lower():
                    datetime_columns.append(col)
                elif df[col].dtype == 'object':  # Check string columns
                    # Try to parse as datetime
                    pd.to_datetime(df[col], errors='raise')
                    datetime_columns.append(col)
            except:
                pass
        
        # Look for categorical columns
        categorical_columns = []
        for col in columns:
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.5:
                categorical_columns.append(col)
        
        # Look for potential foreign key relationships
        potential_relationships = []
        for id_col in id_columns:
            base_entity = id_col.replace('_id', '').replace('id', '')
            for col in columns:
                if base_entity in col.lower() and col != id_col:
                    potential_relationships.append(f"Column '{col}' may be related to '{id_col}'")
        
        # Generate schema context
        schema_info = [
            "Inferred Schema Information:",
            f"Total columns: {len(columns)}",
            f"Column names: {', '.join(columns)}",
            f"ID columns: {', '.join(id_columns) if id_columns else 'None detected'}",
            f"DateTime columns: {', '.join(datetime_columns) if datetime_columns else 'None detected'}",
            f"Categorical columns: {', '.join(categorical_columns) if categorical_columns else 'None detected'}"
        ]
        
        if potential_relationships:
            schema_info.append("Potential relationships:")
            schema_info.extend([f"- {rel}" for rel in potential_relationships])
        
        return "\n".join(schema_info)
    except Exception as e:
        logger.error(f"Error inferring schema relationships: {e}")
        return "" 