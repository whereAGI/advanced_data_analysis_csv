import streamlit as st
import pandas as pd
import os
import traceback
import logging
import re
import sys
import time
import json
from src import data_handler
from src.config import logger

# Page configuration
st.set_page_config(
    page_title="Chat with Data | PandasAI",
    page_icon="🤖",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    /* Chat container styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 0;
        margin-bottom: 20px;
    }
    
    /* Chat message styles */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0;
        display: flex;
        width: 100%;
    }
    .chat-message.user {
        background-color: #2e4c7c;  /* Darker blue for user messages */
        color: #ffffff;  /* White text for dark background */
    }
    .chat-message.assistant {
        background-color: #1a3152;  /* Even darker blue for assistant */
        color: #ffffff;  /* White text for dark background */
    }
    .chat-message .avatar {
        width: 32px;
        height: 32px;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        font-size: 16px;
    }
    .chat-message .content {
        flex-grow: 1;
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
    
    /* Code block styles */
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: monospace;
    }
    
    /* View Details styles */
    .stExpander {
        margin: 0 !important;
        border-top: none !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 0.85rem !important;
        color: #4b8bf4 !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Input container styles - improved fixed positioning */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #0e1117;
        padding: 1rem;
        border-top: 1px solid rgba(49, 51, 63, 0.2);
        z-index: 1000;
        width: 100%;
        display: flex;
        align-items: center;
    }
    
    /* Improve text area styling to match ChatGPT */
    .stTextArea > div {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 0.5rem !important;
    }
    
    /* Improve text area placeholder */
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Chat history container to provide space for fixed input */
    .chat-history {
        margin-bottom: 7rem;  /* Increased to ensure no overlap */
        padding-bottom: 2rem;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        white-space: pre;
        padding-top: 5px;
        padding-bottom: 5px;
        margin-right: 5px !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4b8bf4;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4b8bf4;
    }
    .stTabs [data-baseweb="tab-content"] {
        padding: 5px 10px;
    }
    
    /* Main title style */
    h1 {
        margin-bottom: 2rem !important;
    }
    
    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }
    
    /* Info message style */
    .stInfo {
        margin-bottom: 1rem !important;
    }
    
    /* Custom send button */
    .stButton > button {
        background-color: #4b8bf4 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        height: 40px !important;
        margin-top: 10px !important;
    }
    
    /* Override for data display styling */
    .stDataFrame {
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat messages and details
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "response_details" not in st.session_state:
    st.session_state.response_details = {}
if "question_logs" not in st.session_state:
    st.session_state.question_logs = {}
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "processing_question_id" not in st.session_state:
    st.session_state.processing_question_id = None
if "continue_processing_done" not in st.session_state:
    st.session_state.continue_processing_done = False
if "continue_processing_started" not in st.session_state:
    st.session_state.continue_processing_started = False

def add_debug_log(message, category="general", question_id=None):
    """Add a debug log entry with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    log_entry = f"[{timestamp}] [{category}] {message}"
    
    # Add to question-specific logs if provided
    if question_id is not None:
        if question_id not in st.session_state.question_logs:
            st.session_state.question_logs[question_id] = []
        st.session_state.question_logs[question_id].append(log_entry)
        
    logger.debug(log_entry)

def process_user_input():
    """Process the user input when submitted"""
    # Skip if already processing or input is empty
    if st.session_state.waiting_for_response:
        return
        
    if not st.session_state.user_input or not st.session_state.user_input.strip():
        return
            
    user_input = st.session_state.user_input.strip()
    st.session_state.user_input = ""  # Clear the input
        
    # Set waiting flag to prevent duplicate processing
    st.session_state.waiting_for_response = True
    
    # Generate a unique ID for this question
    question_id = f"q_{int(time.time() * 1000)}"
    st.session_state.processing_question_id = question_id
    
    # Add user message to history
    add_debug_log(f"User input received: {user_input[:50]}{'...' if len(user_input) > 50 else ''}", "user", question_id)
    st.session_state.chat_messages.append({"content": user_input, "is_user": True, "id": question_id})
    
    # Set flag to prevent multiple continues
    if "continue_processing_started" not in st.session_state:
        st.session_state.continue_processing_started = False

def continue_processing():
    """Continue processing the user query after the rerun"""
    # Prevent repeated processing
    if not st.session_state.waiting_for_response or not st.session_state.processing_question_id:
        return
        
    question_id = st.session_state.processing_question_id
    
    # Get the user input from the last message
    user_messages = [msg for msg in st.session_state.chat_messages if msg.get("is_user", False)]
    if not user_messages:
        st.session_state.waiting_for_response = False
        st.session_state.processing_question_id = None
        return
        
    # Get the last user message
    user_input = user_messages[-1]["content"]
    
    try:
        # Process the query and get response
        add_debug_log("Processing query...", "system", question_id)
        response = process_query(user_input, question_id)
        add_debug_log("Query processing complete", "system", question_id)
        
        # Get and append rate limiter logs if available
        try:
            llm = data_handler._data_store.get("llm")
            if hasattr(llm, "get_debug_info"):
                debug_info = llm.get_debug_info()
                if isinstance(debug_info, dict) and "log" in debug_info:
                    for log_entry in debug_info.get("log", []):
                        add_debug_log(f"Rate Limiter: {log_entry.split('] ')[1] if '] ' in log_entry else log_entry}", "rate_limit", question_id)
        except Exception as e:
            add_debug_log(f"Error retrieving rate limiter logs: {str(e)}", "error", question_id)
        
        # Save assistant response to history
        if isinstance(response, str):
            add_debug_log(f"Saving string response to chat history: {response[:50]}{'...' if len(response) > 50 else ''}", "system", question_id)
            st.session_state.chat_messages.append({"content": response, "is_user": False, "id": question_id})
        else:
            # For non-string responses, save a simple text representation
            response_type = response.get("type", "unknown") if isinstance(response, dict) else "unknown"
            if response_type == "error":
                simple_content = f"Error: {response.get('value', 'Unknown error')}"
                add_debug_log(f"Saving error response to chat history: {simple_content}", "error", question_id)
                st.session_state.chat_messages.append({
                    "content": simple_content, 
                    "is_user": False, 
                    "id": question_id
                })
            elif response_type == "dataframe" and isinstance(response, dict) and "value" in response and "analysis" in response:
                # Extract the analysis for display
                st.session_state.chat_messages.append({
                    "content": response["analysis"], 
                    "is_user": False, 
                    "id": question_id,
                    "data": response.get("value")
                })
            else:
                simple_content = response.get("value", str(response)) if isinstance(response, dict) else str(response)
                st.session_state.chat_messages.append({
                    "content": simple_content, 
                    "is_user": False, 
                    "id": question_id
                })
        
        # Process response for view details
        handle_response_details(response, question_id)
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        add_debug_log(error_msg, "error", question_id)
        logger.error(f"Error in continue_processing: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Add error message to chat
        st.session_state.chat_messages.append({
            "content": f"Unfortunately, I was not able to answer your question, because of the following error:\n{str(e)}", 
            "is_user": False, 
            "id": question_id
        })
        
        # Create basic details for the error
        error_details = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": None,
            "query": None,
            "data": None,
            "token_usage": None,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        st.session_state.response_details[question_id] = error_details
    
    finally:
        # Reset flags
        st.session_state.waiting_for_response = False
        st.session_state.processing_question_id = None
        st.session_state.continue_processing_started = False
        st.session_state.continue_processing_done = False
        # Force a rerun to update the UI
        st.rerun()

def handle_response_details(response, question_id):
    """Extract and store details from response for view details section"""
    # Create response details dictionary
    details = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reasoning": None,
        "query": None,
        "data": None,
        "token_usage": None,
        "data_truncated": False,
        "full_size": None,
        "sample_size": None,
        "estimated_tokens": None,
        "token_limit": None
    }
    
    # Extract information based on response type
    if isinstance(response, str):
        # Extract reasoning and query from string response
        details["reasoning"] = extract_thinking(response)
        details["query"] = extract_query(response)
    elif isinstance(response, dict):
        # Extract from dictionary response
        if "reasoning" in response:
            details["reasoning"] = response["reasoning"]
        elif "thinking" in response:
            details["reasoning"] = response["thinking"]
            
        if "query" in response:
            details["query"] = response["query"]
        else:
            details["query"] = extract_query(str(response.get("value", "")), response)
            
        if "token_usage" in response:
            details["token_usage"] = response["token_usage"]
            
        if "value" in response and response.get("type") == "dataframe":
            details["data"] = response["value"]
            
        # Store error information if present
        if "error" in response or (response.get("type") == "error" and "value" in response):
            details["error"] = response.get("error", response.get("value", "Unknown error"))
            
        # Store data truncation information if available
        if "data_truncated" in response:
            details["data_truncated"] = response["data_truncated"]
            
        # Store additional dataset size information
        for field in ["full_size", "sample_size", "estimated_tokens", "token_limit"]:
            if field in response:
                details[field] = response[field]
    
    # Store details for this question
    if question_id:
        st.session_state.response_details[question_id] = details

def display_chat_message(message):
    """Display a chat message with proper formatting"""
    content = message["content"]
    is_user = message.get("is_user", False)
    message_id = message.get("id")
    
    avatar = "👤" if is_user else "🤖"
    message_type = "user" if is_user else "assistant"
    
    # Format content for display
    formatted_content = format_message(content)
    
    # Create message HTML
    message_html = f"""
    <div class="chat-message {message_type}">
        <div class="avatar">{avatar}</div>
        <div class="content">{formatted_content}</div>
    </div>
    """
    
    st.markdown(message_html, unsafe_allow_html=True)
    
    # Add View Details expander for assistant messages
    if not is_user and message_id and message_id in st.session_state.response_details:
        display_view_details(message_id)
        
    # Note: We no longer display the dataframes outside the View Details to avoid duplication

def display_view_details(message_id):
    """Display the View Details expander for a message"""
    details = st.session_state.response_details.get(message_id)
    if not details:
        return
    
    # Create the View Details expander
    with st.expander("🔍 View Details", expanded=False):
        # Show data truncation warning if applicable
        if isinstance(details.get("data_truncated"), bool) and details.get("data_truncated"):
            full_size = details.get("full_size", 0)
            sample_size = details.get("sample_size", 5)
            estimated_tokens = details.get("estimated_tokens", 0)
            token_limit = details.get("token_limit", 4000)
            
            st.warning(f"""
            ⚠️ **Large Dataset Detected**
            
            The complete dataset ({full_size} rows, ~{estimated_tokens} tokens) exceeded the token limit ({token_limit}).
            
            The analysis is based on:
            - Statistics calculated from the **entire** dataset
            - A sample of {sample_size} rows sent to the model
            
            This ensures accurate analysis while respecting token limits.
            """)
        
        # Create tabs
        tab_icons = ["💭", "🔍", "📊", "🐞", "📈"]
        tab_names = ["Reasoning", "Query", "Data", "Debug Log", "Token Usage"]
        tab_labels = [f"{icon} {name}" for icon, name in zip(tab_icons, tab_names)]
        
        tabs = st.tabs(tab_labels)
        
        # Reasoning tab content
        with tabs[0]:
            if details.get("reasoning"):
                st.markdown("### Reasoning Process")
                st.markdown(details["reasoning"])
            else:
                st.info("No reasoning information available for this response.")
        
        # Query tab content - Show only the generated query
        with tabs[1]:
            if details.get("query"):
                st.markdown("### Generated Query")
                st.code(details["query"], language="python")
            else:
                st.info("No query information available for this response.")
        
        # Data tab content
        with tabs[2]:
            if details.get("data") is not None:
                st.markdown("### Raw Data")
                
                # Add data truncation information
                if isinstance(details.get("data_truncated"), bool) and details.get("data_truncated"):
                    full_size = details.get("full_size", 0)
                    sample_size = details.get("sample_size", 5)
                    
                    st.info(f"""
                    **Note:** Showing a sample of {sample_size} rows from the full dataset ({full_size} rows).
                    The statistics in the analysis were calculated using the entire dataset.
                    """)
                
                if isinstance(details["data"], pd.DataFrame):
                    # Add row count information
                    row_count = len(details["data"])
                    if row_count < 1000:
                        st.dataframe(details["data"])
                    else:
                        st.warning(f"Dataset is large ({row_count} rows). Showing first 1000 rows.")
                        st.dataframe(details["data"].head(1000))
                elif isinstance(details["data"], list):
                    # Add row count information for list data
                    row_count = len(details["data"])
                    if row_count < 1000:
                        st.dataframe(pd.DataFrame(details["data"]))
                    else:
                        st.warning(f"Dataset is large ({row_count} rows). Showing first 1000 rows.")
                        st.dataframe(pd.DataFrame(details["data"][:1000]))
                else:
                    st.json(details["data"])
            else:
                st.info("No raw data available for this response.")
        
        # Debug Log tab content
        with tabs[3]:
            if message_id in st.session_state.question_logs:
                st.markdown("### Debug Log")
                for log in st.session_state.question_logs[message_id]:
                    # Color-code different log types
                    if "[error]" in log.lower():
                        st.markdown(f"<span style='color: #ff6b6b; font-family: monospace;'>{log}</span>", unsafe_allow_html=True)
                    elif "[warning]" in log.lower():
                        st.markdown(f"<span style='color: #ffd93d; font-family: monospace;'>{log}</span>", unsafe_allow_html=True)
                    elif "[system]" in log.lower():
                        st.markdown(f"<span style='color: #4dabf7; font-family: monospace;'>{log}</span>", unsafe_allow_html=True)
                    elif "[rate_limit]" in log.lower():
                        st.markdown(f"<span style='color: #da77f2; font-family: monospace;'>{log}</span>", unsafe_allow_html=True)
                    elif "[api]" in log.lower():
                        st.markdown(f"<span style='color: #69db7c; font-family: monospace;'>{log}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='font-family: monospace;'>{log}</span>", unsafe_allow_html=True)
                
                # Show error traceback if available
                if details.get("error"):
                    st.markdown("### Error Details")
                    st.markdown(f"<span style='color: #ff6b6b; font-family: monospace;'>{details['error']}</span>", unsafe_allow_html=True)
                    if details.get("traceback"):
                        with st.expander("Show Error Traceback"):
                            st.code(details["traceback"], language="python")
            else:
                st.info("No debug logs available for this response.")
        
        # Token Usage tab content
        with tabs[4]:
            if details.get("token_usage"):
                st.markdown("### Token Usage")
                st.json(details["token_usage"])
                
                # Add dataset token information if available
                if details.get("estimated_tokens"):
                    st.markdown("### Dataset Token Estimation")
                    st.info(f"""
                    **Dataset Size Information:**
                    - Estimated tokens in full dataset: {details.get('estimated_tokens', 0)}
                    - Token limit for analysis: {details.get('token_limit', 4000)}
                    - Data truncated: {details.get('data_truncated', False)}
                    """)
            else:
                token_info = extract_token_usage(message_id)
                if token_info:
                    st.markdown("### Estimated Token Usage")
                    st.json(token_info)
                else:
                    st.info("No token usage information available for this response.")

def main():
    """Main function for the Chat page"""
    st.title("💬 Chat with your Data")
    
    # Initialize session state if needed
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "response_details" not in st.session_state:
        st.session_state.response_details = {}
    if "question_logs" not in st.session_state:
        st.session_state.question_logs = {}
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
    if "processing_question_id" not in st.session_state:
        st.session_state.processing_question_id = None
    if "continue_processing_started" not in st.session_state:
        st.session_state.continue_processing_started = False
    if "continue_processing_done" not in st.session_state:
        st.session_state.continue_processing_done = False
    
    # Check if data is loaded
    df = data_handler.get_dataframe()
    if df is None:
        st.warning("⚠️ No data loaded. Please upload a CSV file in the Settings page.")
        return
    
    # Display loaded data info
    csv_path = data_handler._data_store.get("csv_path", "")
    if csv_path:
        file_name = os.path.basename(csv_path)
        st.info(f"📊 Using data from: {file_name} ({df.shape[0]} rows, {df.shape[1]} columns)")
    
    # Check if PandasAI is available
    if not data_handler.PANDASAI_AVAILABLE:
        st.error("PandasAI is not available. Please install it with: pip install pandasai==1.5.8")
        return
    
    # Create chat history container
    chat_container = st.container()
    
    # Display chat history in the container first
    with chat_container:
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display all historical messages
        for message in st.session_state.chat_messages:
            display_chat_message(message)
            
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Continue processing if we were waiting for a response
    if st.session_state.waiting_for_response and st.session_state.processing_question_id:
        with st.spinner("Thinking..."):
            if not st.session_state.continue_processing_started:
                continue_processing()
    
    # Chat input at the bottom with better styling
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1])
    with col1:
        st.text_area("Ask a question about your data:", 
                    key="user_input", 
                    height=80,
                    placeholder="Ask a question about your data...",
                    label_visibility="collapsed")
    with col2:
        st.button("Send", 
                on_click=process_user_input,
                use_container_width=True,
                disabled=st.session_state.waiting_for_response)  # Disable button while processing
    st.markdown('</div>', unsafe_allow_html=True)

def format_message(content):
    """Format special content in messages (like code)"""
    if isinstance(content, str):
        # Handle code blocks with triple backticks
        content = re.sub(
            r"```(.*?)```", 
            r'<div class="code-block">\1</div>', 
            content, 
            flags=re.DOTALL
        )
    
    return content

def process_query(prompt: str, question_id=None):
    """
    Process a user query using PandasAI
    
    Args:
        prompt: The user's question
        question_id: ID to associate logs with a specific question
        
    Returns:
        Response from PandasAI (structured or string)
    """
    try:
        add_debug_log(f"Processing query: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", "query", question_id)
        
        # Make sure we have a DataFrame loaded
        df = data_handler.get_dataframe()
        if df is None:
            add_debug_log("No dataset loaded", "error", question_id)
            return "Please load a dataset first in the Settings page."
        
        add_debug_log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns", "data", question_id)
        
        # Check if API key is set
        api_key = data_handler._data_store.get("api_key")
        if not api_key:
            add_debug_log("API key not set", "error", question_id)
            return "API key not set. Please enter your API key in the Settings page."
        
        add_debug_log("API key available", "auth", question_id)
        
        # Get PandasAI instance
        pandas_ai = data_handler.get_pandasai()
        if pandas_ai is None:
            # Try to initialize PandasAI
            add_debug_log("PandasAI not initialized, attempting to initialize...", "system", question_id)
            logger.info("PandasAI not initialized, attempting to initialize...")
            success = data_handler.init_pandasai()
            logger.info(f"Initialization result: {success}")
            add_debug_log(f"PandasAI initialization result: {success}", "system", question_id)
            if not success:
                add_debug_log("Failed to initialize PandasAI", "error", question_id)
                return "PandasAI not initialized. Please check API key in Settings."
            pandas_ai = data_handler.get_pandasai()
            if pandas_ai is None:
                add_debug_log("Failed to get PandasAI instance after initialization", "error", question_id)
                return "Failed to initialize PandasAI."
        
        # Run query
        add_debug_log("Sending query to PandasAI", "api", question_id)
        logger.info(f"Running query: {prompt}")
        
        # Get access to the LLM for query tracking and rate limit checking
        llm = data_handler._data_store.get("llm")
        
        # Reset query tracking
        if llm and hasattr(llm, "last_query"):
            llm.last_query = None
        
        # IMPORTANT: Check rate limit status BEFORE making the API call
        if llm and hasattr(llm, "rate_limiter"):
            try:
                # Get current token usage
                current_tokens = llm.rate_limiter.get_current_tokens()
                token_limit = llm.rate_limiter.rate_limit_tokens
                add_debug_log(f"Current token usage before API call: {current_tokens}/{token_limit} tokens in last minute", "rate_limit", question_id)
                
                # Force waiting before proceeding if close to limit
                if current_tokens > (token_limit * 0.9):  # If we're at 90% of the limit
                    add_debug_log(f"Close to rate limit ({current_tokens}/{token_limit}). Enforcing wait...", "rate_limit", question_id)
                    llm.rate_limiter.wait_if_needed()
                    # Re-check after waiting
                    current_tokens = llm.rate_limiter.get_current_tokens()
                    add_debug_log(f"After waiting, token usage is now: {current_tokens}/{token_limit}", "rate_limit", question_id)
            except Exception as e:
                add_debug_log(f"Error checking rate limit: {str(e)}", "error", question_id)
        
        # Run the query with better error handling for rate limits
        try:
            result = data_handler.run_pandasai_query(prompt)
            logger.info(f"Query result type: {type(result)}")
        except Exception as e:
            error_message = str(e)
            add_debug_log(f"Error during API call: {error_message}", "error", question_id)
            
            # Handle rate limit errors specifically
            if "429" in error_message or "Too Many Requests" in error_message:
                add_debug_log("Rate limit exceeded. Adding delay before next request.", "rate_limit", question_id)
                
                # Get current token usage for better error message
                token_info = ""
                try:
                    if llm and hasattr(llm, "rate_limiter"):
                        current_tokens = llm.rate_limiter.get_current_tokens()
                        token_limit = llm.rate_limiter.rate_limit_tokens
                        token_info = f" (Current usage: {current_tokens}/{token_limit} tokens in last minute)"
                        
                        # Clear the token log to reset the rate limiter
                        if hasattr(llm.rate_limiter, "token_log"):
                            llm.rate_limiter.token_log.clear()
                            add_debug_log("Cleared rate limiter token log due to 429 error", "rate_limit", question_id)
                            
                        # Add a penalty entry to force waiting
                        if hasattr(llm.rate_limiter, "log_usage"):
                            llm.rate_limiter.log_usage(token_limit)
                            add_debug_log(f"Added penalty of {token_limit} tokens to enforce backoff", "rate_limit", question_id)
                except Exception as e:
                    add_debug_log(f"Error handling rate limit: {str(e)}", "error", question_id)
                    
                delay_time = 60  # Default full minute wait
                add_debug_log(f"Rate limit exceeded. Waiting {delay_time} seconds before next request{token_info}", "rate_limit", question_id)
                
                # Wait for a moment to make UI responsive
                time.sleep(5)
                
                return {
                    "type": "error", 
                    "value": f"Unable to analyze the data due to rate limit exceeded. The API allows 6000 tokens per minute. Please wait 60 seconds before trying again{token_info}.",
                    "error": "rate_limit"
                }
            
            # Other API errors
            return {"type": "error", "value": f"Error: {error_message}"}
        
        # Try to get the query from the LLM
        try:
            if llm and hasattr(llm, "last_query") and llm.last_query:
                # If result is a dict, add the query to it
                if isinstance(result, dict):
                    result["query"] = extract_pandas_code(llm.last_query)
                    add_debug_log(f"Added query to result: {result['query'][:50] if result['query'] else 'None'}...", "system", question_id)
                
                # If token usage is available, add it too
                if hasattr(llm, "last_token_usage") and llm.last_token_usage:
                    result["token_usage"] = llm.last_token_usage
                    tokens_info = llm.last_token_usage
                    add_debug_log(f"Token usage - Prompt: {tokens_info.get('prompt_tokens', 0)}, Completion: {tokens_info.get('completion_tokens', 0)}, Total: {tokens_info.get('total_tokens', 0)}", "system", question_id)
        except Exception as e:
            add_debug_log(f"Error retrieving query from LLM: {str(e)}", "error", question_id)
        
        # Log result summary
        if isinstance(result, dict):
            result_type = result.get("type", "unknown")
            add_debug_log(f"Received result of type: {result_type}", "api", question_id)
            if result_type == "dataframe" and "value" in result:
                df_result = result["value"]
                if isinstance(df_result, pd.DataFrame):
                    add_debug_log(f"DataFrame result: {df_result.shape[0]} rows, {df_result.shape[1]} columns", "data", question_id)
        else:
            add_debug_log(f"Received string result of length: {len(str(result))}", "api", question_id)
        
        # Return result (structured or string)
        return result
    except Exception as e:
        error_msg = str(e)
        add_debug_log(f"Error processing query: {error_msg}", "error", question_id)
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return {"type": "error", "value": error_msg}

def extract_thinking(text):
    """Extract thinking sections from response text using regex"""
    if not text or not isinstance(text, str):
        return None
        
    # Look for <thinking> tags
    thinking_pattern = r"<thinking>(.*?)</thinking>"
    thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
    
    if thinking_matches:
        return "\n\n".join(thinking_matches)
    
    # Look for alternative formats like "Thinking:" or "Let me think..."
    alt_patterns = [
        r"(?:Thinking|Reasoning):\s*(.*?)(?:\n\n|$)",
        r"Let me (?:think|reason).*?:\s*(.*?)(?:\n\n|$)",
        r"(?:First|Let's) analyze.*?:\s*(.*?)(?:\n\n|$)",
        r"To answer.*?(?:I'll|I will|Let me).*?:\s*(.*?)(?:\n\n|$)",
        r"(?:Here's|Let's start with) my (?:thinking|analysis).*?:\s*(.*?)(?:\n\n|$)"
    ]
    
    all_matches = []
    for pattern in alt_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        all_matches.extend(matches)
    
    if all_matches:
        return "\n\n".join(all_matches)
    
    return None

def extract_pandas_code(text):
    """Extract only pandas code without thinking sections or explanations"""
    if not text or not isinstance(text, str):
        return None
    
    # Look for result = {...} patterns which are common in pandas code output
    result_pattern = r"result\s*=\s*\{.*?\}"
    
    # First look for Python code blocks
    code_patterns = [
        # Code between ```python and ```
        r"```python\s*(.*?)\s*```",
        # Any code blocks
        r"```\s*(.*?)\s*```",
        # Python sections after "Python code:"
        r"Python code:\s*(.*?)(?:\n\n|$)"
    ]
    
    # Find the first successful match that contains pandas operations
    for pattern in code_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Check if this is pandas code
            if "import pandas" in match or "df" in match or "pd." in match:
                # Clean up the code
                code = match.strip()
                # Remove thinking or explanation comments
                code = re.sub(r"#.*?thinking.*?\n", "\n", code, flags=re.IGNORECASE)
                code = re.sub(r"#.*?explanation.*?\n", "\n", code, flags=re.IGNORECASE)
                # Remove multiple empty lines
                code = re.sub(r"\n\s*\n+", "\n\n", code)
                return code
    
    # If we didn't find good pandas code, try to extract just the relevant parts
    # Look for sections with dataframe operations
    df_sections = re.findall(r"(?:df|result)\s*=.*?(?:\n\n|$)", text, re.DOTALL)
    if df_sections:
        return "\n".join(df_sections)
    
    # If all else fails, return the original query but strip explanation text
    return re.sub(r".*?(import pandas|df =).*?", r"\1", text, flags=re.DOTALL)

def extract_query(text, response_dict=None):
    """Extract query information from response"""
    # First check if the response_dict has a 'query' key
    if response_dict and isinstance(response_dict, dict) and 'query' in response_dict:
        return response_dict['query']
    
    # Otherwise try to extract from text using the more precise pandas code extraction
    return extract_pandas_code(text)

def extract_token_usage(question_id):
    """Extract token usage information from logs for a question"""
    if question_id is None or question_id not in st.session_state.question_logs:
        return None
        
    token_info = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    
    # Try to extract token information from logs
    logs = st.session_state.question_logs[question_id]
    for log in logs:
        if "Token usage" in log:
            matches = re.search(r"Prompt: (\d+), Completion: (\d+), Total: (\d+)", log)
            if matches:
                token_info["prompt_tokens"] += int(matches.group(1))
                token_info["completion_tokens"] += int(matches.group(2))
                token_info["total_tokens"] += int(matches.group(3))
    
    return token_info if token_info["total_tokens"] > 0 else None
    
if __name__ == "__main__":
    main() 