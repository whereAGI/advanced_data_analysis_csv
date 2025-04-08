import pandas as pd
from langchain_groq import ChatGroq
# Remove agent-related imports
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents.agent_types import AgentType
import logging
from src.config import logger, DEFAULT_MODEL
import streamlit as st

# Rename function and simplify
def create_llm(api_key: str):
    """
    Initializes and returns the ChatGroq LLM instance.

    Args:
        api_key: The Groq API key.

    Returns:
        A ChatGroq instance or None if initialization fails.
    """
    if not api_key:
        logger.error("create_llm called without a valid API key.")
        st.error("LLM initialization failed: Missing API Key.")
        return None

    logger.info(f"Initializing ChatGroq model: {DEFAULT_MODEL}")
    try:
        llm = ChatGroq(
            temperature=0.7, # Allow for more conversational responses
            model_name=DEFAULT_MODEL,
            groq_api_key=api_key
        )
        logger.info("ChatGroq LLM initialized successfully.")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq LLM: {e}", exc_info=True)
        st.error(f"LLM initialization failed: {e}")
        return None 