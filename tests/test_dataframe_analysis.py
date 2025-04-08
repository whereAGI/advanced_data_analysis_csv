import os
import pandas as pd
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.error("ERROR: No GROQ_API_KEY found in .env file")
    exit(1)

logger.info(f"API key found (length: {len(api_key)})")

# Create a test dataset
data = {
    'user_id': [1, 2, 3, 4, 5],
    'username': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'email': ['user1@example.com', 'user2@example.com', 'user3@example.com', 'user4@example.com', 'user5@example.com'],
    'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
    'total_orders': [3, 2, 1, 1, 1],
    'total_spending': [376.25, 251.25, 620.00, 175.25, 440.00]
}
df = pd.DataFrame(data)

logger.info(f"Created test DataFrame with shape: {df.shape}")
logger.info(f"Columns: {df.columns.tolist()}")

# Set up schema context
schema_context = """
This data represents users in an e-commerce system:
- user_id: Primary identifier for users
- username: User's username in the system
- email: User's email address
- signup_date: When the user created their account
- total_orders: Number of orders the user has placed
- total_spending: Total amount spent by the user
"""

# Test with our GroqLLM
try:
    logger.info("\n=== Testing DataFrame analysis with GroqLLM ===")
    from src.groq_llm import GroqLLM
    from src.data_handler import analyze_result_dataframe
    import sys
    
    # Create LLM
    llm = GroqLLM(api_token=api_key, model="llama3-8b-8192")
    
    # Test direct analysis
    query = "Show me total spending by users"
    logger.info(f"Testing analysis for query: {query}")
    
    # Direct analysis
    logger.info("Performing direct LLM analysis...")
    analysis = llm.analyze_dataframe(df, query, schema_context)
    logger.info(f"Analysis result:\n{analysis}\n")
    
    # Add the LLM to the _data_store for the test
    # We need to do this since analyze_result_dataframe gets the LLM from _data_store
    from src.data_handler import _data_store
    _data_store["llm"] = llm
    
    # Test with analyze_result_dataframe function
    logger.info("Testing analyze_result_dataframe function...")
    result = analyze_result_dataframe(df, query, schema_context)
    
    # Check if analysis was added
    if "analysis" in result:
        logger.info("✅ Analysis successfully added to result")
        logger.info(f"Result structure: {list(result.keys())}")
        logger.info(f"DataFrame shape: {result['value'].shape}")
        
        # Check analysis content
        analysis_preview = result["analysis"][:200] + "..." if len(result["analysis"]) > 200 else result["analysis"]
        logger.info(f"Analysis preview: {analysis_preview}")
    else:
        logger.error("❌ Analysis was not added to result")
    
    logger.info("DataFrame analysis test completed successfully")
except Exception as e:
    logger.error(f"Test FAILED: {e}")
    import traceback
    traceback.print_exc()

logger.info("\nTest script completed.") 