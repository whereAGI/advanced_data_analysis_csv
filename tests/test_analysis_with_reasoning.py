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

# Create a test dataset with more columns for a detailed analysis
data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'username': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10'],
    'age': [25, 34, 28, 45, 19, 31, 27, 36, 42, 21],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'country': ['US', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Spain', 'Italy', 'Japan', 'Brazil'],
    'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', 
                   '2023-06-18', '2023-07-22', '2023-08-30', '2023-09-14', '2023-10-05'],
    'total_orders': [12, 8, 5, 20, 3, 15, 7, 10, 9, 4],
    'total_spending': [876.25, 451.75, 320.00, 1450.50, 125.90, 942.60, 380.25, 690.10, 550.75, 275.40],
    'avg_order_value': [73.02, 56.47, 64.00, 72.53, 41.97, 62.84, 54.32, 69.01, 61.19, 68.85],
    'preferred_category': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports', 'Beauty', 'Toys', 'Food', 'Garden', 'Automotive']
}
df = pd.DataFrame(data)

logger.info(f"Created test DataFrame with shape: {df.shape}")
logger.info(f"Columns: {df.columns.tolist()}")

# Set up schema context
schema_context = """
This data represents users in an e-commerce system:
- user_id: Primary identifier for users
- username: User's username in the system
- age: User's age in years
- gender: User's gender (M/F)
- country: User's country of residence
- signup_date: When the user created their account
- total_orders: Number of orders the user has placed
- total_spending: Total amount spent by the user
- avg_order_value: Average value of each order placed by the user
- preferred_category: The product category the user shops most frequently
"""

# Test with our GroqLLM
try:
    logger.info("\n=== Testing DataFrame analysis with reasoning extraction ===")
    from src.groq_llm import GroqLLM
    from src.data_handler import analyze_result_dataframe
    
    # Create LLM
    llm = GroqLLM(api_token=api_key, model="llama3-8b-8192")
    
    # Add the LLM to the _data_store for the test
    from src.data_handler import _data_store
    _data_store["llm"] = llm
    
    # Test with a complex query that requires detailed analysis
    query = "What insights can we derive about shopping habits based on age and country?"
    logger.info(f"Testing analysis for complex query: {query}")
    
    # Direct analysis with expected reasoning section
    logger.info("Performing direct LLM analysis with reasoning...")
    analysis_result = llm.analyze_dataframe(df, query, schema_context)
    
    # Check the format of the result
    if isinstance(analysis_result, dict):
        logger.info("✅ Analysis result returned as a dictionary with separate sections")
        
        if "analysis" in analysis_result:
            analysis_preview = analysis_result["analysis"][:200] + "..." if len(analysis_result["analysis"]) > 200 else analysis_result["analysis"]
            logger.info(f"Analysis content: {analysis_preview}")
        else:
            logger.error("❌ No analysis section in result")
            
        if "reasoning" in analysis_result:
            reasoning_preview = analysis_result["reasoning"][:200] + "..." if len(analysis_result["reasoning"]) > 200 else analysis_result["reasoning"]
            logger.info(f"Reasoning content: {reasoning_preview}")
        else:
            logger.warning("⚠️ No reasoning section in result")
    else:
        logger.error(f"❌ Analysis result not returned as a dictionary: {type(analysis_result)}")
    
    # Test with analyze_result_dataframe function
    logger.info("\nTesting analyze_result_dataframe function...")
    result = analyze_result_dataframe(df, query, schema_context)
    
    # Check structure of the final result
    logger.info(f"Result structure: {list(result.keys())}")
    
    # Check if analysis and reasoning were added
    if "analysis" in result:
        logger.info("✅ Analysis added to result")
    else:
        logger.error("❌ Analysis not added to result")
        
    if "reasoning" in result:
        logger.info("✅ Reasoning added to result")
    else:
        logger.error("❌ Reasoning not added to result")
    
    logger.info("Analysis with reasoning test completed successfully")
except Exception as e:
    logger.error(f"Test FAILED: {e}")
    import traceback
    traceback.print_exc()

logger.info("\nTest script completed.") 