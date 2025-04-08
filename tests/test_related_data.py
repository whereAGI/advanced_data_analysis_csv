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

# Create a more complex test dataset with relationships
# Create users table
users_data = {
    'user_id': [1, 2, 3, 4, 5],
    'username': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'email': ['user1@example.com', 'user2@example.com', 'user3@example.com', 'user4@example.com', 'user5@example.com'],
    'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
}
users_df = pd.DataFrame(users_data)

# Create orders table with relationship to users
orders_data = {
    'order_id': [101, 102, 103, 104, 105, 106, 107],
    'user_id': [1, 2, 1, 3, 2, 4, 5],
    'order_date': ['2023-03-10', '2023-03-15', '2023-04-20', '2023-04-25', '2023-05-05', '2023-05-10', '2023-05-15'],
    'total_amount': [150.50, 200.75, 75.25, 310.00, 50.50, 175.25, 220.00]
}
orders_df = pd.DataFrame(orders_data)

# Create order_items table with relationship to orders
order_items_data = {
    'item_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'order_id': [101, 101, 102, 103, 104, 104, 105, 106, 107, 107],
    'product_name': ['Laptop', 'Mouse', 'Monitor', 'Keyboard', 'Headphones', 'Speakers', 'Phone case', 'Charger', 'Tablet', 'Stylus'],
    'quantity': [1, 1, 1, 2, 1, 2, 1, 1, 1, 2],
    'price': [120.50, 30.00, 200.75, 37.50, 310.00, 95, 50.50, 175.25, 200.00, 20.00]
}
order_items_df = pd.DataFrame(order_items_data)

# Combine into a single DataFrame for testing
# In a real scenario, these would be separate tables, but for this test, we'll merge them
users_orders = pd.merge(users_df, orders_df, on='user_id')
full_data = pd.merge(users_orders, order_items_df, on='order_id')

logger.info(f"Created test DataFrame with shape: {full_data.shape}")
logger.info(f"Columns: {full_data.columns.tolist()}")

# Save to CSV for visualization
full_data.to_csv('test_related_data.csv', index=False)
logger.info("Saved test data to test_related_data.csv")

# Try with our GroqLLM
try:
    logger.info("\n=== Testing with custom GroqLLM and related data ===")
    from src.groq_llm import GroqLLM
    from src.data_handler import infer_schema_relationships, set_schema_context
    
    # Create LLM
    llm = GroqLLM(api_token=api_key, model="llama3-8b-8192")
    
    # Get automated schema relationships
    schema_context = infer_schema_relationships(full_data)
    logger.info(f"Inferred schema context:\n{schema_context}")
    
    # Test with a simple query that should include related data
    logger.info("\nTesting query for user1's orders...")
    prompt = "Find all orders made by user1 and include related data about the items purchased"
    
    # Convert to enhanced prompt with schema
    enhanced_prompt = f"""Schema Information: {schema_context}

Data Relationships Guide:
1. When querying specific data, also include relevant related information from the schema
2. Look for relationships between tables/entities in the schema
3. Include columns that provide context to the primary data requested
4. Consider foreign key relationships and join related data when appropriate

User Query: {prompt}"""
    
    # Generate code
    logger.info("Generating code...")
    code = llm.generate_pandas_code(enhanced_prompt, df_name="full_data")
    logger.info(f"Generated code:\n{code}")
    
    # Execute the code
    logger.info("Executing generated code...")
    try:
        local_namespace = {"full_data": full_data.copy(), "pd": pd}
        exec(code, globals(), local_namespace)
        
        if 'result' in local_namespace:
            result = local_namespace['result']
            if isinstance(result, dict) and 'type' in result and 'value' in result:
                if result['type'] == 'dataframe':
                    df_result = result['value']
                    logger.info(f"Result shape: {df_result.shape}")
                    logger.info(f"Result columns: {df_result.columns.tolist()}")
                    logger.info(f"First few rows:\n{df_result.head()}")
                else:
                    logger.info(f"Result: {result['value']}")
            else:
                logger.info(f"Result: {result}")
        else:
            logger.error("No result variable found in executed code")
    except Exception as e:
        logger.error(f"Error executing generated code: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with a different query that should include related data
    logger.info("\nTesting query for most expensive products and related order information...")
    prompt2 = "Find the 3 most expensive products and show who ordered them"
    
    # Convert to enhanced prompt with schema
    enhanced_prompt2 = f"""Schema Information: {schema_context}

Data Relationships Guide:
1. When querying specific data, also include relevant related information from the schema
2. Look for relationships between tables/entities in the schema
3. Include columns that provide context to the primary data requested
4. Consider foreign key relationships and join related data when appropriate

User Query: {prompt2}"""
    
    # Generate code
    logger.info("Generating code...")
    code2 = llm.generate_pandas_code(enhanced_prompt2, df_name="full_data")
    logger.info(f"Generated code:\n{code2}")
    
    # Execute the code
    logger.info("Executing generated code...")
    try:
        local_namespace = {"full_data": full_data.copy(), "pd": pd}
        exec(code2, globals(), local_namespace)
        
        if 'result' in local_namespace:
            result = local_namespace['result']
            if isinstance(result, dict) and 'type' in result and 'value' in result:
                if result['type'] == 'dataframe':
                    df_result = result['value']
                    logger.info(f"Result shape: {df_result.shape}")
                    logger.info(f"Result columns: {df_result.columns.tolist()}")
                    logger.info(f"First few rows:\n{df_result.head()}")
                else:
                    logger.info(f"Result: {result['value']}")
            else:
                logger.info(f"Result: {result}")
        else:
            logger.error("No result variable found in executed code")
    except Exception as e:
        logger.error(f"Error executing generated code: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with an even more explicit query about related data
    logger.info("\nTesting explicit related data query...")
    prompt3 = "Show total spending by each user, and include their email, signup date, and what products they purchased"
    
    # Convert to enhanced prompt with schema
    enhanced_prompt3 = f"""Schema Information: {schema_context}

Data Relationships Guide:
1. When querying specific data, also include relevant related information from the schema
2. Look for relationships between tables/entities in the schema
3. Include columns that provide context to the primary data requested
4. Consider foreign key relationships and join related data when appropriate

User Query: {prompt3}"""
    
    # Let's manually construct a proper query for this complex case
    logger.info("Using manually constructed query for this complex case...")
    
    # Execute the manually constructed query
    logger.info("Executing manual query...")
    try:
        # Group by user to get total spending
        user_spending = full_data.groupby('user_id')['total_amount'].sum().reset_index()
        user_spending = user_spending.rename(columns={'total_amount': 'total_spending'})
        
        # Get user details (related data - contextual information)
        user_details = full_data[['user_id', 'username', 'email', 'signup_date']].drop_duplicates()
        
        # Merge spending with user details
        result_with_related_data = pd.merge(user_spending, user_details, on='user_id')
        
        # Get products purchased by each user (related data)
        user_products = full_data.groupby('user_id')['product_name'].apply(lambda x: list(set(x))).reset_index()
        user_products = user_products.rename(columns={'product_name': 'products_purchased'})
        
        # Merge with product information
        final_result = pd.merge(result_with_related_data, user_products, on='user_id')
        
        # Create a proper result dictionary in the expected format
        result = {"type": "dataframe", "value": final_result}
        
        # Display result
        logger.info(f"Result shape: {result['value'].shape}")
        logger.info(f"Result columns: {result['value'].columns.tolist()}")
        logger.info(f"Results with related data:\n{result['value']}")
        
    except Exception as e:
        logger.error(f"Error executing manual query: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("GroqLLM related data test completed")
except Exception as e:
    logger.error(f"GroqLLM test FAILED: {e}")
    import traceback
    traceback.print_exc()

logger.info("\nTest script completed.") 