import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get the logger for this application
logger = logging.getLogger('data_chat')

# Prevent duplicate logging
logger.propagate = False

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create handlers that properly handle Unicode characters
console_handler = logging.StreamHandler(sys.stdout)
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/app.log')

# Create formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Set levels
logger.setLevel(logging.DEBUG)  # Capture all levels
console_handler.setLevel(logging.INFO)  # Show INFO and above in console
file_handler.setLevel(logging.DEBUG)  # Log everything to file

# Ensure the handler can handle Unicode characters by setting encoding
try:
    # Explicitly set encoding for both console and file handlers
    console_handler.setStream(sys.stdout)
    console_handler.encoding = 'utf-8'
    file_handler.encoding = 'utf-8'
    
    # Fix potential Unicode issues in logging
    logger.info("Console and file handler encodings set to utf-8")
except Exception as e:
    logger.warning(f"Could not set encoding for log handlers: {e}")

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- App configuration ---
# Define default paths
APP_DATA_DIR = os.path.join(os.path.expanduser('~'), '.data_chat')
STATE_FILE_PATH = os.path.join(APP_DATA_DIR, 'app_state.json')
os.makedirs(APP_DATA_DIR, exist_ok=True)

# Load API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Default PandasAI models
# DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Visualization configuration
VISUALIZATION_CONFIG = {
    "max_rows": 100,
    "max_cols": 20,
    "sample_size": 1000,
    "max_unique_values": 30
}

# Log configuration information
logger.info("Configuration loaded")
logger.info(f"APP_DATA_DIR: {APP_DATA_DIR}")
logger.info(f"GROQ_API_KEY: {'Set' if GROQ_API_KEY else 'Not set'}")

# Get API keys
PANDASAI_API_KEY = os.getenv("PANDASAI_API_KEY")

# Use GROQ_API_KEY as fallback for PANDASAI_API_KEY if not provided
if not PANDASAI_API_KEY and GROQ_API_KEY:
    logger.info("PANDASAI_API_KEY not found, using GROQ_API_KEY as fallback.")
    PANDASAI_API_KEY = GROQ_API_KEY

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables.")

if not PANDASAI_API_KEY:
    logger.error("No API key available for PandasAI. Please set PANDASAI_API_KEY or GROQ_API_KEY in your .env file.")
else:
    logger.info("API key for PandasAI available.")

# Default model settings
logger.info(f"Default model set to: {DEFAULT_MODEL}")

# Docker sandbox settings (if used)
SANDBOX_CONFIG = {
    "enabled": False,              # Whether to use Docker sandbox
    "timeout": 30,                 # Timeout in seconds for sandbox execution
}