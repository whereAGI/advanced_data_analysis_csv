# Data Chat

An interactive data analysis application that lets you chat with your data using natural language queries.

## Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Automatic Visualizations**: Get charts and graphs based on your data analysis
- **Data Persistence**: Your settings and data are saved between sessions
- **Custom Schema Context**: Provide additional information to improve AI understanding
- **API Integration**: Uses Groq's powerful LLMs for natural language understanding

## Technology Stack

- **Streamlit**: For the web interface
- **PandasAI**: For natural language to data analysis
- **Pandas**: For data manipulation and analysis
- **Plotly**: For interactive visualizations
- **Groq**: For fast and efficient language model processing

## Setup

### Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- Pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data_chat.git
cd data_chat
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Go to the Settings page to:
   - Upload your CSV data
   - Set your API key
   - Add schema context

3. Visit the Chat page to start asking questions about your data

## Example Queries

- "Show me the distribution of values in column X"
- "What's the average of column Y grouped by column Z?"
- "Find the top 5 rows with the highest values in column A"
- "Plot the trend of column B over time"

## Running Tests

The project includes a comprehensive test suite to ensure functionality:

### Running tests on Windows:
```bash
.\run_tests.bat
```

### Running tests with pytest directly:
```bash
python -m pytest tests/
```

### Running with coverage:
```bash
python -m pytest tests/ --cov=src/ --cov-report=term
```

## Project Structure

```
data_chat/
├── app.py              # Main Streamlit application entry point
├── pages/
│   ├── 1_Settings.py   # Settings page for configuration
│   └── 2_Chat.py       # Chat interface for data analysis
├── src/
│   ├── config.py       # Configuration settings
│   ├── data_handler.py # Data handling functions
│   └── groq_llm.py     # Custom Groq LLM implementation for PandasAI
├── tests/
│   ├── test_config.py          # Tests for config module
│   ├── test_data_handler.py    # Tests for data_handler module
│   └── run_tests.py            # Test runner script
├── .github/
│   └── workflows/
│       └── python-tests.yml    # GitHub Actions workflow
├── .env                # Environment variables (not in repository)
├── requirements.txt    # Python dependencies
├── pytest.ini         # pytest configuration
└── README.md          # This file
```

## Continuous Integration

The project uses GitHub Actions for continuous integration:
- Automatic testing on multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Test coverage reporting
- Triggered on push to main and pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PandasAI](https://github.com/sinaptik-ai/pandas-ai) for the amazing conversational data analysis library
- [Streamlit](https://streamlit.io/) for the web application framework
- [Groq](https://groq.com/) for the LLM API
