# Advanced Data Analysis (CSV)

An intelligent data exploration platform that transforms how you interact with your data through natural language. Ask complex questions in plain English and get instant visualizations, insights, and analysis without writing a single line of code.

## 🚀 Overview

Data Chat bridges the gap between data analysis and natural conversation, making data exploration accessible to everyone regardless of their technical background. By combining the power of cutting-edge language models with robust data analysis tools, Data Chat enables you to:

- **Discover insights** from complex datasets through simple conversations
- **Visualize trends** and patterns automatically based on your questions  
- **Make data-driven decisions** faster without needing specialized programming skills
- **Share insights** with stakeholders using interactive visualizations

## ✨ Features

- **Natural Language Queries**: Ask questions about your data in plain English and get immediate answers
- **Intelligent Analysis**: The system understands context and can perform complex analyses based on conversational prompts
- **Automatic Visualizations**: Get appropriate charts and graphs based on your queries without specifying visualization type
- **Data Persistence**: Your settings, data, and conversation history are saved between sessions
- **Custom Schema Context**: Provide additional information about your data to improve AI understanding
- **Multi-Model Support**: Choose from various language models to balance performance and cost
- **Interactive Interface**: User-friendly Streamlit interface with separate settings and chat pages

## 🛠️ Technology Stack

- **Streamlit**: For the intuitive web interface
- **PandasAI**: For transforming natural language to data analysis
- **Pandas**: For robust data manipulation and analysis
- **Plotly**: For interactive and beautiful visualizations
- **Groq**: For fast and efficient language model processing

## 🔧 Setup

### Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- Pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/whereAGI/advanced_data_analysis_csv.git
cd advanced_data_analysis_csv
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Set API key through environment:
   - Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   - Or, set up the API key through the Settings page (recommended)

## 🚀 Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Go to the Settings page to:
   - Upload your CSV data
   - Set your API key (in the API Settings tab)
   - Choose your preferred model
   - Add schema context

3. Visit the Chat page to start asking questions about your data

## 🔑 Setting Up API Access

This application requires an API key from Groq to function. You have two options for setting up your API key:

### Option 1: Settings Page (Recommended)
1. Launch the application with `streamlit run app.py`
2. Navigate to the Settings page
3. In the "API Settings" tab, enter your Groq API key in the password field
4. Click "Save Settings" to store your key securely
5. Your key will be stored in your user directory and loaded automatically in future sessions

### Option 2: Environment File
1. Create a file named `.env` in the project root directory
2. Add your Groq API key: `GROQ_API_KEY=your_key_here`
3. The application will automatically load this key when it starts

**Note**: When both options are used, the key from the Settings page takes precedence.

## 📊 Example Use Cases

### Business Analytics
- **Sales Performance**: "Show me monthly sales trends compared to last year and highlight months with significant growth"
- **Customer Segmentation**: "Identify customer segments with highest lifetime value and their common characteristics"
- **Inventory Analysis**: "Which products have the highest stock turnover rates in the last quarter?"
- **Marketing ROI**: "Compare conversion rates across different marketing channels and calculate ROI for each"

### Financial Analysis
- **Expense Tracking**: "Create a pie chart of my expenses by category and show which categories increased the most"
- **Investment Performance**: "Calculate the annual return on each investment and rank them by risk-adjusted performance"
- **Budget Planning**: "Compare actual spending versus budget across all departments and highlight variances over 10%"
- **Cash Flow Projection**: "Analyze historical cash flow patterns and project next quarter's cash position"

### Research & Academia
- **Data Exploration**: "Identify outliers in my experiment results and show their impact on the overall distribution"
- **Statistical Analysis**: "Run a correlation analysis between variables X, Y, and Z and show the strongest relationships"
- **Literature Review**: "Summarize the key metrics from all studies in my dataset and identify trends over time"
- **Research Validation**: "Compare my experimental results with the control group and calculate statistical significance"

### Personal Data Management
- **Health Tracking**: "Show my average steps per day by month and identify any correlation with my weight changes"
- **Budget Management**: "Analyze my spending habits over the last 6 months and suggest areas to reduce expenses"
- **Time Usage**: "Compare how I spent my time this month versus last month and highlight the biggest changes"
- **Learning Progress**: "Plot my test scores over time and identify subjects where I'm improving the fastest"

## 🧪 Running Tests

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

## 📁 Project Structure

```
advanced_data_analysis_csv/
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
├── .env                # Environment variables (optional, not in repository)
├── requirements.txt    # Python dependencies
├── pytest.ini         # pytest configuration
└── README.md          # This file
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [PandasAI](https://github.com/sinaptik-ai/pandas-ai) for the amazing conversational data analysis library
- [Streamlit](https://streamlit.io/) for the web application framework
- [Groq](https://groq.com/) for the LLM API

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
