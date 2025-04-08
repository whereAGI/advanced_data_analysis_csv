# Contributing to Data Chat

Thank you for considering contributing to Data Chat! This guide will help you get started with development, testing, and submitting your changes.

## Development Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/data_chat.git
   cd data_chat
   ```

3. Set up a virtual environment:
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

5. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Code Style

We follow PEP 8 guidelines for Python code. Please use the following tools to ensure your code meets the standards:

- Use Black for code formatting:
  ```bash
  black src/ tests/
  ```

- Use Flake8 for linting:
  ```bash
  flake8 src/ tests/
  ```

## Running Tests

Run tests using pytest:

```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=src/ --cov-report=term

# Run a specific test
python -m pytest tests/test_data_handler.py::TestDataHandler::test_load_csv
```

On Windows, you can also use the provided batch script:

```bash
.\run_tests.bat
```

## Pull Request Process

1. Create a new branch from main:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

3. Run the tests to ensure your changes don't break existing functionality:
   ```bash
   python -m pytest tests/
   ```

4. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request against the main branch of the original repository.

6. Describe your changes in the pull request description, explaining what the changes do and why they are needed.

7. Wait for the GitHub Actions tests to pass. Address any issues that arise.

8. Once tests pass and the pull request is approved, it will be merged.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Include as much information as possible:

- For bugs: steps to reproduce, expected behavior, actual behavior, and environment information
- For features: clear description of the feature and why it would be valuable

## Code of Conduct

Please be respectful and constructive in all interactions. We want to maintain a welcoming and inclusive community for all contributors. 