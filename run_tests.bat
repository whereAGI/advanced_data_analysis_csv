@echo off
REM Run all tests script for Windows
echo Running all tests...

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Running with system Python.
)

REM Run the tests
python tests\run_tests.py

REM Store the exit code
set EXIT_CODE=%ERRORLEVEL%

REM Deactivate virtual environment if it was activated
if exist .venv\Scripts\deactivate.bat (
    call .venv\Scripts\deactivate.bat
)

REM Return the test exit code
exit /b %EXIT_CODE% 