@echo off
REM Run application script for Windows
echo Starting Data Chat application...

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: Virtual environment not found. Running with system Python.
)

REM Run Streamlit application
streamlit run app.py

REM Store the exit code
set EXIT_CODE=%ERRORLEVEL%

REM Deactivate virtual environment if it was activated
if exist .venv\Scripts\deactivate.bat (
    call .venv\Scripts\deactivate.bat
)

exit /b %EXIT_CODE%
