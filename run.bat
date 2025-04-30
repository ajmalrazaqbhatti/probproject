@echo off
echo Running Streamlit application...

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Please run setup.bat first.
    exit /b 1
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Run the Streamlit app
echo Starting Streamlit application...
streamlit run main.py
