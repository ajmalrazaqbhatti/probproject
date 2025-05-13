@echo off
echo Running Streamlit application...

:: Check if virtual environment exists
if not exist venv (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment
    echo Please run setup.bat to ensure the environment is properly created
    pause
    exit /b 1
)

:: Check if streamlit is installed
where streamlit > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Streamlit is not installed in the virtual environment
    echo Please run setup.bat to install required packages
    pause
    exit /b 1
)

:: Check if main.py exists
if not exist main.py (
    echo Error: main.py file not found
    echo Please make sure you're running this script from the correct directory
    pause
    exit /b 1
)

:: Run the Streamlit app
echo Starting Streamlit application...
streamlit run main.py
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to start Streamlit application
    pause
    exit /b 1
)

pause
