@echo off
echo Setting up virtual environment for Windows...

:: Check if Python is installed
where python > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! To run the application:
echo.
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: streamlit run main.py
echo    or use: run.bat
