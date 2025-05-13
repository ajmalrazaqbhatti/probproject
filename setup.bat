@echo off
echo Setting up virtual environment for Windows...

:: Check if Python is installed
where python > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and ensure it's added to your PATH
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create virtual environment
        echo Please make sure you have the venv module installed
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Install requirements
echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install requirements
    echo Please check if requirements.txt exists and is valid
    pause
    exit /b 1
)

echo.
echo Setup complete! To run the application:
echo.
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: streamlit run main.py
echo    or use: run.bat
echo.
echo You can now close this window and run the application
pause
