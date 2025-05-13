#!/bin/bash

# Function to detect the operating system
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS="Linux";;
        Darwin*)    OS="macOS";;
        CYGWIN*)    OS="Windows";;
        MINGW*)     OS="Windows";;
        MSYS*)      OS="Windows";;
        *)          OS="Unknown";;
    esac
    echo "Detected operating system: $OS"
}

# Function to check if streamlit is installed
check_streamlit() {
    echo "Checking if Streamlit is installed..."
    if ! command -v streamlit &>/dev/null; then
        echo "Error: Streamlit is not installed in the virtual environment."
        echo "Please run setup.sh first to create the environment and install dependencies."
        exit 1
    fi
}

# Function to run the application
run_app() {
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Virtual environment not found. Please run setup.sh first."
        exit 1
    fi
    
    # Activate the virtual environment based on OS
    if [ "$OS" = "Windows" ]; then
        echo "Activating virtual environment (Windows)..."
        source venv/Scripts/activate
    else
        echo "Activating virtual environment (Unix)..."
        source venv/bin/activate
    fi
    
    if [ $? -ne 0 ]; then
        echo "Failed to activate virtual environment."
        echo "Please run setup.sh first to create the environment."
        exit 1
    fi
    
    # Check if Streamlit is installed
    check_streamlit
    
    # Run the Streamlit app
    echo "Starting Streamlit application..."
    streamlit run main.py
    
    if [ $? -ne 0 ]; then
        echo "Failed to start Streamlit application."
        echo "Please make sure main.py exists and is valid."
        exit 1
    fi
}

# Main execution
detect_os
run_app
