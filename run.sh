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
    
    # Run the Streamlit app
    echo "Starting Streamlit application..."
    streamlit run main.py
}

# Main execution
detect_os
run_app
