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

# Create and setup virtual environment
setup_venv() {
    echo "Setting up virtual environment..."
    
    # Check if Python is installed
    if ! command -v python3 &>/dev/null && command -v python &>/dev/null; then
        PYTHON_CMD="python"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python is not installed or not in PATH"
        exit 1
    fi
    
    echo "Using $PYTHON_CMD"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment. Please install Python venv package."
            exit 1
        fi
    else
        echo "Virtual environment already exists."
    fi
    
    # Activate the virtual environment based on OS
    if [ "$OS" = "Windows" ]; then
        # For Windows
        echo "Activating virtual environment (Windows)..."
        source venv/Scripts/activate
    else
        # For Linux/macOS
        echo "Activating virtual environment (Unix)..."
        source venv/bin/activate
    fi
    
    if [ $? -ne 0 ]; then
        echo "Failed to activate virtual environment."
        exit 1
    fi
    
    # Install requirements
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements."
        exit 1
    fi
    
    echo "Installation completed successfully."
}

# Main execution
detect_os
setup_venv

echo ""
echo "Setup complete! To run the application:"
echo ""

if [ "$OS" = "Windows" ]; then
    echo "1. Activate the virtual environment: venv\\Scripts\\activate"
    echo "2. Run the application: streamlit run main.py"
    echo "   or use: ./run.sh"
else
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the application: streamlit run main.py"  
    echo "   or use: ./run.sh"
fi
