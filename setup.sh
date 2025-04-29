#!/bin/bash

# Make sure python3-full and python3-venv are installed
echo "Checking if required packages are installed..."
if ! dpkg -l | grep -q python3-full; then
    echo "Installing python3-full..."
    sudo apt update
    sudo apt install -y python3-full
fi

if ! dpkg -l | grep -q python3-venv; then
    echo "Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the application, use:"
echo "streamlit run main.py"
