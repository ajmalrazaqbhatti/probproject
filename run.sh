#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Run the Streamlit app
streamlit run main.py
