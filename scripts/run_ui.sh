#!/bin/bash

# Script to run the LSTM Frequency Extraction UI

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install streamlit plotly pandas
fi

echo "Starting LSTM Frequency Extraction Dashboard..."
echo "Press Ctrl+C to stop."

# Run the Streamlit app
streamlit run src/ui/dashboard.py
